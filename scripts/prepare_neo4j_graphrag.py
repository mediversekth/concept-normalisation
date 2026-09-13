#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from typing import Iterable

import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from concept_normalisation import config
from concept_normalisation import utils


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

LOG = logging.getLogger("prepare_snomed_graphrag")

SEARCHABLE_DESCRIPTION_TYPE_IDS = [config.FSN_TYPE_ID, config.SYNONYM_TYPE_ID]


# -----------------------------------------------------------------------------
# CLI and logging
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse the only run-specific command-line option."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an existing SNOMED CT Neo4j database for GraphRAG by "
            "adding searchable text, embeddings, and Neo4j search indexes."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings even when they already exist.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure application logging and quiet noisy third-party libraries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    noisy_loggers = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "neo4j.notifications",
    )

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# RF2 TextDefinition handling
# -----------------------------------------------------------------------------


def clean_text(value: str | None) -> str:
    """Collapse repeated whitespace while preserving readable medical text."""
    return " ".join((value or "").split()).strip()


def load_textdefinitions(path) -> dict[str, list[str]]:
    """
    Load active English TextDefinitions grouped by conceptId.

    For RF2 Full releases, first select the latest effectiveTime for each
    component ID, then apply the active filter. This reproduces current-state
    semantics rather than keeping historical active rows.
    """
    latest_by_component: dict[str, dict[str, str]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        # Filter out rows that are missing required columns before checking for duplicates
        required_columns = {
            "id",
            "effectiveTime",
            "active",
            "conceptId",
            "languageCode",
            "term",
        }

        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"TextDefinition file is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if any(part.casefold() == "full" for part in path.parts):
            # For RF2 Full releases, select the latest effectiveTime for each component ID
            LOG.info(
                "RF2 Full detected: selecting the latest effectiveTime "
                "for each TextDefinition before filtering active rows."
            )

            for row in reader:
                component_id = row["id"]
                previous = latest_by_component.get(component_id)

                if (
                    previous is None
                    or row["effectiveTime"] >= previous["effectiveTime"]
                ):
                    latest_by_component[component_id] = row

            rows: Iterable[dict[str, str]] = latest_by_component.values()
        else:
            # For RF2 Snapshot releases, use all rows as-is
            rows = reader

        # Build a mapping of conceptId -> list of active English TextDefinition terms
        definitions_by_concept: defaultdict[str, list[str]] = defaultdict(list)
        seen_terms: defaultdict[str, set[str]] = defaultdict(set)

        for row in rows:
            # Has to be active and English
            if row["active"] != "1":
                continue
            if row.get("languageCode", "").casefold() != "en":
                continue

            concept_id = row["conceptId"]
            term = clean_text(row["term"])
            normalized_term = term.casefold()

            # Skip duplicate terms for the same conceptId, which can occur in RF2 Full releases
            if not concept_id or not term:
                continue
            if normalized_term in seen_terms[concept_id]:
                continue

            seen_terms[concept_id].add(normalized_term)
            definitions_by_concept[concept_id].append(term)

    return dict(definitions_by_concept)


# -----------------------------------------------------------------------------
# Embedding model
# -----------------------------------------------------------------------------


def load_embedding_model(
    model_name: str,
    device: str,
) -> tuple[SentenceTransformer, int]:
    """Load the SentenceTransformer model and determine its vector dimension."""
    LOG.info("Loading %s on %s", model_name, device)

    model = SentenceTransformer(model_name, device=device)

    dimensions = model.get_embedding_dimension()

    if dimensions is None:
        probe_embedding = model.encode(
            ["probe"],
            show_progress_bar=False,
        )
        dimensions = int(probe_embedding.shape[1])

    return model, int(dimensions)


def create_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    """Embed text using the batch size configured for GraphRAG preparation."""
    embeddings = model.encode(
        texts,
        batch_size=config.GRAPHRAG_EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [embedding.astype("float32").tolist() for embedding in embeddings]


# -----------------------------------------------------------------------------
# Neo4j concept retrieval and enrichment
# -----------------------------------------------------------------------------


def count_concepts_to_process(session, force: bool) -> int:
    """Count active ObjectConcept nodes that still need processing."""
    embedding_filter = "" if force else "AND c.embedding IS NULL"

    query = f"""
    MATCH (c:ObjectConcept)
    WHERE (c.active = true OR toString(c.active) = '1')
      {embedding_filter}
    RETURN count(c) AS count
    """

    result = session.run(query).single()
    return int(result["count"])


def fetch_concept_batch(
    session,
    after_sctid: str,
    batch_size: int,
    force: bool,
) -> list[dict]:
    """Fetch one keyset-paginated batch of concepts and active descriptions."""
    embedding_filter = "" if force else "AND c.embedding IS NULL"

    query = f"""
    MATCH (c:ObjectConcept)
    WHERE (c.active = true OR toString(c.active) = '1')
      AND toString(c.sctid) > $after_sctid
      {embedding_filter}

    WITH c
    ORDER BY toString(c.sctid)
    LIMIT $batch_size

    OPTIONAL MATCH (c)-[:HAS_DESCRIPTION]->(d:Description)
    WHERE (d.active = true OR toString(d.active) = '1')
      AND toLower(coalesce(d.languageCode, '')) = 'en'
      AND toString(d.typeId) IN $description_type_ids

    RETURN
        toString(c.sctid) AS sctid,
        elementId(c) AS node_id,
        coalesce(c.FSN, '') AS fsn,
        [term IN collect(DISTINCT d.term) WHERE term IS NOT NULL] AS descriptions
    ORDER BY sctid
    """

    result = session.run(
        query,
        after_sctid=after_sctid,
        batch_size=batch_size,
        description_type_ids=SEARCHABLE_DESCRIPTION_TYPE_IDS,
    )

    return [record.data() for record in result]


def build_search_text(
    fsn: str,
    descriptions: Iterable[str],
    textdefinitions: Iterable[str] | None,
) -> tuple[str, str]:
    """
    Build the two text representations stored on ObjectConcept.

    search_terms:
        FSN + synonyms/descriptions, used for full-text retrieval.

    embedding_text:
        FSN + synonyms/descriptions + optional TextDefinition, used to produce
        the semantic embedding.
    """
    fsn = clean_text(fsn)
    fsn_key = fsn.casefold()

    unique_synonyms: dict[str, str] = {}

    for raw_term in descriptions:
        term = clean_text(raw_term)
        normalized_term = term.casefold()

        if not term or normalized_term == fsn_key:
            continue

        unique_synonyms.setdefault(normalized_term, term)

    synonyms = sorted(unique_synonyms.values(), key=str.casefold)

    searchable_terms = ([fsn] if fsn else []) + synonyms
    search_terms = " | ".join(searchable_terms)

    embedding_parts: list[str] = []

    if fsn:
        embedding_parts.append(f"FSN: {fsn}")

    if synonyms:
        embedding_parts.append(f"Synonyms: {'; '.join(synonyms)}")

    definitions = [
        cleaned
        for definition in (textdefinitions or [])
        if (cleaned := clean_text(definition))
    ]

    if definitions:
        embedding_parts.append(f"Definition: {' '.join(definitions)}")

    if not embedding_parts and search_terms:
        readable_terms = search_terms.replace(" | ", "; ")
        embedding_parts.append(f"Terms: {readable_terms}")

    embedding_text = "\n".join(embedding_parts)
    return search_terms, embedding_text


def chunked(items: list, chunk_size: int):
    """Yield fixed-size slices from a list."""
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def write_concept_rows(
    session,
    rows: list[dict],
    write_batch_size: int,
) -> int:
    """Write enriched concept properties back to Neo4j in batches."""
    query = """
    UNWIND $rows AS row

    MATCH (c:ObjectConcept)
    WHERE elementId(c) = row.node_id

    SET
        c.search_terms = row.search_terms,
        c.embedding_text = row.embedding_text,
        c.embedding = row.embedding,
        c.embedding_model = row.embedding_model,
        c.has_text_definition = row.has_text_definition

    RETURN count(c) AS updated
    """

    updated_total = 0

    for batch in chunked(rows, write_batch_size):
        result = session.run(query, rows=batch).single()
        updated_total += int(result["updated"])

    return updated_total


def enrich_concepts(
    session,
    model: SentenceTransformer,
    model_name: str,
    textdefinitions: dict[str, list[str]],
    total: int,
    fetch_batch_size: int,
    write_batch_size: int,
    force: bool,
) -> int:
    """Create search text and embeddings for all eligible ObjectConcept nodes."""
    after_sctid = ""
    total_updated = 0

    with logging_redirect_tqdm():
        with tqdm(
            total=total,
            desc="Enriching ObjectConcepts",
            unit="concept",
            dynamic_ncols=True,
        ) as progress:
            while True:
                records = fetch_concept_batch(
                    session=session,
                    after_sctid=after_sctid,
                    batch_size=fetch_batch_size,
                    force=force,
                )

                if not records:
                    break

                prepared_rows: list[dict] = []
                texts_to_embed: list[str] = []

                for record in records:
                    sctid = record["sctid"]
                    definitions = textdefinitions.get(sctid, [])

                    search_terms, embedding_text = build_search_text(
                        fsn=record["fsn"],
                        descriptions=record["descriptions"],
                        textdefinitions=definitions,
                    )

                    if not embedding_text:
                        LOG.warning("Skipping %s: no searchable text found.", sctid)
                        continue

                    texts_to_embed.append(embedding_text)
                    prepared_rows.append(
                        {
                            "node_id": record["node_id"],
                            "search_terms": search_terms,
                            "embedding_text": embedding_text,
                            "embedding_model": model_name,
                            "has_text_definition": bool(definitions),
                        }
                    )

                if prepared_rows:
                    embeddings = create_embeddings(model, texts_to_embed)

                    for row, embedding in zip(prepared_rows, embeddings):
                        row["embedding"] = embedding

                    total_updated += write_concept_rows(
                        session=session,
                        rows=prepared_rows,
                        write_batch_size=write_batch_size,
                    )

                after_sctid = records[-1]["sctid"]
                progress.update(len(records))

    return total_updated


# -----------------------------------------------------------------------------
# Neo4j indexes
# -----------------------------------------------------------------------------


def create_indexes(session, dimensions: int) -> None:
    """Create the full-text and vector indexes required for hybrid retrieval."""
    LOG.info("Creating full-text index: %s", config.GRAPHRAG_FULLTEXT_INDEX_NAME)

    session.run(
        f"""
        CREATE FULLTEXT INDEX {config.GRAPHRAG_FULLTEXT_INDEX_NAME} IF NOT EXISTS
        FOR (c:ObjectConcept)
        ON EACH [c.search_terms]
        """
    ).consume()

    LOG.info("Creating vector index: %s", config.GRAPHRAG_VECTOR_INDEX_NAME)

    session.run(
        f"""
        CREATE VECTOR INDEX {config.GRAPHRAG_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:ObjectConcept)
        ON (c.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: $dimensions,
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        dimensions=dimensions,
    ).consume()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    args = parse_args()

    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()

        with driver.session(database=config.NEO4J_DATABASE) as session:
            LOG.info("TextDefinition file: %s", config.DEFINITION_FILE)
            textdefinitions = load_textdefinitions(config.DEFINITION_FILE)
            LOG.info(
                "Loaded active current TextDefinitions for %d concepts.",
                len(textdefinitions),
            )

            device = utils.get_device()
            model, dimensions = load_embedding_model(
                model_name=config.GRAPHRAG_EMBEDDING_MODEL_NAME,
                device=device,
            )

            total = count_concepts_to_process(session=session, force=args.force)

            if total > 0:
                LOG.info(
                    "Processing %d concepts | fetch=%d | encode=%d | write=%d",
                    total,
                    config.GRAPHRAG_FETCH_BATCH_SIZE,
                    config.GRAPHRAG_EMBEDDING_BATCH_SIZE,
                    config.GRAPHRAG_WRITE_BATCH_SIZE,
                )

                updated = enrich_concepts(
                    session=session,
                    model=model,
                    model_name=config.GRAPHRAG_EMBEDDING_MODEL_NAME,
                    textdefinitions=textdefinitions,
                    total=total,
                    fetch_batch_size=config.GRAPHRAG_FETCH_BATCH_SIZE,
                    write_batch_size=config.GRAPHRAG_WRITE_BATCH_SIZE,
                    force=args.force,
                )
                LOG.info("Updated %d ObjectConcept nodes.", updated)
            else:
                LOG.info(
                    "No concepts need embedding. Use --force to rebuild "
                    "existing embeddings."
                )

            create_indexes(session, dimensions)
            LOG.info("SNOMED GraphRAG preparation complete.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
