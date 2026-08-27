from pathlib import Path

import numpy as np
import pandas as pd

from concept_normalisation.data_prep.ai_context import (
    generate_ai_contexts,
    metadata_key,
)
from concept_normalisation.semantic_matching.dense_index import (
    DenseIndex,
)
from concept_normalisation.semantic_matching.embedding.biolord_embedder import (
    BioLordEmbedder,
)
from concept_normalisation.semantic_matching.embedding.sapbert_embedder import (
    SapBertEmbedder,
)


def run_algorithm_1(
    data: pd.DataFrame,
    embedder: SapBertEmbedder,
    candidate_index: DenseIndex,
    text_column: str = "algorithm_1_text",
    output_column: str = "algorithm_1_matches",
    top_k: int = 5,
    search_batch_size: int = 50,
) -> pd.DataFrame:
    """
    Semantic Algorithm 1:
    query column only -> SapBERT -> dense SNOMED search.
    """

    result = data.copy()

    texts = (
        result[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    query_embeddings = np.vstack(
        [embedder.embed_single(text) for text in texts]
    )

    result[output_column] = candidate_index.match_batch(
        query_embeddings=query_embeddings,
        top_k=top_k,
        query_batch_size=search_batch_size,
    )

    return result


def run_algorithm_2(
    data: pd.DataFrame,
    embedder: BioLordEmbedder,
    candidate_index: DenseIndex,
    text_column: str = "algorithm_2_text",
    output_column: str = "algorithm_2_matches",
    top_k: int = 5,
    embedding_batch_size: int | None = None,
    search_batch_size: int = 50,
) -> pd.DataFrame:
    """
    Semantic Algorithm 2:
    query + selected context columns -> BioLORD -> dense SNOMED search.
    """

    result = data.copy()

    texts = (
        result[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    query_embeddings = embedder.encode(
        texts,
        batch_size=embedding_batch_size,
    )

    result[output_column] = candidate_index.match_batch(
        query_embeddings=query_embeddings,
        top_k=top_k,
        query_batch_size=search_batch_size,
        extra_columns=["source"],
    )

    return result


def generate_algorithm_ai_text(
    data: pd.DataFrame,
    metadata_column: str = "algorithm_ai_metadata",
    output_column: str = "algorithm_ai_text",
    model: str = "llama3.1",
    progress_every: int = 50,
    checkpoint_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Generate the AI representation only.

    Deduplicates identical (table_name, query_column, query_value,
    context_values) combinations before calling the model -- on tables
    where the same query recurs across many rows (e.g. the same
    diagnosisstring appearing for many different patients in
    diagnosis_icd9_snomed), each unique combination is generated once and
    copied to every row that shares it, instead of paying for the same
    Ollama call over and over.

    Pass checkpoint_path to make the run resumable -- see
    generate_ai_contexts.

    This is separate so the AI text can be inspected before embedding.
    """

    result = data.copy()

    metadata_series = result[metadata_column]
    keys = metadata_series.apply(metadata_key)

    unique_keys = keys.drop_duplicates()
    unique_metadata = metadata_series.loc[unique_keys.index].tolist()

    n_total = len(result)
    n_unique = len(unique_metadata)

    if n_unique < n_total:
        print(
            f"{n_total:,} rows -> {n_unique:,} unique AI prompts "
            f"({n_total - n_unique:,} duplicate rows skipped)"
        )

    unique_outputs = generate_ai_contexts(
        metadata_items=unique_metadata,
        model=model,
        progress_every=progress_every,
        checkpoint_path=checkpoint_path,
    )

    key_to_output = dict(zip(unique_keys.tolist(), unique_outputs))
    result[output_column] = keys.map(key_to_output)

    return result


def match_algorithm_ai(
    data: pd.DataFrame,
    embedder: BioLordEmbedder,
    candidate_index: DenseIndex,
    text_column: str = "algorithm_ai_text",
    output_column: str = "algorithm_ai_matches",
    top_k: int = 5,
    embedding_batch_size: int | None = None,
    search_batch_size: int = 50,
) -> pd.DataFrame:
    """
    Embed already-generated AI context and map it to SNOMED.
    """

    result = data.copy()

    texts = (
        result[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    query_embeddings = embedder.encode(
        texts,
        batch_size=embedding_batch_size,
    )

    result[output_column] = candidate_index.match_batch(
        query_embeddings=query_embeddings,
        top_k=top_k,
        query_batch_size=search_batch_size,
        extra_columns=["source"],
    )

    return result


def run_algorithm_ai(
    data: pd.DataFrame,
    embedder: BioLordEmbedder,
    candidate_index: DenseIndex,
    model: str = "llama3.1",
    top_k: int = 5,
    embedding_batch_size: int | None = None,
    search_batch_size: int = 50,
    progress_every: int = 50,
    checkpoint_path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Complete AI semantic workflow.
    """

    result = generate_algorithm_ai_text(
        data=data,
        model=model,
        progress_every=progress_every,
        checkpoint_path=checkpoint_path,
    )

    result = match_algorithm_ai(
        data=result,
        embedder=embedder,
        candidate_index=candidate_index,
        top_k=top_k,
        embedding_batch_size=embedding_batch_size,
        search_batch_size=search_batch_size,
    )

    return result


def run_ai_experiment_embeddings(
    ai_results,
    embedder,
    candidate_index,
    text_column="ai_output",
    output_column="snomed_matches",
    top_k=5,
    embedding_batch_size=64,
    search_batch_size=50,
):
    """
    Embed previously generated AI-context sentences with BioLORD
    and retrieve their top-k SNOMED candidates.

    This does NOT generate AI text.
    It only embeds and maps existing experimental outputs.
    """

    result = ai_results.copy()

    texts = (
        result[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        f"Embedding {len(texts):,} AI-generated sentences..."
    )

    embeddings = embedder.encode(
        texts,
        batch_size=embedding_batch_size,
        show_progress_bar=True,
    )

    result[output_column] = candidate_index.match_batch(
        query_embeddings=embeddings,
        top_k=top_k,
        query_batch_size=search_batch_size,
        extra_columns=["source"],
    )

    return result, embeddings
