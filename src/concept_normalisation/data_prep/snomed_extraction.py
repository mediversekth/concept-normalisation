"""
Extract short terms (FSN + synonyms) and text definitions from a SNOMED CT
RF2 release snapshot. 
"""

import pandas as pd

from concept_normalisation.config import (
    CONCEPT_FILE, DESCRIPTION_FILE, DEFINITION_FILE, RELATIONSHIP_FILE,
    FSN_TYPE_ID, SYNONYM_TYPE_ID, IS_A_TYPE_ID,
    SHORT_TERMS_PARQUET, DEFINITIONS_PARQUET, ISA_RELATIONSHIPS_PARQUET,
)


class SnomedExtractor:
    def __init__(
        self,
        concept_file=CONCEPT_FILE,
        description_file=DESCRIPTION_FILE,
        definition_file=DEFINITION_FILE,
        relationship_file=RELATIONSHIP_FILE,
        fsn_type_id=FSN_TYPE_ID,
        synonym_type_id=SYNONYM_TYPE_ID,
        is_a_type_id=IS_A_TYPE_ID,
        short_terms_out=SHORT_TERMS_PARQUET,
        definitions_out=DEFINITIONS_PARQUET,
        isa_relationships_out=ISA_RELATIONSHIPS_PARQUET,
    ):
        self.concept_file = concept_file
        self.description_file = description_file
        self.definition_file = definition_file
        self.relationship_file = relationship_file
        self.fsn_type_id = fsn_type_id
        self.synonym_type_id = synonym_type_id
        self.is_a_type_id = is_a_type_id
        self.short_terms_out = short_terms_out
        self.definitions_out = definitions_out
        self.isa_relationships_out = isa_relationships_out

        self.active_concept_ids = None
        self.short_terms = None
        self.definitions = None
        self.isa_relationships = None

    def load_concepts(self):
        concepts = pd.read_csv(self.concept_file, sep="\t", dtype=str)
        active_concepts = concepts.loc[concepts["active"] == "1", "id"]
        self.active_concept_ids = set(active_concepts)

        print(f"Active concepts: {len(self.active_concept_ids):,}")
        return self.active_concept_ids

    def extract_short_terms(self):
        if self.active_concept_ids is None:
            self.load_concepts()

        # Short term FSN + synonym descriptions
        descriptions = pd.read_csv(self.description_file, sep="\t", dtype=str)
        descriptions = descriptions[
            (descriptions["active"] == "1")
            & (descriptions["conceptId"].isin(self.active_concept_ids))
            & (descriptions["typeId"].isin([self.fsn_type_id, self.synonym_type_id]))
        ].copy()

        descriptions["description_type"] = descriptions["typeId"].map({
            self.fsn_type_id: "FSN",
            self.synonym_type_id: "Synonym",
        })

        short_terms = descriptions[["conceptId", "term", "description_type"]].rename(
            columns={"conceptId": "concept_id", "term": "description"}
        ).drop_duplicates()

        # Remove rows that cannot be embedded
        short_terms = short_terms.dropna(
            subset=["concept_id", "description"]
        )
        # Remove empty/whitespace-only descriptions
        short_terms = short_terms[
            short_terms["description"].str.strip().ne("")
        ].reset_index(drop=True)

        short_terms.to_parquet(self.short_terms_out)
        print(f"Short terms (FSN+synonyms): {len(short_terms):,} rows")

        self.short_terms = short_terms
        return short_terms

    def extract_definitions(self):
        if self.active_concept_ids is None:
            self.load_concepts()

        if self.definition_file.exists():
            definitions_raw = pd.read_csv(self.definition_file, sep="\t", dtype=str)
            definitions_raw = definitions_raw[
                (definitions_raw["active"] == "1")
                & (definitions_raw["conceptId"].isin(self.active_concept_ids))
            ]
            definitions = definitions_raw[["conceptId", "term"]].rename(
                columns={"conceptId": "concept_id", "term": "definition"}
            ).drop_duplicates()

            # Remove rows that cannot be embedded
            definitions = definitions.dropna(
                subset=["concept_id", "definition"]
            )
            # Remove empty/whitespace-only descriptions
            definitions = definitions[
                definitions["definition"].str.strip().ne("")
            ].reset_index(drop=True)
        else:
            print(f"No TextDefinition file found at {self.definition_file} -- check the path.")
            definitions = pd.DataFrame(columns=["concept_id", "definition"])

        definitions.to_parquet(self.definitions_out)
        print(f"Text definitions: {len(definitions):,} rows")

        # Sanity check
        if len(definitions) > 0:
            print("\nSample definitions: ")
            print(definitions.sample(min(4, len(definitions)))["definition"].to_list())

        self.definitions = definitions
        return definitions

    def extract_isa_relationships(self) -> pd.DataFrame:
        """Pull every active "Is a" relationship (typeId 116680003) between
        active concepts out of the RF2 Relationship file -- one row per
        edge: source_id (the more specific/child concept) -> destination_id
        (its direct parent). This is what hierarchy.py's SnomedHierarchy
        walks to check whether two different matched concepts are related
        in the ontology, even when they're not the exact same concept_id."""
        if self.active_concept_ids is None:
            self.load_concepts()

        relationships = pd.read_csv(self.relationship_file, sep="\t", dtype=str)
        relationships = relationships[
            (relationships["active"] == "1")
            & (relationships["typeId"] == self.is_a_type_id)
            & (relationships["sourceId"].isin(self.active_concept_ids))
            & (relationships["destinationId"].isin(self.active_concept_ids))
        ]

        isa = relationships.rename(
            columns={"sourceId": "source_id", "destinationId": "destination_id"}
        )[["source_id", "destination_id"]].drop_duplicates()

        isa.to_parquet(self.isa_relationships_out)
        print(f"Is-a relationships: {len(isa):,} rows")

        self.isa_relationships = isa
        return isa

    def build_bm25_documents(self) -> pd.DataFrame:
        """Same prep as SNOMED_linking.ipynb cells 4-9: descriptions (all
        types, not just FSN/synonym) attached to active concepts, with
        definitions left-joined on, ready for ElasticBM25Index.bulk_load().
        Kept separate from extract_short_terms() because it keeps typeId and
        every description type, unlike the FSN/synonym-only short terms."""
        concepts = pd.read_csv(self.concept_file, sep="\t", dtype=str)
        concepts = concepts[concepts["active"] == "1"]
        print(f"Active concepts: {len(concepts):,}")

        descriptions = pd.read_csv(self.description_file, sep="\t", dtype=str)
        descriptions = descriptions[descriptions["active"] == "1"]
        print(f"Active descriptions: {len(descriptions):,}")

        # Keep only descriptions whose concept is active
        snomed = descriptions.merge(
            concepts[["id"]],
            left_on="conceptId",
            right_on="id",
            how="inner"
        )
        print(f"Descriptions attached to active concepts: {len(snomed)}")

        definitions = pd.read_csv(self.definition_file, sep="\t", dtype=str)
        definitions = definitions[definitions["active"] == "1"]
        definitions = definitions.rename(columns={"term": "definition"})

        snomed = snomed.merge(
            definitions[["conceptId", "definition"]],
            on="conceptId",
            how="left"
        )
        print(snomed["definition"].notna().sum())

        # Removing the concepts where the term doesn't exist
        before = len(snomed)
        snomed = snomed.dropna(subset=["term"])
        after = len(snomed)
        print(f"Removed {before - after} rows")

        return snomed

    def run(self):
        """Runs the full extraction (concepts -> short terms + definitions),
        same order as extract_context.py top to bottom."""
        self.load_concepts()
        self.extract_short_terms()
        self.extract_definitions()
        return self.short_terms, self.definitions
