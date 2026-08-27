"""
Environment configuration: filesystem paths, SNOMED CT identifiers,
model names/batch sizes, and external service settings.

Experiment settings (which table, which methods, top_k, ...) are NOT
here -- those live in `pipeline.run.ExperimentConfig`, since they change
per run rather than per environment.
"""

import os
from pathlib import Path


# ============================================================
# Project directories
# ============================================================

# Where the raw MIMIC-IV / SNOMED CT files live, and where every prepared
# artifact the pipeline produces gets written. Defaults to ./data in
# whatever directory the pipeline is run from; override with the
# CONCEPT_NORM_DATA_DIR environment variable to point at a fixed location
# (e.g. `export CONCEPT_NORM_DATA_DIR=/data`) regardless of cwd.
DATA_DIR = Path(os.environ.get("CONCEPT_NORM_DATA_DIR", "data")).resolve()
OUTPUT_DIR = DATA_DIR / "output"

# Created on import so every module that writes here can assume it exists.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Raw input data
# ============================================================

# The table currently being mapped to SNOMED CT. Change this one line to
# point the whole pipeline at a different table -- nothing else needs
# editing.
ACTIVE_TABLE = DATA_DIR / "diagnosis_icd10_snomed.csv"

MICRO_CSV_PATH = (
    DATA_DIR
    / "mimic-iv-clinical-database-demo-2.2"
    / "hosp"
    / "microbiologyevents.csv.gz"
)


# ============================================================
# SNOMED CT release files
# ============================================================

SNOMED_ROOT_DIR = (
    DATA_DIR
    / "SnomedCT_InternationalRF2_PRODUCTION_20260601T120000Z"
)

SNOMED_TERMINOLOGY_DIR = SNOMED_ROOT_DIR / "Snapshot" / "Terminology"

CONCEPT_FILE = SNOMED_TERMINOLOGY_DIR / "sct2_Concept_Snapshot_INT_20260601.txt"
DESCRIPTION_FILE = SNOMED_TERMINOLOGY_DIR / "sct2_Description_Snapshot-en_INT_20260601.txt"
DEFINITION_FILE = SNOMED_TERMINOLOGY_DIR / "sct2_TextDefinition_Snapshot-en_INT_20260601.txt"
RELATIONSHIP_FILE = SNOMED_TERMINOLOGY_DIR / "sct2_Relationship_Snapshot_INT_20260601.txt"


# ============================================================
# Prepared SNOMED data (built once by SnomedExtractor, then reused)
# ============================================================

SHORT_TERMS_PARQUET = OUTPUT_DIR / "concepts_short_terms.parquet"
DEFINITIONS_PARQUET = OUTPUT_DIR / "concepts_definitions.parquet"
ISA_RELATIONSHIPS_PARQUET = OUTPUT_DIR / "concepts_isa_relationships.parquet"
JACCARD_NGRAM_INDEX_PARQUET = OUTPUT_DIR / "jaccard_ngram_index.parquet"


# ============================================================
# Candidate embeddings
# ============================================================

SHORT_TERMS_SAPBERT_EMB = OUTPUT_DIR / "short_terms_sapbert_embeddings.npy"
SHORT_TERMS_BIOLORD_EMB = OUTPUT_DIR / "short_terms_biolord_embeddings.npy"
DEFINITIONS_BIOLORD_EMB = OUTPUT_DIR / "definitions_biolord_embeddings.npy"

SAPBERT_CHUNK_DIR = OUTPUT_DIR / "sapbert_chunks"
BIOLORD_CHUNK_DIR = OUTPUT_DIR / "biolord_chunks"
BIOLORD_SHORT_TERMS_CHUNK_DIR = OUTPUT_DIR / "biolord_short_terms_chunks"


# ============================================================
# Pipeline result files
# ============================================================

SEMANTIC_RESULTS_PARQUET = OUTPUT_DIR / "semantic_data_mapped_full.parquet"
ALGORITHM_1_RESULTS_PARQUET = OUTPUT_DIR / "algorithm_1_results.parquet"
ALGORITHM_2_RESULTS_PARQUET = OUTPUT_DIR / "algorithm_2_results.parquet"
ALGORITHM_AI_RESULTS_PARQUET = OUTPUT_DIR / "algorithm_ai_results.parquet"
COMPARISON_TABLE_PARQUET = OUTPUT_DIR / "method_comparison.parquet"
FINAL_CANDIDATES_PARQUET = OUTPUT_DIR / "final_candidates.parquet"


# ============================================================
# Fixed SNOMED CT identifiers
# ============================================================
# Part of the SNOMED CT standard itself, not something a user of this
# pipeline would ever need to change.

FSN_TYPE_ID = "900000000000003001"
SYNONYM_TYPE_ID = "900000000000013009"
IS_A_TYPE_ID = "116680003"


# ============================================================
# Models
# ============================================================

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BIOLORD_MODEL_NAME = "FremyCompany/BioLORD-2023"
OLLAMA_MODEL_NAME = "llama3.1"


# ============================================================
# Batch sizes
# ============================================================

DEFAULT_SAPBERT_BATCH_SIZE = 32
DEFAULT_BIOLORD_BATCH_SIZE = 64
DEFAULT_QUERY_BATCH_SIZE = 50


# ============================================================
# Matching defaults
# ============================================================

DEFAULT_TOP_K = 5


# ============================================================
# Text preprocessing defaults (syntactic matching)
# ============================================================

PREPROCESS_LOWERCASE = True
PREPROCESS_REMOVE_PUNCTUATION = True
PREPROCESS_REMOVE_STOPWORDS = False


# ============================================================
# External services
# ============================================================
# Elasticsearch is required for the multi-match and fuzzy syntactic
# methods. Neo4j is only used by the optional graph-backed hierarchy
# (ranking.hierarchy.Neo4jHierarchy) -- the default consensus step uses
# SnomedHierarchy instead, which needs no database at all.

ELASTIC_URL = os.environ.get("CONCEPT_NORM_ELASTIC_URL", "http://localhost:9200")
ELASTIC_INDEX_NAME = "snomed"

NEO4J_URI = os.environ.get("CONCEPT_NORM_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("CONCEPT_NORM_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("CONCEPT_NORM_NEO4J_PASSWORD", "")

NEO4J_CONCEPT_LABEL = "ObjectConcept"
NEO4J_CONCEPT_ID_PROP = "sctid"
NEO4J_ISA_REL_TYPE = "ISA"
