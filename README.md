# Toolkit for concept normalisation 

One of the processes in a data harmonisation pipeline and currently a bottleneck, concept normalisation or entity linking solves mapping a local code or string to a concept in a terminology system (e.g., SNOMED, LOINC). 

# concept-normalisation

Uncertainty-aware clinical concept normalisation: maps free-text clinical
concepts (e.g. MIMIC-IV diagnosis strings) to SNOMED CT using several
independent methods, then reconciles them with a hierarchy-aware
consensus step.

Methods implemented:

**Semantic**
1. SapBERT — query text only
2. BioLORD — query + context columns
3. LLM (Ollama) generated context + BioLORD

**Syntactic**
4. Elasticsearch multi-match (BM25)
5. Elasticsearch fuzzy matching
6. Character n-gram Jaccard similarity (no external services needed)

## Install

```bash
pip install -e .
```

This installs the `concept_normalisation` package.

## Data layout

The pipeline expects a `data/` directory (by default, next to wherever you
run it from) containing:

```
data/
  diagnosis_icd10_snomed.csv                 # your input table
  SnomedCT_InternationalRF2_.../              # a SNOMED CT RF2 release
  output/                                     # created automatically
```

To point at a data directory somewhere else (e.g. `/data`), set:

```bash
export CONCEPT_NORM_DATA_DIR=/data
```

Elasticsearch and Neo4j URLs default to localhost; override with
`CONCEPT_NORM_ELASTIC_URL`, `CONCEPT_NORM_NEO4J_URI`,
`CONCEPT_NORM_NEO4J_USER`, `CONCEPT_NORM_NEO4J_PASSWORD` if needed.

## First run: build the SNOMED candidate data

Before running the pipeline, the SNOMED RF2 release needs to be extracted
into the parquet files the matchers read, and those candidates need to be
embedded:

```python
from concept_normalisation.data_prep.snomed_extraction import SnomedExtractor
from concept_normalisation.semantic_matching.embedding.sapbert_embedder import SapBertEmbedder
from concept_normalisation.semantic_matching.embedding.biolord_embedder import BioLordEmbedder
from concept_normalisation import config

extractor = SnomedExtractor()
extractor.run()  # writes short terms + definitions parquet
extractor.extract_isa_relationships()  # writes the is-a hierarchy parquet

SapBertEmbedder().embed_parquet_chunked(config.SHORT_TERMS_PARQUET)

biolord = BioLordEmbedder()
biolord.embed_short_terms(config.SHORT_TERMS_PARQUET)
biolord.embed_definitions(config.DEFINITIONS_PARQUET)
```

This is a one-off step per SNOMED release — the parquet/embedding files it
produces are reused by every pipeline run after that.

## Running the pipeline

**As a script**, using the settings in `main.py`:

```bash
python main.py
```

**From Python**, for full control over every setting:

```python
from concept_normalisation.pipeline.run import ExperimentConfig, run_pipeline

cfg = ExperimentConfig(
    query_column="diagnosis_text",
    context_columns=[],
    top_k=5,
    run_semantic_algorithm_ai=False,   # skip if Ollama isn't running
    run_syntactic_multi_match=False,   # skip if Elasticsearch isn't running
    run_syntactic_fuzzy=False,
)
result = run_pipeline(cfg)
```

Every stage checkpoints to `data/output/<table>_pipeline_checkpoint.parquet`
after it finishes. Re-running resumes from the last completed stage
instead of starting over; pass `fresh_start=True` in `ExperimentConfig`
to force a full rerun.

## Outputs

All written to `data/output/`:

| File | Contents |
| --- | --- |
| `semantic_data_mapped_full.parquet` | every row with all methods' matches |
| `method_comparison.parquet` | one row per query, top match per method side by side |
| `final_candidates.parquet` | hierarchy-aware consensus candidates |

## Package layout

```
src/concept_normalisation/
  config/             environment settings: paths, SNOMED IDs, models, services
  pipeline/
    run.py            ExperimentConfig + run_pipeline() — the whole experiment
    preparation.py    build the semantic input columns
    semantic.py        SapBERT / BioLORD / AI-context matching
    syntactic.py        Elasticsearch + Jaccard matching
    evaluation.py       comparison table + consensus
  data_prep/          SNOMED extraction, context building, AI-context generation
  semantic_matching/  dense index + embedders
  syntactic_matching/ Elasticsearch index, fuzzy matcher, text preprocessing
  ranking/            SNOMED hierarchy, consensus, comparison table
```

Experiment settings (which table, which methods, top_k, ...) live in
`ExperimentConfig`. Environment settings (paths, model names, service
URLs) live in `config/`.
