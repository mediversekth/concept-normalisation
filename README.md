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

**Semantic + Syntactic**

7. GraphRAG

## Install

```bash
pip install -e .
```

This installs the `concept_normalisation` package.

You might have to manually install torch libraries to get correct GPU packages.

## Data layout

The pipeline expects a `data/` directory (by default, next to wherever you
run it from) containing:

```
data/
  diagnosis_icd10_snomed.csv                  # your input table
  SnomedCT_InternationalRF2_.../              # a SNOMED CT RF2 release
  output/                                     # created automatically
  neo4j-output/                               # Has to be created manually, for loading SNOMED data into neo4j
```

To point at a data directory somewhere else (e.g. `/data`), set:

```bash
export CONCEPT_NORM_DATA_DIR=/data
```

Elasticsearch and Neo4j URLs default to localhost; override with
`CONCEPT_NORM_ELASTIC_URL`, `CONCEPT_NORM_NEO4J_URI`,
`CONCEPT_NORM_NEO4J_USER`, `CONCEPT_NORM_NEO4J_PASSWORD` if needed.

## First run

### Requirements

- [Docker](https://docs.docker.com/desktop/)
- Docker Compose (Comes with Docker Desktop)
- [Python](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/download)

**NOTE:** Run following python scripts from repository root.

### build the SNOMED candidate data

Before running the pipeline, the SNOMED RF2 release needs to be extracted
into the parquet files the matchers read, and those candidates need to be
embedded:

```bash
python scripts\prepare_snomed.py
```

This is a one-off step per SNOMED release — the parquet/embedding files it
produces are reused by every pipeline run after that.

### Start Neo4J and ElasticSearch

Before following steps, we need to start Neo4J and ElasticSearch engine. There is already a configured docker compose file which can be started by running:

**NOTE:** Adjust the path under Neo4J volumes to match the location of your `neo4j-output` folder. This is an absolute path and it has to be the same both inside and outside the container. Note that some versions of docker compose struggle with colons, so if you are on a system that uses colons in the disk name or similar, you may need to remove that part.

```bash
docker compose up -d
```

The docker compose file is configured for persistent storage, so doing `docker compose down` will not remove data that has been loaded into neo4j or elasticsearch. One can simply rerun `docker compose up -d` if you want to run the Neo4J and ElasticSearch instances with all the data still there.

### Load data into ElasticSearch

```bash
python scripts\prepare_elastic.py
```

### Load data into Neo4J

The SNOMED loader for Neo4J exists [here](https://github.com/IHTSDO/snomed-database-loader/tree/master/).

This step requires you to go to the location where you cloned the repository then into the `NEO4J` folder. Subsequently, run the following command with the placeholders replaced with actual values:

```bash
python snomed_g_graphdb_build_tools.py db_build --action create --rf2 <rf2-release-directory> --release_type full --neopw <password> --output_dir <output-directory-path>
```

The following is an example command, how it should look. _IMPORTANT:_ The `neo4j-output` folder path must match that described in the docker compose file.

```bash
python snomed_g_graphdb_build_tools.py db_build --action create --rf2 C:\repos\concept-normalisation\data\SnomedCT_InternationalRF2_PRODUCTION_20250901T120000Z\Full\ --release_type full --neopw conceptnorm --output_dir C:\repos\concept-normalisation\data\neo4j-output\
```

### Prepare the loaded Neo4j graph for GraphRAG retrieval

After loading SNOMED CT into Neo4j, enrich the active `ObjectConcept` nodes with:

- `search_terms`: the FSN and active English synonyms used for full-text retrieval.
- `embedding_text`: the FSN, synonyms, and TextDefinition when available.
- `embedding`: normalized BioLORD embeddings used for vector retrieval.

The existing Neo4j nodes and relationships remain intact. The TextDefinition file, BioLORD model, Neo4j connection, index names, and batch sizes are configured in `concept_normalisation/config.py`.

Run the script from the repository root:

```bash
python scripts/prepare_neo4j_graphrag.py
```

The script automatically uses CUDA when available, followed by Apple MPS and then CPU.

The process is resumable because nodes with an existing `embedding` property are skipped. To rebuild the retrieval properties and embeddings for every active concept, use:

```bash
python scripts/prepare_neo4j_graphrag.py --force
```

The script creates the following Neo4j indexes:

- `snomed_concept_fulltext` for full-text retrieval using `search_terms`.
- `snomed_concept_embeddings` for cosine vector retrieval using `embedding`.

Useful Neo4j Browser inspection queries:

```cypher
MATCH (c:ObjectConcept)
WHERE c.active = true OR toString(c.active) = '1'
RETURN count(c) AS active,
       count(c.search_terms) AS fulltext_ready,
       count(c.embedding_text) AS vector_text_ready,
       count(c.embedding) AS vector_ready;
```

```cypher
SHOW INDEXES YIELD name, type, state, options
WHERE name IN [
    'snomed_concept_fulltext',
    'snomed_concept_embeddings'
]
RETURN name, type, state, options;
```

```cypher
CALL db.index.fulltext.queryNodes(
    'snomed_concept_fulltext',
    'heart attack'
)
YIELD node, score
RETURN node.sctid,
       node.FSN,
       node.search_terms,
       score
ORDER BY score DESC
LIMIT 10;
```

### Before running the pipeline

Finally, before running the pipeline you must install and run the corresponding ollama model described [here](src/concept_normalisation/config.py). Default would be:

```bash
ollama run llama3.1
```

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

| File                                | Contents                                             |
| ----------------------------------- | ---------------------------------------------------- |
| `semantic_data_mapped_full.parquet` | every row with all methods' matches                  |
| `method_comparison.parquet`         | one row per query, top match per method side by side |
| `final_candidates.parquet`          | hierarchy-aware consensus candidates                 |

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
