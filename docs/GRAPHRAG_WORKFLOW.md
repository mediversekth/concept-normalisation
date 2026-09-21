# GraphRAG Workflow

> [!NOTE]
> Implemented in [`graphrag_matcher.py`](../src/concept_normalisation/graphrag/graphrag_matcher.py), orchestrated by [`pipeline/graphrag.py`](../src/concept_normalisation/pipeline/graphrag.py).

---

## 1. Pipeline Workflow

```mermaid
flowchart TD
    A([Diagnosis string]) --> B[Clean query]
    B --> C[Hybrid retrieval\nvector index + fulltext index\ntop_k candidates]
    C --> D[Expand each candidate\nvia Cypher into graph context\nparents, children, role groups…]
    D --> E[LLM reranks candidates\nusing diagnosis + graph context]
    E --> F{Valid JSON response?}
    F -->|Yes| G([Ranked SNOMED matches])
    F -->|No / count mismatch| H([Warning logged\nempty result])
```

---

## 2. SNOMED CT Schema in Neo4j

```mermaid
flowchart LR
    CONCEPT["ObjectConcept\n─────────────\nsctid\nFSN\nsearch_terms\nembedding_text\nembedding"]

    PARENT["ObjectConcept\n(parent)"]
    GRANDPARENT["ObjectConcept\n(grandparent)"]
    CHILD["ObjectConcept\n(child)"]
    GRANDCHILD["ObjectConcept\n(grandchild)"]

    RG["RoleGroup"]

    SITE["ObjectConcept\nFINDING_SITE"]
    MORPH["ObjectConcept\nASSOCIATED_MORPHOLOGY"]
    AGENT["ObjectConcept\nCAUSATIVE_AGENT"]
    DUETO["ObjectConcept\nDUE_TO"]
    COURSE["ObjectConcept\nCLINICAL_COURSE"]
    INTERP["ObjectConcept\nINTERPRETS"]

    CONCEPT -->|ISA| PARENT
    PARENT -->|ISA| GRANDPARENT
    CHILD -->|ISA| CONCEPT
    GRANDCHILD -->|ISA| CHILD
    CONCEPT -->|HAS_ROLE_GROUP| RG
    RG -->|FINDING_SITE| SITE
    RG -->|ASSOCIATED_MORPHOLOGY| MORPH
    RG -->|CAUSATIVE_AGENT| AGENT
    RG -->|DUE_TO| DUETO
    RG -->|CLINICAL_COURSE| COURSE
    RG -->|INTERPRETS| INTERP
```

| Relationship            | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `ISA`                   | Is-a hierarchy; traversed 2 levels up (parents) and 2 levels down (children) |
| `HAS_ROLE_GROUP`        | Groups one or more role-filler relationships; a concept can have many        |
| `FINDING_SITE`          | Body structure where the finding occurs                                      |
| `ASSOCIATED_MORPHOLOGY` | Structural/tissue change (e.g. infarct, inflammation)                        |
| `CAUSATIVE_AGENT`       | Organism or substance causing the disorder                                   |
| `DUE_TO`                | Causal condition or event                                                    |
| `CLINICAL_COURSE`       | Temporal pattern (e.g. acute, chronic)                                       |
| `INTERPRETS`            | Observable entity being interpreted                                          |

---

## 3. Neo4j Enrichment Script

Run once before GraphRAG can be used. Source: [`scripts/prepare_neo4j_graphrag.py`](../scripts/prepare_neo4j_graphrag.py).

```mermaid
flowchart TD
    A([prepare_neo4j_graphrag.py]) --> B[Load RF2 TextDefinitions]
    B --> C[Fetch ObjectConcept batch\nwith Description nodes]
    C --> D[Build search_terms\nFSN + synonyms\nfor fulltext index]
    C --> E[Build embedding_text\nFSN + synonyms + definition\nfor vector index]
    D & E --> F[Encode with BioLORD\n→ embedding vector]
    F --> G[Write back to Neo4j]
    G --> H{More concepts?}
    H -->|Yes| C
    H -->|No| I[Create FULLTEXT INDEX\nCreate VECTOR INDEX]
    I --> J([Done])
```

### Configuration knobs

| Config key                      | Default                     | Effect                                                                                      |
| ------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| `GRAPHRAG_EMBEDDING_MODEL_NAME` | `FremyCompany/BioLORD-2023` | Embedding model — must match between enrichment and query time                              |
| `GRAPHRAG_FETCH_BATCH_SIZE`     | `2000`                      | Concepts fetched per Neo4j round-trip                                                       |
| `GRAPHRAG_EMBEDDING_BATCH_SIZE` | `256`                       | Texts encoded per model call                                                                |
| `GRAPHRAG_WRITE_BATCH_SIZE`     | `500`                       | Concepts updated per Neo4j write                                                            |
| `GRAPHRAG_DEFAULT_TOP_K`        | `5`                         | Candidates retrieved at query time; higher values increase LLM context and can hurt ranking |

> [!IMPORTANT]
> The embedding model must be the same at enrichment time and at query time. Mismatching models will silently produce poor retrieval.

> [!TIP]
> Use `--force` to rebuild all embeddings, e.g. after switching the embedding model.
