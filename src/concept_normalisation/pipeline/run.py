"""
The end-to-end concept-normalisation experiment.

It's broken up here into:

  - ExperimentConfig: every setting for one run, in one place, with
    sensible defaults -- this is what used to be hardcoded local
    variables at the top of main().

  - run_pipeline(config): orchestrates the stages below in order.

  - one private `_stage_...` function per pipeline stage, each doing
    exactly what its equivalent block in the old main() did.

Each semantic/syntactic stage is skipped automatically if its output
column is already present in the checkpoint (see `_load_or_build_data`),
so re-running after a crash resumes instead of
redoing finished work. Delete the checkpoint file, or set
`fresh_start=True`, to force a full rerun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from concept_normalisation import config
from concept_normalisation.data_prep.context_builder import build_semantic_inputs
from concept_normalisation.pipeline.evaluation import run_comparison, run_consensus
from concept_normalisation.pipeline.semantic import (
    run_algorithm_1,
    run_algorithm_2,
    run_algorithm_ai,
)
from concept_normalisation.pipeline.syntactic import (
    run_elastic_fuzzy,
    run_jaccard,
    run_multi_match,
)
from concept_normalisation.ranking.hierarchy import SnomedHierarchy
from concept_normalisation.semantic_matching.dense_index import (
    build_combined_pool,
    load_dense_index_from_parquet,
)
from concept_normalisation.semantic_matching.embedding.biolord_embedder import (
    BioLordEmbedder,
)
from concept_normalisation.semantic_matching.embedding.sapbert_embedder import (
    SapBertEmbedder,
)
from concept_normalisation.syntactic_matching.elastic_bm25 import ElasticBM25Index
from concept_normalisation.syntactic_matching.fuzzy_matcher import JaccardMatcher
from concept_normalisation.syntactic_matching.text_preprocessing import (
    TextPreprocessor,
)


@dataclass
class ExperimentConfig:
    """Every setting for one pipeline run."""

    # --- Input table ---
    table_path: Path = field(default_factory=lambda: config.ACTIVE_TABLE)
    query_column: str = "diagnosis_text"
    context_columns: list[str] = field(default_factory=list)

    # --- Matching parameters ---
    top_k: int = config.DEFAULT_TOP_K
    semantic_search_batch_size: int = config.DEFAULT_QUERY_BATCH_SIZE
    sapbert_batch_size: int = config.DEFAULT_SAPBERT_BATCH_SIZE
    biolord_batch_size: int = config.DEFAULT_BIOLORD_BATCH_SIZE
    jaccard_ngram_size: int = 3

    # --- Models ---
    sapbert_model: str = config.SAPBERT_MODEL_NAME
    biolord_model: str = config.BIOLORD_MODEL_NAME
    ollama_model: str = config.OLLAMA_MODEL_NAME

    # --- Which methods to run ---
    run_semantic_algorithm_1: bool = True
    run_semantic_algorithm_2: bool = True
    # Uses few-shot + the "named" prompt style -- see data_prep/ai_context.py.
    run_semantic_algorithm_ai: bool = True

    # Elasticsearch must be running at config.ELASTIC_URL for these two.
    # jaccard doesn't need it.
    run_syntactic_multi_match: bool = True
    run_syntactic_fuzzy: bool = True
    run_syntactic_jaccard: bool = True

    run_evaluation: bool = True
    run_final_consensus: bool = True

    # --- Consensus step ---
    consensus_top_k: int = 20       # candidates contributed by each method
    consensus_final_top_k: int = 10  # candidates retained after consensus
    consensus_max_depth: int = 3

    # --- Checkpointing ---
    # Delete the checkpoint file yourself, or set this to True, to force a
    # full rerun instead of resuming.
    fresh_start: bool = False

    @property
    def table_name(self) -> str:
        # Kept as a plain string, not a Path -- it ends up saved inside
        # algorithm_ai_metadata, and a Path there breaks to_parquet()
        # with an ArrowInvalid error.
        return Path(self.table_path).stem

    @property
    def checkpoint_path(self) -> Path:
        return config.OUTPUT_DIR / f"{self.table_name}_pipeline_checkpoint.parquet"


def run_pipeline(cfg: ExperimentConfig | None = None) -> pd.DataFrame:
    """
    Run the full concept-normalisation experiment and return the final
    per-query data (matches from every method plus the AI text, if
    generated). Also writes every intermediate/final result to
    config.OUTPUT_DIR, same as the individual stages always did.
    """

    cfg = cfg or ExperimentConfig()
    print(f"Active table: {cfg.table_name}")

    data = _stage_prepare_data(cfg)

    data = _stage_algorithm_1(cfg, data)

    biolord_embedder, biolord_index = _stage_load_biolord_resources(cfg, data)

    data = _stage_algorithm_2(cfg, data, biolord_embedder, biolord_index)
    data = _stage_algorithm_ai(cfg, data, biolord_embedder, biolord_index)

    data = _stage_syntactic_matching(cfg, data)

    data = _stage_save_results(cfg, data)

    if cfg.run_evaluation:
        _stage_comparison(cfg, data)

    if cfg.run_final_consensus:
        _stage_consensus(cfg, data)

    print("\n========================================")
    print("FINISHED")
    print("========================================")

    return data


# ============================================================
# Stage 1: prepare data (with checkpoint resume)
# ============================================================

def _load_or_build_data(cfg: ExperimentConfig) -> pd.DataFrame:
    if cfg.checkpoint_path.exists() and not cfg.fresh_start:
        print(f"Resuming from checkpoint: {cfg.checkpoint_path}")
        data = pd.read_parquet(cfg.checkpoint_path)
        print(f"Loaded {len(data):,} rows, columns already present: {list(data.columns)}")
        return data

    raw_data = pd.read_csv(cfg.table_path)

    # diagnosisstring's "|" hierarchy separators -> readable text.
    if "diagnosisstring" in raw_data.columns:
        raw_data[cfg.query_column] = raw_data["diagnosisstring"].str.replace(
            "|", " - ", regex=False
        )

    # Ground-truth/ID columns that are large integers (e.g. SNOMED codes
    # like 129032061000119103) -- keep them as strings so nothing gets
    # silently rounded/converted to float.
    for id_like_column in ("snomed", "icd9", "icd10"):
        if id_like_column in raw_data.columns:
            raw_data[id_like_column] = raw_data[id_like_column].astype(str)

    return build_semantic_inputs(
        data=raw_data,
        table_name=cfg.table_name,
        query_column=cfg.query_column,
        context_columns=cfg.context_columns,
    )


def _checkpoint(cfg: ExperimentConfig, data: pd.DataFrame, stage_name: str) -> None:
    data.to_parquet(cfg.checkpoint_path)
    print(f"Checkpoint saved after {stage_name}: {cfg.checkpoint_path}")


def _stage_prepare_data(cfg: ExperimentConfig) -> pd.DataFrame:
    print("\n========================================")
    print("PREPARING DATA")
    print("========================================")

    data = _load_or_build_data(cfg)

    print("\nPrepared dataframe:")
    print(data.shape)
    print(data[[cfg.query_column, "algorithm_1_text", "algorithm_2_text"]].head())

    return data


# ============================================================
# Stage 2: semantic algorithm 1 (query only -> SapBERT)
# ============================================================

def _stage_algorithm_1(cfg: ExperimentConfig, data: pd.DataFrame) -> pd.DataFrame:
    needed = cfg.run_semantic_algorithm_1 and "algorithm_1_matches" not in data.columns

    if cfg.run_semantic_algorithm_1 and not needed:
        print("\nalgorithm_1_matches already in checkpoint, skipping SEMANTIC ALGORITHM 1")
        return data

    if not needed:
        return data

    print("\n========================================")
    print("SEMANTIC ALGORITHM 1")
    print("Query only + SapBERT")
    print("========================================")

    sapbert_embedder = SapBertEmbedder(
        model_name=cfg.sapbert_model,
        batch_size=cfg.sapbert_batch_size,
    )

    index = load_dense_index_from_parquet(
        parquet_path=config.SHORT_TERMS_PARQUET,
        embeddings_path=config.SHORT_TERMS_SAPBERT_EMB,
        text_column="description",
    )

    data = run_algorithm_1(
        data=data,
        embedder=sapbert_embedder,
        candidate_index=index,
        text_column="algorithm_1_text",
        output_column="algorithm_1_matches",
        top_k=cfg.top_k,
        search_batch_size=cfg.semantic_search_batch_size,
    )

    _checkpoint(cfg, data, "SEMANTIC ALGORITHM 1")
    return data


# ============================================================
# Stage 3: shared BioLORD candidate index (algorithm 2 and AI both use it)
# ============================================================

def _stage_load_biolord_resources(cfg: ExperimentConfig, data: pd.DataFrame):
    algorithm_2_needed = (
        cfg.run_semantic_algorithm_2 and "algorithm_2_matches" not in data.columns
    )
    algorithm_ai_needed = (
        cfg.run_semantic_algorithm_ai and "algorithm_ai_matches" not in data.columns
    )

    if not (algorithm_2_needed or algorithm_ai_needed):
        return None, None

    print("\nLoading BioLORD resources...")

    biolord_embedder = BioLordEmbedder(
        model_name=cfg.biolord_model,
        batch_size=cfg.biolord_batch_size,
    )

    short_terms = pd.read_parquet(config.SHORT_TERMS_PARQUET).reset_index(drop=True)
    short_term_embeddings = np.load(config.SHORT_TERMS_BIOLORD_EMB)

    definitions = pd.read_parquet(config.DEFINITIONS_PARQUET).reset_index(drop=True)
    definition_embeddings = np.load(config.DEFINITIONS_BIOLORD_EMB)

    biolord_index = build_combined_pool(
        short_terms=short_terms,
        short_term_embeddings=short_term_embeddings,
        definitions=definitions,
        definition_embeddings=definition_embeddings,
    )

    return biolord_embedder, biolord_index


# ============================================================
# Stage 4: semantic algorithm 2 (query + context -> BioLORD)
# ============================================================

def _stage_algorithm_2(
    cfg: ExperimentConfig,
    data: pd.DataFrame,
    biolord_embedder: BioLordEmbedder | None,
    biolord_index,
) -> pd.DataFrame:
    needed = cfg.run_semantic_algorithm_2 and "algorithm_2_matches" not in data.columns

    if cfg.run_semantic_algorithm_2 and not needed:
        print("\nalgorithm_2_matches already in checkpoint, skipping SEMANTIC ALGORITHM 2")
        return data

    if not needed:
        return data

    print("\n========================================")
    print("SEMANTIC ALGORITHM 2")
    print("Query + context + BioLORD")
    print("========================================")

    data = run_algorithm_2(
        data=data,
        embedder=biolord_embedder,
        candidate_index=biolord_index,
        text_column="algorithm_2_text",
        output_column="algorithm_2_matches",
        top_k=cfg.top_k,
        embedding_batch_size=cfg.biolord_batch_size,
        search_batch_size=cfg.semantic_search_batch_size,
    )

    _checkpoint(cfg, data, "SEMANTIC ALGORITHM 2")
    return data


# ============================================================
# Stage 5: semantic algorithm AI (metadata -> Ollama -> BioLORD -> SNOMED)
# ============================================================

def _stage_algorithm_ai(
    cfg: ExperimentConfig,
    data: pd.DataFrame,
    biolord_embedder: BioLordEmbedder | None,
    biolord_index,
) -> pd.DataFrame:
    needed = cfg.run_semantic_algorithm_ai and "algorithm_ai_matches" not in data.columns

    if cfg.run_semantic_algorithm_ai and not needed:
        print("\nalgorithm_ai_matches already in checkpoint, skipping SEMANTIC ALGORITHM AI")
        return data

    if not needed:
        return data

    print("\n========================================")
    print("SEMANTIC ALGORITHM AI")
    print("AI context + BioLORD")
    print("========================================")

    data = run_algorithm_ai(
        data=data,
        embedder=biolord_embedder,
        candidate_index=biolord_index,
        model=cfg.ollama_model,
        top_k=cfg.top_k,
        embedding_batch_size=cfg.biolord_batch_size,
        search_batch_size=cfg.semantic_search_batch_size,
        checkpoint_path=config.OUTPUT_DIR / f"{cfg.table_name}_ai_context_checkpoint.csv",
    )

    _checkpoint(cfg, data, "SEMANTIC ALGORITHM AI")
    return data


# ============================================================
# Stage 6: syntactic matching (multi-match, fuzzy, jaccard)
# ============================================================

def _stage_syntactic_matching(cfg: ExperimentConfig, data: pd.DataFrame) -> pd.DataFrame:
    multi_match_needed = (
        cfg.run_syntactic_multi_match and "multi_match_matches" not in data.columns
    )
    elastic_fuzzy_needed = (
        cfg.run_syntactic_fuzzy and "elastic_fuzzy_matches" not in data.columns
    )
    jaccard_needed = (
        cfg.run_syntactic_jaccard and "jaccard_matches" not in data.columns
    )

    if not (multi_match_needed or elastic_fuzzy_needed or jaccard_needed):
        return data

    print("\n========================================")
    print("SYNTACTIC MATCHING")
    print("========================================")

    preprocessor = TextPreprocessor()

    elastic_index = None
    if multi_match_needed or elastic_fuzzy_needed:
        elastic_index = ElasticBM25Index()

    if cfg.run_syntactic_multi_match and not multi_match_needed:
        print("\nmulti_match_matches already in checkpoint, skipping Multi-match")

    if multi_match_needed:
        print("\n--- Multi-match ---")
        data = run_multi_match(
            data=data,
            query_column=cfg.query_column,
            elastic_index=elastic_index,
            top_k=cfg.top_k,
            preprocessor=preprocessor,
            output_column="multi_match_matches",
        )
        _checkpoint(cfg, data, "Multi-match")

    if cfg.run_syntactic_fuzzy and not elastic_fuzzy_needed:
        print("\nelastic_fuzzy_matches already in checkpoint, skipping Elasticsearch fuzzy")

    if elastic_fuzzy_needed:
        print("\n--- Elasticsearch fuzzy ---")
        data = run_elastic_fuzzy(
            data=data,
            query_column=cfg.query_column,
            elastic_index=elastic_index,
            top_k=cfg.top_k,
            preprocessor=preprocessor,
            output_column="elastic_fuzzy_matches",
        )
        _checkpoint(cfg, data, "Elasticsearch fuzzy")

    if cfg.run_syntactic_jaccard and not jaccard_needed:
        print("\njaccard_matches already in checkpoint, skipping Jaccard")

    if jaccard_needed:
        print("\n--- Jaccard ---")
        jaccard_candidates = pd.read_parquet(config.SHORT_TERMS_PARQUET).reset_index(drop=True)

        jaccard_matcher = JaccardMatcher(
            candidates=jaccard_candidates,
            term_column="description",
            id_column="concept_id",
            preprocessor=preprocessor,
            ngram_size=cfg.jaccard_ngram_size,
            index_path=config.JACCARD_NGRAM_INDEX_PARQUET,
        )

        data = run_jaccard(
            data=data,
            query_column=cfg.query_column,
            matcher=jaccard_matcher,
            top_k=cfg.top_k,
            output_column="jaccard_matches",
        )
        _checkpoint(cfg, data, "Jaccard")

    return data


# ============================================================
# Stage 7: save combined results
# ============================================================

def _stage_save_results(cfg: ExperimentConfig, data: pd.DataFrame) -> pd.DataFrame:
    print("\nSaving matching results...")

    # algorithm_ai_metadata was only needed to drive the AI generation step
    # (already consumed -> algorithm_ai_text) and isn't used downstream.
    # Dropped here rather than saved: when context_columns is empty, its
    # context_values field is an empty dict on every row, which pyarrow
    # can't write to parquet (a zero-field struct type isn't representable).
    data = data.drop(columns=["algorithm_ai_metadata"], errors="ignore")

    data.to_parquet(config.SEMANTIC_RESULTS_PARQUET)
    print(f"Saved: {config.SEMANTIC_RESULTS_PARQUET}")

    return data


# ============================================================
# Stage 8: method comparison
# ============================================================

def _stage_comparison(cfg: ExperimentConfig, data: pd.DataFrame) -> pd.DataFrame:
    print("\n========================================")
    print("METHOD COMPARISON")
    print("========================================")

    comparison = run_comparison(data=data, query_column=cfg.query_column)
    comparison.to_parquet(config.COMPARISON_TABLE_PARQUET)

    print(f"Saved comparison: {config.COMPARISON_TABLE_PARQUET}")
    print(comparison.head())

    return comparison


# ============================================================
# Stage 9: final hierarchy-aware consensus
# ============================================================

def _stage_consensus(cfg: ExperimentConfig, data: pd.DataFrame) -> pd.DataFrame:
    print("\n========================================")
    print("FINAL CONSENSUS")
    print("========================================")

    hierarchy = SnomedHierarchy(relationship_file=config.RELATIONSHIP_FILE)
    hierarchy.load()

    final_candidates = run_consensus(
        data=data,
        hierarchy=hierarchy,
        top_k=cfg.consensus_top_k,
        final_top_k=cfg.consensus_final_top_k,
        max_depth=cfg.consensus_max_depth,
        query_column=cfg.query_column,
    )

    final_candidates.to_parquet(config.FINAL_CANDIDATES_PARQUET)
    print(f"Saved final candidates: {config.FINAL_CANDIDATES_PARQUET}")
    print(final_candidates.head())

    return final_candidates


# ============================================================
# Optional: AI prompt-experiment mapping
#
# Not part of run_pipeline() -- this maps previously generated AI-context
# sentences (from a separate prompt/model sweep, see
# data_prep/ai_experiments.py) onto SNOMED via BioLORD. Call it directly
# when you're running a new round of prompt/model exploration; the main
# pipeline's algorithm_ai stage generates and maps a single condition on
# its own.
# ============================================================

def run_ai_experiment_mapping(
    biolord_embedder: BioLordEmbedder,
    biolord_index,
    top_k: int = config.DEFAULT_TOP_K,
    biolord_batch_size: int = config.DEFAULT_BIOLORD_BATCH_SIZE,
    semantic_search_batch_size: int = config.DEFAULT_QUERY_BATCH_SIZE,
) -> pd.DataFrame:
    from concept_normalisation.pipeline.semantic import run_ai_experiment_embeddings

    print("\n========================================")
    print("AI PROMPT EXPERIMENT MAPPING")
    print("Saved AI conditions + BioLORD")
    print("========================================")

    ai_results = pd.read_parquet(config.OUTPUT_DIR / "ai_experiment_all_outputs.parquet")
    print(f"Loaded {len(ai_results):,} AI-generated sentences")

    ai_results, ai_embeddings = run_ai_experiment_embeddings(
        ai_results=ai_results,
        embedder=biolord_embedder,
        candidate_index=biolord_index,
        text_column="ai_output",
        output_column="snomed_matches",
        top_k=top_k,
        embedding_batch_size=biolord_batch_size,
        search_batch_size=semantic_search_batch_size,
    )

    np.save(config.OUTPUT_DIR / "ai_experiment_biolord_embeddings.npy", ai_embeddings)
    ai_results.to_parquet(config.OUTPUT_DIR / "ai_experiment_snomed_results.parquet", index=False)

    print("Saved AI experiment embeddings and SNOMED matches.")
    return ai_results
