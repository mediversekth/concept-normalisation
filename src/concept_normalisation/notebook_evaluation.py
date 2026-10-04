"""Helpers for the GraphRAG baseline evaluation notebook.

Data loading, scoring, trace analysis, plotting, and exports are separate steps.
Functions take their inputs explicitly; importing this module performs no I/O.
Display helpers produce notebook output; export helpers write evaluation artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import hashlib
import json
import os
import platform
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, display


METHODS = {
    "SapBERT": ("algorithm_1_matches", "concept_id"),
    "BioLORD + context": ("algorithm_2_matches", "concept_id"),
    "AI context + BioLORD": ("algorithm_ai_matches", "concept_id"),
    "BM25": ("multi_match_matches", "concept_id"),
    "BM25 + fuzzy": ("elastic_fuzzy_matches", "concept_id"),
    "Jaccard": ("jaccard_matches", "concept_id"),
    "GraphRAG": ("algorithm_graphrag_matches", "sctid"),
}


COLORS = dict(zip(METHODS, ["#4c78a8", "#72b7b2", "#b279a2", "#e0a133",
                            "#8c6d31", "#999999", "#b52242"]))


def find_repository_root(start: Path | None = None) -> Path:
    """Find the repository when running from its root or a nested notebook folder."""
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").exists():
            return directory
    raise FileNotFoundError("Open this notebook from the concept-normalisation repository.")


def resolve_data_directory(root: Path) -> Path:
    """Respect the same data-directory environment override as the pipeline."""
    return Path(os.environ.get("CONCEPT_NORM_DATA_DIR", str(root / "data"))).resolve()


def dataset_specs(data_dir: Path) -> dict:
    """
    Discover evaluation datasets from pipeline input CSVs.

    Known ICD datasets keep their ICD-code join semantics. Any additional CSV
    with a matching pipeline checkpoint is treated as a direct SNOMED-labelled
    dataset and evaluated from its diagnosis text + `snomed` labels.

    Expected artifact naming:
        data/<stem>.csv
        data/output/<stem>_pipeline_checkpoint.parquet
        data/logs/<stem>_graphrag_candidates.jsonl

    This means files such as `snomed_challenge_50.csv` are picked up
    automatically once their matching pipeline checkpoint exists.
    """
    output_dir = data_dir / "output"
    log_dir = data_dir / "logs"

    known = {
        "diagnosis_icd9_snomed": ("ICD9", "icd9"),
        "diagnosis_icd10_snomed": ("ICD10", "icd10"),
    }

    specs = {}

    for truth_path in sorted(data_dir.glob("*.csv")):
        stem = truth_path.stem
        checkpoint = output_dir / f"{stem}_pipeline_checkpoint.parquet"

        # Only include datasets that have actually been run through the pipeline.
        if not checkpoint.exists():
            continue

        if stem in known:
            label, code = known[stem]
            dataset_type = "icd"
        else:
            label = stem
            code = None
            dataset_type = "direct_snomed"

        specs[label] = {
            "type": dataset_type,
            "code": code,
            "truth": truth_path,
            "checkpoint": checkpoint,
            "log": log_dir / f"{stem}_graphrag_candidates.jsonl",
        }

    return specs


def configure_display() -> None:
    """Apply the shared table and figure style when requested by the notebook."""
    pd.set_option("display.max_colwidth", 110)
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, name: str, output_dir: Path) -> None:
    """Save PNG/SVG versions, display the figure, and release its memory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def normalise_id(value):
    """Preserve identifier strings and reject potentially rounded float IDs."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        raise ValueError("Float concept ID is unsafe; reload IDs as strings or integers.")
    value = str(value).strip()
    return None if value.lower() in {"", "nan", "none", "null", "<na>"} else value


def gold_set(value):
    """Parse pipe-separated source labels into a set of exact SNOMED IDs."""
    value = normalise_id(value)
    if not value:
        return frozenset()
    concepts = [normalise_concept(concept) for concept in value.split("|")]
    return frozenset(concept for concept in concepts if concept is not None)


def normalise_concept(value):
    """Repair safe legacy SNOMED strings without changing ICD code formatting."""
    value = normalise_id(value)
    # Older ICD9 checkpoints contain stringified integral floats (e.g. "66657009.0").
    # Only repair exactly representable integer IDs; never apply this to ICD codes.
    if value and re.fullmatch(r"[0-9]+\.0+", value):
        integer = value.split(".")[0]
        if int(integer) >= 2**53:
            raise ValueError("Possible rounded SNOMED ID; recover it from the original source.")
        value = integer
    return value


def match_list(value):
    """Read saved candidate containers; distinguish missing from empty results."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(not isinstance(m, dict) for m in value):
        raise ValueError("Expected a list of candidate dictionaries.")
    return list(value)


def candidate_ids(value, key):
    """Return distinct concept IDs in their original prediction order."""
    ids = []
    for match in match_list(value) or []:
        concept = normalise_concept(match.get(key))
        if concept is None:
            raise ValueError(f"Candidate lacks a valid {key}: {match}")
        if concept not in ids:
            ids.append(concept)
    return ids


def score_ids(
    ids: list[str], gold: set[str] | frozenset[str], ks: list[int], depth: int
) -> dict[str, float]:
    """Compute exact Top-1 Accuracy, truncated MRR, and set Recall@K."""
    if not gold:
        raise ValueError("Unlabelled queries must be excluded before evaluation.")
    if depth < 1 or any(k < 1 for k in ks):
        raise ValueError("K and MRR depth must be positive.")
    ids = list(dict.fromkeys(ids))
    reciprocal_rank = 0.0
    for rank, concept in enumerate(ids[:depth], start=1):
        if concept in gold:
            reciprocal_rank = 1.0 / rank
            break

    scores = {
        "Top-1 Accuracy": float(bool(ids) and ids[0] in gold),
        f"MRR@{depth}": reciprocal_rank,
    }
    for k in ks:
        recovered_targets = set(ids[:k]) & gold
        scores[f"Recall@{k}"] = len(recovered_targets) / len(gold)
    return scores


def attach_gold(data, truth, code_column):
    """Restore source ICD codes and attach validated labels without positional joins."""
    # Join on the ICD key, never by CSV/checkpoint row position.
    truth = truth.copy()
    data = data.copy().reset_index(drop=True)
    data["icd_key_restored"] = False
    identity = ["diagnosisid", "patientunitstayid", "diagnosisstring"]
    if all(c in data and c in truth for c in identity):
        # Legacy CSV inference removed ICD9 zeroes: 564.00 -> 564.0, 135 -> 135.0.
        # Recover the exact source code via stable record identity, not numeric rounding.
        for column in identity:
            data[column] = data[column].map(normalise_id)
            truth[column] = truth[column].map(normalise_id)
        lookup = truth[identity + [code_column]].drop_duplicates()
        if lookup.duplicated(identity).any():
            raise ValueError("Record identity maps to multiple ICD codes in source CSV.")
        lookup = lookup.rename(columns={code_column: "source_icd_code"})
        data = data.merge(lookup, on=identity, how="left", validate="many_to_one", indicator=True)
        if data["_merge"].ne("both").any():
            raise ValueError("Checkpoint records missing from source; check dataset alignment.")
        data["icd_key_restored"] = data[code_column].map(normalise_id).ne(data.source_icd_code.map(normalise_id))
        data[code_column] = data.pop("source_icd_code")
        data = data.drop(columns="_merge")
    truth[code_column] = truth[code_column].map(normalise_id)
    truth["gold"] = truth["snomed"].map(gold_set)
    labelled = truth[truth[code_column].notna()]
    ambiguous = labelled.groupby(code_column)["gold"].nunique()
    if (ambiguous > 1).any():
        raise ValueError("Conflicting target sets for an ICD code; define a finer join key before scoring.")
    mapping = labelled.drop_duplicates(code_column).set_index(code_column)["gold"]
    data[code_column] = data[code_column].map(normalise_id)
    data["gold"] = data[code_column].map(mapping).map(
        lambda x: x if isinstance(x, frozenset) else frozenset())
    if "snomed" in data:
        checkpoint_gold = data["snomed"].map(gold_set)
        if not checkpoint_gold.eq(data["gold"]).all():
            raise ValueError("Checkpoint labels disagree with source labels; check dataset/version alignment.")
    if "diagnosis_text" not in data:
        data["diagnosis_text"] = data["diagnosisstring"].str.replace("|", " - ", regex=False)
    if data["diagnosis_text"].isna().any():
        raise ValueError("Missing diagnosis text; cannot define the unique-query cohort.")
    data["query_key"] = list(zip(data["diagnosis_text"], data[code_column]))
    data["row_id"] = np.arange(len(data))
    return data



def attach_direct_snomed_gold(data, truth):
    """
    Attach gold labels for datasets whose source CSV already contains SNOMED IDs.

    The cohort is keyed by diagnosis text rather than an ICD code. Multiple
    source rows with the same diagnosis text are allowed and become a set of
    valid SNOMED targets.
    """
    truth = truth.copy()
    data = data.copy().reset_index(drop=True)

    if "snomed" not in truth:
        raise ValueError("Direct SNOMED dataset must contain a 'snomed' column.")

    if "diagnosis_text" not in truth:
        if "diagnosisstring" not in truth:
            raise ValueError(
                "Direct SNOMED dataset must contain 'diagnosisstring' or 'diagnosis_text'."
            )
        truth["diagnosis_text"] = truth["diagnosisstring"].str.replace(
            "|", " - ", regex=False
        )

    if "diagnosis_text" not in data:
        if "diagnosisstring" not in data:
            raise ValueError(
                "Checkpoint must contain 'diagnosisstring' or 'diagnosis_text'."
            )
        data["diagnosis_text"] = data["diagnosisstring"].str.replace(
            "|", " - ", regex=False
        )

    truth["diagnosis_text"] = truth["diagnosis_text"].map(normalise_id)
    data["diagnosis_text"] = data["diagnosis_text"].map(normalise_id)

    if truth["diagnosis_text"].isna().any() or data["diagnosis_text"].isna().any():
        raise ValueError("Missing diagnosis text; cannot define the evaluation cohort.")

    truth["gold_single"] = truth["snomed"].map(gold_set)

    mapping = (
        truth.groupby("diagnosis_text", sort=False)["gold_single"]
        .agg(lambda values: frozenset().union(*values))
    )

    data["gold"] = data["diagnosis_text"].map(mapping).map(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )

    if "snomed" in data:
        checkpoint_gold = data["snomed"].map(gold_set)
        for saved, gold in zip(checkpoint_gold, data["gold"]):
            if saved and not saved.issubset(gold):
                raise ValueError(
                    "Checkpoint labels disagree with source labels; "
                    "check dataset/version alignment."
                )

    data["query_key"] = data["diagnosis_text"]
    data["row_id"] = np.arange(len(data))
    data["icd_key_restored"] = False

    return data

def load_datasets(specs: dict, mrr_depth: int = 5):
    """Load labelled cohorts and return datasets, label audit, and prediction coverage."""
    datasets, audits, coverage_rows = {}, [], []
    for label, spec in specs.items():
        for required in ("truth", "checkpoint"):
            if not spec[required].exists():
                raise FileNotFoundError(f"Missing {label} {required}: {spec[required]}. Run the matching pipeline first.")
        truth = pd.read_csv(spec["truth"], dtype=str, keep_default_na=False)
        raw = pd.read_parquet(spec["checkpoint"])
        if spec.get("type") == "direct_snomed" or spec.get("code") is None:
            data = attach_direct_snomed_gold(raw, truth)
        else:
            data = attach_gold(raw, truth, spec["code"])
        labelled = data[data.gold.map(bool)].copy()
        if labelled.empty:
            raise ValueError(f"{label} has no labelled evaluation rows.")
        datasets[label] = labelled
        sizes = labelled.drop_duplicates("query_key").gold.map(len)
        audits.append({"dataset": label, "source rows": len(truth), "checkpoint rows": len(data),
            "restored ICD keys": int(data.icd_key_restored.sum()),
            "exact duplicate source rows": int(truth.duplicated().sum()),
            "excluded unlabelled rows": len(data) - len(labelled),
            "labelled rows": len(labelled), "unique queries": len(sizes),
            "singleton queries": int(sizes.eq(1).sum()), "multi-target queries": int(sizes.gt(1).sum()),
            "min targets": sizes.min(), "max targets": sizes.max()})
        for method, (column, key) in METHODS.items():
            if column not in labelled:
                coverage_rows.append({"dataset": label, "method": method, "status": "UNAVAILABLE"})
                continue
            values = labelled[column].map(match_list)
            counts = labelled[column].map(lambda v: len(candidate_ids(v, key)))
            coverage_rows.append({"dataset": label, "method": method, "status": "available",
                "labelled rows": len(labelled), "missing predictions": int(values.map(lambda v: v is None).sum()),
                "empty predictions": int(values.map(lambda v: v == []).sum()),
                "min distinct predictions": counts.min(), "max distinct predictions": counts.max(),
                "rows with fewer than 5": int(counts.lt(mrr_depth).sum())})
    audit = pd.DataFrame(audits)
    coverage = pd.DataFrame(coverage_rows)
    return datasets, audit, coverage


def plot_target_cardinality(datasets: dict, output_dir: Path):
    """Plot target-set sizes separately for each dataset."""
    nr_of_datasets = len(datasets)
    fig, axes = plt.subplots(1, nr_of_datasets, figsize=(12, 3.5), layout="constrained")
    for ax, (label, data) in zip(axes, datasets.items()):
        counts = data.drop_duplicates("query_key").gold.map(len).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
        ax.set(title=f"{label}: target-set cardinality", xlabel="Valid SNOMED targets per query",
               ylabel="Unique queries")
        ax.bar_label(ax.containers[0], padding=3)
        ax.margins(y=.2)
    save_figure(fig, "target_cardinality", output_dir)


def evaluate_methods(datasets: dict, ks: list[int], mrr_depth: int):
    """Return per-row scores, per-query scores, and both averaging summaries."""
    records = []
    for label, data in datasets.items():
        for method, (column, id_key) in METHODS.items():
            if column not in data:
                continue
            for row in data.to_dict("records"):
                ids = candidate_ids(row[column], id_key)
                records.append({"dataset": label, "method": method, "row_id": row["row_id"],
                    "query_key": row["query_key"], "diagnosis_text": row["diagnosis_text"],
                    "target_count": len(row["gold"]), "prediction_count": len(ids),
                    **score_ids(ids, row["gold"], ks, mrr_depth)})
    per_row = pd.DataFrame(records)
    metrics = ["Top-1 Accuracy", f"MRR@{mrr_depth}", *[f"Recall@{k}" for k in ks]]
    per_query = per_row.groupby(["dataset", "method", "query_key"], sort=False)[metrics].mean().reset_index()
    summaries = []
    for view, frame in [("unique query", per_query), ("row", per_row)]:
        table = frame.groupby(["dataset", "method"], sort=False)[metrics].mean().reset_index()
        table["view"] = view
        table["n"] = frame.groupby(["dataset", "method"], sort=False).size().values
        summaries.append(table)
    scores = pd.concat(summaries, ignore_index=True)
    assert scores[metrics].ge(0).all().all() and scores[metrics].le(1).all().all()
    assert (scores[[f"Recall@{k}" for k in ks]].diff(axis=1).iloc[:, 1:] >= -1e-12).all().all()
    return per_row, per_query, scores


def metric_names(ks: list[int], mrr_depth: int) -> list[str]:
    """List the requested metric columns in report order."""
    return ["Top-1 Accuracy", f"MRR@{mrr_depth}", *[f"Recall@{k}" for k in ks]]


def dataset_score_table(scores: pd.DataFrame, label: str, ks: list[int], mrr_depth: int):
    """Select both averaging views for one dataset without changing their values."""
    metrics = metric_names(ks, mrr_depth)
    return scores[scores.dataset.eq(label)].set_index(["view", "method"])[["n", *metrics]].round(4)


def plot_method_comparison(
    scores: pd.DataFrame,
    label: str,
    ks: list[int],
    mrr_depth: int,
    view: str,
    output_dir: Path,
):
    """Plot the three headline metrics with GraphRAG highlighted."""
    nr_of_datasets = len(scores.dataset.unique())

    plot = scores[scores.dataset.eq(label) & scores.view.eq(view)].set_index("method")
    order = [method for method in METHODS if method in plot.index]
    fig, axes = plt.subplots(1, nr_of_datasets, figsize=(15, 4.5), sharey=True, layout="constrained")
    for ax, metric in zip(axes, ["Top-1 Accuracy", f"MRR@{mrr_depth}", f"Recall@{max(ks)}"]):
        values = plot.loc[order, metric]
        bars = ax.barh(order, values, color=[COLORS[m] for m in order])
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set(xlim=(0, 1.15), xlabel=metric)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle(f"{label} · {view} average · n={int(plot.n.iloc[0])}")
    save_figure(fig, f"{label.lower()}_method_comparison", output_dir)


def plot_recall_curves(
    datasets: dict,
    scores: pd.DataFrame,
    ks: list[int],
    view: str,
    output_dir: Path,
):
    """Plot each method and the target-set recall ceiling."""
    nr_of_datasets = len(datasets)

    fig, axes = plt.subplots(1, nr_of_datasets, figsize=(13, 4.5), sharey=True, layout="constrained")
    for ax, (label, data) in zip(axes, datasets.items()):
        table = scores[scores.dataset.eq(label) & scores.view.eq(view)]
        for _, row in table.iterrows():
            method = row["method"]
            ax.plot(ks, [row[f"Recall@{k}"] for k in ks], marker="o", color=COLORS[method],
                    linewidth=3 if method == "GraphRAG" else 1.4, label=method)
        gold_sizes = (data.drop_duplicates("query_key") if view == "unique query" else data).gold.map(len)
        ceiling = [np.minimum(k, gold_sizes).div(gold_sizes).mean() for k in ks]
        ax.plot(ks, ceiling, "k--", alpha=.7, label="Label-set ceiling")
        ax.set(title=label, xlabel="K distinct predictions", ylabel="Mean Recall@K", xticks=ks, ylim=(0, 1.05))
        ax.grid(alpha=.15)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle(f"Recall curves · {view} average")
    save_figure(fig, "recall_curves", output_dir)


def baseline_differences(scores: pd.DataFrame, ks: list[int], mrr_depth: int, view: str):
    """Subtract each baseline from GraphRAG within the same dataset and averaging view."""
    delta_rows = []
    for label in scores.dataset.unique():
        table = scores[scores.dataset.eq(label) & scores.view.eq(view)].set_index("method")
        if "GraphRAG" not in table.index:
            continue
        for baseline in table.index.drop("GraphRAG"):
            delta_rows.append({"dataset": label, "baseline": baseline,
                **{metric: table.loc["GraphRAG", metric] - table.loc[baseline, metric]
                   for metric in ["Top-1 Accuracy", f"MRR@{mrr_depth}", f"Recall@{max(ks)}"]}})
    deltas = pd.DataFrame(delta_rows)
    return deltas


def clean_query(text):
    """Mirror GraphRAGMatcher._clean_query for matching saved logs."""
    text = str(text)

    text = text.replace("/", " or ")
    text = re.sub(r'[+\-!(){}\[\]^"~*?:\\]', " ", text)
    text = text.replace("&&", " ")
    text = text.replace("||", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_traces(datasets: dict, specs: dict):
    """Return consistent GraphRAG traces and coverage; missing logs are audited, not fatal."""
    trace_records, trace_audit = [], []

    for label, data in datasets.items():
        path = specs[label]["log"]
        logs_by_query = {}

        if path.exists():
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue

                entry = json.loads(line)
                entry["log_line"] = line_no

                logged_query = clean_query(entry["diagnosis"])
                logs_by_query.setdefault(logged_query, []).append(entry)

        cleaned = data.diagnosis_text.map(clean_query)

        collisions = (
            data.assign(cleaned=cleaned)
            .groupby("cleaned")
            .diagnosis_text
            .nunique()
        )

        column, key = METHODS["GraphRAG"]

        for query_key, group in data.groupby("query_key", sort=False):

            # ICD datasets use (diagnosis_text, code).
            # Direct-SNOMED datasets use diagnosis_text directly.
            diagnosis = (
                query_key[0]
                if isinstance(query_key, tuple)
                else query_key
            )

            cleaned_diagnosis = clean_query(diagnosis)

            status = "no matching log"
            entry = None

            if column not in group:
                status = "GraphRAG unavailable"

            elif collisions.get(cleaned_diagnosis, 0) > 1:
                status = "ambiguous cleaned query"

            else:
                predictions = {
                    tuple(candidate_ids(v, key))
                    for v in group[column]
                }

                if len(predictions) != 1:
                    status = "multiple checkpoint outputs for query"

                else:
                    matching = [
                        e
                        for e in logs_by_query.get(cleaned_diagnosis, [])
                        if tuple(candidate_ids(e.get("matches"), key))
                        in predictions
                        and "llm_candidates" in e
                        and "candidates" in e
                    ]

                    if matching:
                        entry = max(
                            matching,
                            key=lambda e: (
                                e.get("timestamp", ""),
                                e["log_line"],
                            ),
                        )
                        status = "matched final IDs"

            trace_audit.append({
                "dataset": label,
                "query_key": query_key,
                "diagnosis_text": diagnosis,
                "status": status,
                "row_count": len(group),
            })

            if entry is not None:
                trace_records.append({
                    "dataset": label,
                    "query_key": query_key,
                    "diagnosis_text": diagnosis,
                    "gold": group.gold.iloc[0],
                    "entry": entry,
                })

    trace_coverage = pd.DataFrame(trace_audit)

    return trace_records, trace_coverage


def analyse_traces(trace_records: list[dict]):
    """Return stage recall, query outcomes, and candidate-membership violations.

    When retrieval scores are not used (use_score=False), score-based filtering is
    bypassed and 100% of retrieved candidates pass into the LLM pool, so filtering
    loss is 0. When Reciprocal Rank Fusion is active, candidates are sorted by RRF
    score before filtering.
    """
    stage_rows, outcome_rows, membership_rows = [], [], []
    for trace in trace_records:
        entry, gold = trace["entry"], trace["gold"]
        retrieved = candidate_ids(entry["candidates"], "sctid")
        surviving = candidate_ids(entry["llm_candidates"], "sctid")
        final = candidate_ids(entry["matches"], "sctid")
        pools = {"Retrieved union": retrieved, "After filter": surviving, "Final list": final,
                 "Final top-1": final[:1]}
        for stage, ids in pools.items():
            stage_rows.append({"dataset": trace["dataset"], "query_key": trace["query_key"], "stage": stage,
                               "candidates": len(ids), "any target present": float(bool(set(ids) & gold)),
                               "pool recall": len(set(ids) & gold) / len(gold)})
        outside = set(final) - set(surviving)
        if outside:
            outcome = "Outside-pool output"
        elif final and final[0] in gold:
            outcome = "Correct top-1"
        elif set(final) & gold:
            outcome = "Correct at lower rank"
        elif set(surviving) & gold:
            outcome = "LLM selection loss"
        elif set(retrieved) & gold:
            outcome = "Filtering loss"
        else:
            outcome = "Retrieval miss"
        outcome_rows.append({"dataset": trace["dataset"], "query_key": trace["query_key"],
                             "diagnosis_text": trace["diagnosis_text"], "outcome": outcome})
        membership_rows.append({"dataset": trace["dataset"], "diagnosis_text": trace["diagnosis_text"],
            "outside_pool_ids": "|".join(sorted(outside)), "outside_pool_count": len(outside),
            "filter_ids_not_retrieved": len(set(surviving) - set(retrieved))})
    stages = pd.DataFrame(stage_rows)
    outcomes = pd.DataFrame(outcome_rows)
    membership = pd.DataFrame(membership_rows)
    return stages, outcomes, membership


def show_trace_diagnostics(
    stages: pd.DataFrame,
    outcomes: pd.DataFrame,
    membership: pd.DataFrame,
    output_dir: Path,
):
    """Display stage summaries, membership violations, and the diagnostic figure."""
    if not stages.empty:
        stage_summary = stages.groupby(["dataset", "stage"], sort=False).agg(
            queries=("query_key", "size"), mean_candidates=("candidates", "mean"),
            any_target_rate=("any target present", "mean"), mean_pool_recall=("pool recall", "mean"))
        display(stage_summary.round(4))
        display(membership.groupby("dataset")[["outside_pool_count", "filter_ids_not_retrieved"]].sum())
        violations = membership[membership.outside_pool_count.gt(0) | membership.filter_ids_not_retrieved.gt(0)]
        if not violations.empty:
            display(Markdown("**Candidate membership violations (main scores retain the saved outputs):**"))
            display(violations)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for label, frame in stages.groupby("dataset", sort=False):
            rates = frame.groupby("stage", sort=False)["any target present"].mean()
            axes[0].plot(rates.index, rates.values, marker="o", linewidth=2, label=label)
        axes[0].set(ylabel="Queries with any valid target", ylim=(0, 1.05), title="Target survival by stage")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend()
        counts = pd.crosstab(outcomes.dataset, outcomes.outcome)
        counts.plot.barh(stacked=True, ax=axes[1], colormap="tab20")
        axes[1].set(xlabel="Matched unique queries", ylabel="", title="Where GraphRAG succeeds or fails")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        save_figure(fig, "graphrag_stage_diagnostics", output_dir)
    else:
        print("No consistent logs available; main method scores remain valid, stage diagnostics unavailable.")


def _candidate_rows(entry: dict, gold: frozenset, survivors: set, final_ranks: dict):
    """Annotate retrieved candidates without changing their fused input order.

    Adapts to the pipeline configuration: includes score columns only when scores
    are present, and includes RRF/rank columns when available.
    """
    has_scores = any(
        c.get("score") is not None
        or c.get("base_score") is not None
        or c.get("enriched_score") is not None
        for c in entry.get("candidates", [])
    )
    has_rrf = any(c.get("rrf_score") is not None for c in entry.get("candidates", []))
    has_ranks = any(
        c.get("base_rank") is not None or c.get("enriched_rank") is not None
        for c in entry.get("candidates", [])
    )
    candidates = []
    for position, candidate in enumerate(entry.get("candidates", []), 1):
        concept = normalise_id(candidate.get("sctid"))
        row = {
            "input_position": position,
            "sctid": concept,
            "fsn": candidate.get("fsn"),
            "found_by": candidate.get("found_by", "not logged"),
        }
        if has_scores:
            row["base_score"] = candidate.get("base_score")
            row["enriched_score"] = candidate.get("enriched_score")
            row["fused_score"] = candidate.get("score")
        if has_rrf:
            row["rrf_score"] = candidate.get("rrf_score")
        if has_ranks:
            row["base_rank"] = candidate.get("base_rank")
            row["enriched_rank"] = candidate.get("enriched_rank")
        row["sent_to_LLM"] = concept in survivors
        row["final_rank"] = final_ranks.get(concept)
        row["is_target"] = concept in gold
        candidates.append(row)
    return candidates


def _show_graph_context(entry: dict, context_fields: list[str]) -> None:
    """Render expandable evidence panels for every candidate supplied to the LLM."""
    for candidate in entry["llm_candidates"]:
        detail = {key: candidate.get(key) for key in context_fields}
        display(HTML("<details><summary>" + escape(str(candidate.get("sctid")) + " — " + str(candidate.get("fsn")))
                     + "</summary><pre style='white-space:pre-wrap'>"
                     + escape(json.dumps(detail, ensure_ascii=False, indent=2)) + "</pre></details>"))


def _plot_candidate_movement(
    candidates: list[dict],
    final_ranks: dict,
    figure_name: str,
    output_dir: Path,
):
    """Connect fused input positions to final ranks, marking outside-pool predictions."""
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(candidates) * .28)), layout="constrained")
    for candidate in candidates:
        pos, rank = candidate["input_position"], candidate["final_rank"]
        color = "#b52242" if candidate["is_target"] else "#999999"
        ax.scatter(0, pos, color=color, s=30)
        ax.text(-.03, pos, candidate["sctid"], ha="right", va="center", fontsize=8)
        if rank is not None:
            ax.plot([0, 1], [pos, rank], color=color, alpha=.65)
            ax.scatter(1, rank, color=color, s=40)
            ax.text(1.03, rank, f"#{rank} {candidate['sctid']}", va="center", fontsize=8)
    retrieved_ids = {candidate["sctid"] for candidate in candidates}
    for concept, rank in final_ranks.items():
        if concept not in retrieved_ids:
            ax.scatter(1, rank, color="#e0a133", marker="x", s=60)
            ax.text(1.03, rank, f"#{rank} {concept} (not retrieved)", va="center", fontsize=8)
    ax.set(xticks=[0, 1], xticklabels=["Fused input position", "Final LLM rank"],
           xlim=(-.35, 1.4), title="Candidate movement (red = labelled target; unconnected = omitted)")
    ax.invert_yaxis()
    ax.set_yticks([])
    save_figure(fig, figure_name, output_dir)


def explain_trace(trace: dict, ks: list[int], mrr_depth: int, output_dir: Path):
    """Display and export a query walkthrough, adapting to the pipeline configuration.

    Handles both scored and unscored configurations, and shows RRF score columns
    when available.
    """
    entry, gold = trace["entry"], trace["gold"]
    final_ids = candidate_ids(entry["matches"], "sctid")
    survivors = set(candidate_ids(entry["llm_candidates"], "sctid"))
    final_ranks = {concept: i for i, concept in enumerate(final_ids, 1)}

    has_retrieval_scores = any(
        c.get("score") is not None
        or c.get("base_score") is not None
        or c.get("enriched_score") is not None
        for c in entry.get("candidates", [])
    )
    has_rrf = any(c.get("rrf_score") is not None for c in entry.get("candidates", []))
    has_llm_scores = any(
        m.get("score") is not None or m.get("rrf_score") is not None
        for m in entry.get("matches", [])
    )

    display(Markdown(f"### {trace['dataset']} — {trace['diagnosis_text']}"))
    print("1. Cleaned input:", clean_query(trace["diagnosis_text"]))
    print("Ground truth (evaluation only; not supplied as labels to GraphRAG):", ", ".join(sorted(gold)))
    print("Log timestamp:", entry.get("timestamp"), "| line:", entry["log_line"])

    config_notes = []
    if has_rrf:
        config_notes.append("Candidates fused with Reciprocal Rank Fusion (RRF).")
    if has_retrieval_scores and has_llm_scores:
        config_notes.append("Retrieval scores logged and copied by LLM.")
    elif has_retrieval_scores:
        config_notes.append("Retrieval scores logged; not returned by LLM.")
    else:
        config_notes.append("Retrieval scores not used.")
    print(" ".join(config_notes))

    candidates = _candidate_rows(entry, gold, survivors, final_ranks)
    table = pd.DataFrame(candidates)

    if has_rrf:
        display(Markdown("**2–5. Retrieved candidates, RRF fusion, and observed filtering**"))
    elif has_retrieval_scores:
        display(Markdown("**2–5. Retrieved candidates, fusion, and observed filtering**"))
    else:
        display(Markdown("**2–5. Retrieved candidates and fusion (scores not used; all forwarded to LLM)**"))
    display(table)

    display(Markdown("**3. Actual graph evidence supplied for every surviving candidate**"))
    context_fields = ["sctid", "fsn", "description", "parents", "grandparents", "children",
                      "finding_sites", "morphologies", "causative_agents", "due_to", "clinical_course", "interprets"]
    context = pd.DataFrame(entry["llm_candidates"]).reindex(columns=context_fields)
    _show_graph_context(entry, context_fields)

    if has_llm_scores:
        display(Markdown("**6. Final LLM order and its saved explanations (with retrieval scores)**"))
    else:
        display(Markdown("**6. Final LLM order and its saved explanations (pure reranking)**"))

    final_rows = []
    for rank, m in enumerate(entry["matches"], 1):
        concept = normalise_id(m.get("sctid"))
        row = {"rank": rank, "sctid": concept, "fsn": m.get("fsn")}
        if "score" in m and m["score"] is not None:
            row["score"] = m["score"]
        if "base_score" in m and m["base_score"] is not None:
            row["base_score"] = m["base_score"]
        if "enriched_score" in m and m["enriched_score"] is not None:
            row["enriched_score"] = m["enriched_score"]
        if "rrf_score" in m and m["rrf_score"] is not None:
            row["rrf_score"] = m["rrf_score"]
        row["reason"] = m.get("reason")
        row["is_target"] = concept in gold
        row["in_LLM_pool"] = concept in survivors
        final_rows.append(row)

    final_table = pd.DataFrame(final_rows)
    with pd.option_context("display.max_colwidth", None):
        display(final_table)
    display(pd.DataFrame([score_ids(final_ids, gold, ks, mrr_depth)]).round(4))

    slug = hashlib.sha256(trace["diagnosis_text"].encode()).hexdigest()[:10]
    if candidates and final_ids:
        figure_name = f"{trace['dataset'].lower()}_trace_{slug}"
        _plot_candidate_movement(candidates, final_ranks, figure_name, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{trace['dataset'].lower()}_{slug}_candidates.csv", index=False)
    context.to_json(output_dir / f"{trace['dataset'].lower()}_{slug}_context.json", orient="records", indent=2)
    final_table.to_csv(output_dir / f"{trace['dataset'].lower()}_{slug}_final.csv", index=False)


def select_examples(trace_records: list[dict], example_queries: dict[str, list[str]]):
    """Choose requested queries, or representative success, failure, and violation cases."""
    examples = []
    for label in dict.fromkeys(t["dataset"] for t in trace_records):
        available = [t for t in trace_records if t["dataset"] == label]
        requested = example_queries.get(label, [])
        if requested:
            selected = [t for t in available if t["diagnosis_text"] in requested]
            missing = set(requested) - {t["diagnosis_text"] for t in selected}
            if missing:
                print("No matched traces for:", sorted(missing))
        else:
            selected = []
            for want_success in [True, False]:
                for trace in available:
                    ids = candidate_ids(trace["entry"]["matches"], "sctid")
                    success = bool(ids) and ids[0] in trace["gold"]
                    if success == want_success:
                        selected.append(trace)
                        break
            for trace in available:
                ids = set(candidate_ids(trace["entry"]["matches"], "sctid"))
                pool = set(candidate_ids(trace["entry"]["llm_candidates"], "sctid"))
                if ids - pool:
                    if not any(t["query_key"] == trace["query_key"] for t in selected):
                        selected.append(trace)
                    break
        examples.extend(selected)
    return examples


def print_comparison_summary(scores: pd.DataFrame, ks: list[int], mrr_depth: int, view: str):
    """Print GraphRAG against the strongest available baseline for each headline metric."""
    for label in scores.dataset.unique():
        table = scores[scores.dataset.eq(label) & scores.view.eq(view)].set_index("method")
        if "GraphRAG" not in table.index:
            print(f"{label}: GraphRAG predictions unavailable.")
            continue
        print(f"{label} ({view}):")
        baselines = table.drop(index="GraphRAG")
        for metric in ["Top-1 Accuracy", f"MRR@{mrr_depth}", f"Recall@{max(ks)}"]:
            if baselines.empty:
                print(f"  {metric}: GraphRAG {table.loc['GraphRAG', metric]:.4f}; no baseline available")
            else:
                best = baselines[metric].idxmax()
                value = table.loc["GraphRAG", metric]
                print(f"  {metric}: GraphRAG {value:.4f}; best baseline {best} {baselines.loc[best, metric]:.4f}; "
                      f"difference {value - baselines.loc[best, metric]:+.4f}")


def file_info(path: Path):
    """Describe an input artifact, including its SHA-256 digest when present."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _graphrag_config_info() -> dict:
    """Capture current GraphRAG pipeline configuration for the evaluation manifest."""
    try:
        from concept_normalisation import config
        return {
            "retrieval_method": getattr(config, "GRAPHRAG_DEFAULT_RETRIEVEAL_METHOD", "unknown"),
            "use_score": getattr(config, "GRAPHRAG_USE_SCORE", "unknown"),
            "min_score": getattr(config, "GRAPHRAG_DEFAULT_MIN_SCORE", "unknown"),
            "top_k": getattr(config, "GRAPHRAG_DEFAULT_TOP_K", "unknown"),
            "max_llm_results": getattr(config, "GRAPHRAG_DEFAULT_MAX_LLM_RESULTS", "unknown"),
            "llm_model": getattr(config, "GRAPHRAG_LLM_MODEL_NAME", "unknown"),
        }
    except ImportError:
        return {"note": "config module unavailable at evaluation time"}


def export_evaluation(
    exports: dict[str, pd.DataFrame],
    specs: dict,
    root: Path,
    output_dir: Path,
    ks: list[int],
    mrr_depth: int,
    view: str,
):
    """Write evaluation tables and a provenance manifest without altering pipeline artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in exports.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "primary_view": view, "recall_k": ks, "mrr_depth": mrr_depth,
        "matching": "exact ID; stable deduplication; stored order", "methods": METHODS,
        "missing_predictions": "score zero; absent method columns unavailable",
        "graphrag_configuration": _graphrag_config_info(),
        "truth_join": "Restore source ICD via stable record identity, then ICD-key join with conflict/label validation",
        "inputs": {label: {kind: file_info(spec[kind]) for kind in ["truth", "checkpoint", "log"]}
                   for label, spec in specs.items()},
        "current_source_only": [file_info(root / "src" / "concept_normalisation" / "graphrag" / name)
                                for name in ["graphrag_matcher.py", "graphrag_queries.py"]],
        "historical_run_configuration": "Not recorded in supplied checkpoints/logs",
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(exports)} tables, PNG/SVG figures, worked-example evidence, and manifest to {output_dir}")
