"""Prepare SNOMED CT Entity Linking Challenge annotations for concept normalisation.

Output columns: diagnosisstring (annotation span), snomed (concept ID).
Optionally sample a limited number of unique diagnosis strings for GraphRAG tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def limit_unique_queries(
    data: pd.DataFrame,
    max_unique_queries: int | None,
    *,
    random_state: int = 0,
) -> pd.DataFrame:
    """Keep every row belonging to a random subset of unique diagnosis strings."""
    if max_unique_queries is None:
        return data

    if max_unique_queries <= 0:
        raise ValueError("max_unique_queries must be greater than 0")

    unique_queries = data["diagnosisstring"].drop_duplicates()
    if len(unique_queries) <= max_unique_queries:
        return data

    rng = np.random.default_rng(random_state)
    selected = rng.choice(
        unique_queries.to_numpy(), size=max_unique_queries, replace=False
    )
    return data[data["diagnosisstring"].isin(selected)].copy()


def prepare_snomed_challenge(
    annotations_path: str | Path,
    *,
    max_unique_queries: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Convert annotation spans and SNOMED IDs into the pipeline's two columns."""
    annotations = pd.read_csv(
        annotations_path,
        dtype={"span": "string", "concept_id": "string"},
    )

    required_columns = {"span", "concept_id"}
    missing = required_columns - set(annotations.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    data = annotations[["span", "concept_id"]].rename(
        columns={"span": "diagnosisstring", "concept_id": "snomed"}
    ).copy()

    # Normalize surrounding/internal whitespace without adding note context.
    data["diagnosisstring"] = data["diagnosisstring"].str.split().str.join(" ")
    data["snomed"] = data["snomed"].str.strip()

    # Empty annotations and IDs cannot be linked to a concept.
    invalid = (
        data["diagnosisstring"].isna()
        | data["diagnosisstring"].eq("")
        | data["snomed"].isna()
        | data["snomed"].eq("")
    )
    if invalid.any():
        print(f"Skipping {int(invalid.sum()):,} rows with missing/empty spans or IDs")
        data = data.loc[~invalid].copy()

    total_unique = data["diagnosisstring"].nunique()
    data = limit_unique_queries(
        data, max_unique_queries=max_unique_queries, random_state=random_state
    )

    print(f"Rows: {len(data):,}")
    print(f"Unique diagnosis strings: {data['diagnosisstring'].nunique():,} / {total_unique:,}")
    if max_unique_queries is not None:
        print(f"Unique-query limit: {max_unique_queries:,} (random_state={random_state})")

    return data.reset_index(drop=True)


def prepare_snomed_challenge_csv(
    annotations_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """Prepare and save the two-column CSV."""
    data = prepare_snomed_challenge(annotations_path, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SNOMED challenge annotation spans for concept normalisation."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-unique-queries",
        type=int,
        default=None,
        help="Maximum number of unique diagnosis strings (keep all matching rows).",
    )
    parser.add_argument(
        "--random-state", type=int, default=0,
        help="Random seed for unique-query sampling.",
    )
    args = parser.parse_args()

    prepare_snomed_challenge_csv(
        annotations_path=args.annotations,
        output_path=args.output,
        max_unique_queries=args.max_unique_queries,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

