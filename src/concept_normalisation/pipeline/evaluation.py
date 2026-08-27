import pandas as pd

from concept_normalisation.ranking.comparison import (
    build_comparison_table,
)
from concept_normalisation.ranking.consensus import (
    build_final_candidates,
)


def run_comparison(
    data: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Build the side-by-side comparison table.
    """

    return build_comparison_table(
        data,
        **kwargs,
    )


def run_consensus(
    data: pd.DataFrame,
    hierarchy=None,
    top_k: int = 3,
    final_top_k: int = 5,
    max_depth: int = 3,
    query_column: str = "test_name",
) -> pd.DataFrame:
    """
    Run the final consensus step.

    top_k:
        Number of candidates contributed by each matching method.

    final_top_k:
        Number of candidates retained after consensus ranking.
    """

    return build_final_candidates(
        semantic_data=data,
        hierarchy=hierarchy,
        top_k=top_k,
        final_top_k=final_top_k,
        max_depth=max_depth,
        query_column=query_column,
    )
