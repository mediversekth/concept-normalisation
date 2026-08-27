import pandas as pd

from src.concept_normalisation.data_prep.context_builder import (
    build_semantic_inputs,
)


def prepare_data(
    data_path,
    table_name: str,
    query_column: str,
    context_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a table and build the inputs used by the semantic algorithms.

    Adds:
        algorithm_1_text
        algorithm_2_text
        algorithm_ai_metadata
    """

    data = pd.read_csv(data_path)

    semantic_data = build_semantic_inputs(
        data=data,
        table_name=table_name,
        query_column=query_column,
        context_columns=context_columns,
    )

    return semantic_data