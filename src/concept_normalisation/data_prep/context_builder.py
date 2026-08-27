"""
Generic context construction for terminology matching

Algorithm 1:
    Query-column value only.
Algorithm 2:
Query-column value + selected context-column values

AI metadata:
    Strfuctured table and column information that can be passed to an AI context generator
"""

import pandas as pd

from typing import Any



def validate_columns(
        data: pd.DataFrame, 
        query_column: str,
        context_columns: list[str] | None = None,
    ) -> None:

    # Check that all requested columns exist in the dataframe

    context_columns = context_columns or []

    required_columns = {
        query_column,
        *context_columns,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

def clean_value(value) -> str:
    # Convert one df calue to clean text
    # Missing values will become an empty string

    if pd.isna(value):
        return ""

    return str(value).strip()

def build_algorithm_1_text(
        row: pd.Series,
        query_column: str,
    ) -> str:
    return clean_value(row[query_column])

def build_algorithm_2_text(
        row: pd.Series,
        query_column: str,
        context_columns: list[str],
    ) -> str:

    lines = []
    query_value = clean_value(row[query_column])

    if query_value:
        lines.append(
            f"{query_column}: {query_value}"
        )
    for column in context_columns:
        value = clean_value(row[column])

        if value:
            lines.append(
                f"{column}: {value}"
            )
    return "\n".join(lines)


def build_algorithm_ai_metadata(
        row: pd.Series,
        table_name: str,
        query_column: str,
        context_columns: list[str],
    ) -> dict[str, Any]:

    return {
        # str() defensively: this ends up inside algorithm_ai_metadata,
        # which gets saved to parquet -- pyarrow raises ArrowInvalid if a
        # non-string (e.g. a pathlib.Path, if one ever gets passed here
        # by mistake) ends up in this dict.
        "table_name": str(table_name),
        "query_column": str(query_column),
        "query_value": clean_value(row[query_column]),
        "context_columns": list(context_columns),
        "context_values": (
            {
                column: clean_value(row[column])
                for column in context_columns
            }
            if context_columns
            else None
        ),

    }

def build_semantic_inputs(
        data: pd.DataFrame,
        table_name: str, 
        query_column: str,
        context_columns: list[str] | None = None,
    ) -> pd.DataFrame:
    # Adding all semantic input representations to the df

    context_columns = context_columns or []

    validate_columns(
        data=data,
        query_column=query_column,
        context_columns=context_columns
    )

    result = data.copy()

    result["algorithm_1_text"] = result.apply(
        lambda row: build_algorithm_1_text(
            row=row,
            query_column=query_column,
        ),
        axis=1,
    )

    result["algorithm_2_text"] = result.apply(
        lambda row: build_algorithm_2_text(
            row=row,
            query_column=query_column,
            context_columns=context_columns
        ),
        axis=1,
    )

    result["algorithm_ai_metadata"] = result.apply(
        lambda row: build_algorithm_ai_metadata(
            row=row,
            table_name=table_name,
            query_column=query_column,
            context_columns=context_columns,
        ),
        axis=1,
    )

    return result
