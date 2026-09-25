from pathlib import Path
import pandas as pd

def run_graphrag(
    data: pd.DataFrame,
    matcher,
    text_column: str = "diagnosis_text",
    output_column: str = "algorithm_graphrag_matches",
    top_k: int = 10,
    min_score: float = 0.94,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
):
    result = data.copy()

    # Create column if this is the first run.
    if output_column not in result.columns:
        result[output_column] = None

    # Reuse already-computed results from previous run
    cached = {}
    completed = result[result[output_column].notna()]

    for _, row in completed.iterrows():
            cached[row[text_column]] = row[output_column]

    # Only unique queries that do not already have a cached result.
    unique_queries = (
        result.loc[result[output_column].isna(), text_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    unique_queries = [
        query for query in unique_queries 
        if query not in cached
    ]

    print(
        f"GraphRAG: {len(unique_queries)} unqiue queries remaining "
        f"from {len(result)} total rows"
    )

    for count, query in enumerate(unique_queries, start=1):

        try:
            response = matcher.search(
                diagnosis=query,
                top_k=top_k,
                min_score=min_score,
            )

            value = response["matches"]
            cached[query] = value

            # Fill identical rows with the result
            mask = result[text_column].astype(str) == query
            result.loc[mask, output_column] = pd.Series(
                [value] * mask.sum(),
                index=result.index[mask],
                dtype="object",
            )

        except Exception as exc:
            print(f"\nGraphRAG failed for query: {query}")
            print(exc)

            if checkpoint_path is not None:
                result.to_parquet(checkpoint_path)

            raise

        if (checkpoint_path is not None and count % checkpoint_every == 0):
                result.to_parquet(checkpoint_path)

                print(f"GraphRAG: {count}/{len(unique_queries)} unqiue queries completed") 

    if checkpoint_path is not None:
            result.to_parquet(checkpoint_path)

    return result

    