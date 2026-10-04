from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_graphrag(
    data: pd.DataFrame,
    matcher,
    text_column: str = "diagnosis_text",
    output_column: str = "algorithm_graphrag_matches",
    top_k: int = 10,
    min_score: float = 0.90,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 5,
    log_candidates: bool = True,
    log_errors: bool = True,
    max_workers: int = 2,
):
    result = data.copy()

    if output_column not in result.columns:
        result[output_column] = None

    cached = {}
    completed = result[result[output_column].notna()]

    for _, row in completed.iterrows():
        cached[row[text_column]] = row[output_column]

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
        f"GraphRAG: {len(unique_queries)} unique queries remaining "
        f"from {len(result)} total rows"
    )

    def process_query(query):
        response = matcher.search(
            diagnosis=query,
            top_k=top_k,
            min_score=min_score,
            log_candidates=log_candidates,
            log_errors=log_errors,
        )

        return query, response["matches"]

    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_query, query): query
            for query in unique_queries
        }

        for future in as_completed(futures):
            query = futures[future]

            try:
                query, value = future.result()

                cached[query] = value

                mask = result[text_column].astype(str) == query
                result.loc[mask, output_column] = pd.Series(
                    [value] * mask.sum(),
                    index=result.index[mask],
                    dtype="object",
                )

                completed_count += 1

                if (
                    checkpoint_path is not None
                    and completed_count % checkpoint_every == 0
                ):
                    result.to_parquet(checkpoint_path)

                    print(
                        f"GraphRAG: {completed_count}/"
                        f"{len(unique_queries)} unique queries completed"
                    )

            except Exception as exc:
                print(f"\nGraphRAG failed for query: {query}")
                print(exc)

                if checkpoint_path is not None:
                    result.to_parquet(checkpoint_path)

                raise

    if checkpoint_path is not None:
        result.to_parquet(checkpoint_path)

    return result