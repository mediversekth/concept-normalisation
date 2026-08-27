import pandas as pd

from concept_normalisation.syntactic_matching.elastic_bm25 import (
    ElasticBM25Index,
)
from concept_normalisation.syntactic_matching.elastic_matcher import (
    fuzzy_match,
    multi_match,
)
from concept_normalisation.syntactic_matching.fuzzy_matcher import (
    JaccardMatcher,
)
from concept_normalisation.syntactic_matching.text_preprocessing import (
    TextPreprocessor,
)


def run_multi_match(
    data: pd.DataFrame,
    query_column: str,
    elastic_index: ElasticBM25Index,
    top_k: int = 5,
    preprocessor: TextPreprocessor | None = None,
    output_column: str = "multi_match_matches",
) -> pd.DataFrame:

    result = data.copy()

    queries = (
        result[query_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    lookup = {}

    for query in queries:
        lookup[query] = multi_match(
            query=query,
            index=elastic_index,
            k=top_k,
            preprocessor=preprocessor,
        )

    result[output_column] = result[query_column].map(lookup)

    return result


def run_elastic_fuzzy(
    data: pd.DataFrame,
    query_column: str,
    elastic_index: ElasticBM25Index,
    top_k: int = 5,
    preprocessor: TextPreprocessor | None = None,
    output_column: str = "elastic_fuzzy_matches",
) -> pd.DataFrame:

    result = data.copy()

    queries = (
        result[query_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    lookup = {}

    for query in queries:
        lookup[query] = fuzzy_match(
            query=query,
            index=elastic_index,
            k=top_k,
            preprocessor=preprocessor,
        )

    result[output_column] = result[query_column].map(lookup)

    return result


def run_jaccard(
    data: pd.DataFrame,
    query_column: str,
    matcher: JaccardMatcher,
    top_k: int = 5,
    output_column: str = "jaccard_matches",
) -> pd.DataFrame:
    # NOTE: `matcher` should be built with `index_path=...` (see
    # JaccardMatcher) so its n-gram index is loaded from a premade
    # parquet cache instead of rebuilt every run.

    result = data.copy()

    queries = (
        result[query_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    lookup = {}

    for query in queries:
        lookup[query] = matcher.match(
            query=query,
            k=top_k,
        )

    result[output_column] = result[query_column].map(lookup)

    return result
