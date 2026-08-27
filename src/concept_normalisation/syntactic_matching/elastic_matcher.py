"""
Format Elasticsearch multi-match and fuzzy-search results.
"""

from concept_normalisation.syntactic_matching.elastic_bm25 import ElasticBM25Index


def _prepare_query(query, preprocessor=None) -> str:
    """Clean one query before sending it to Elasticsearch."""

    if preprocessor is not None:
        return preprocessor.clean(query)

    return str(query).strip()


def _format_hits(hits: list[dict]) -> list[dict]:
    """
    Convert raw Elasticsearch hits into the common match-dictionary format.
    """

    results = []

    for rank, hit in enumerate(hits, start=1):
        source = hit["_source"]

        results.append(
            {
                "rank": rank,
                "concept_id": source["conceptId"],
                "term": source["term"],
                "score": float(hit["_score"]),
            }
        )

    return results


def multi_match(
    query: str,
    index: ElasticBM25Index,
    k: int = 10,
    preprocessor=None,
) -> list[dict]:
    """
    Standard Elasticsearch multi-match/BM25 retrieval.
    """

    query_clean = _prepare_query(query, preprocessor)

    if not query_clean:
        return []

    hits = index.multi_match_search(
        query=query_clean,
        k=k,
    )

    return _format_hits(hits)


def fuzzy_match(
    query: str,
    index: ElasticBM25Index,
    k: int = 10,
    preprocessor=None,
) -> list[dict]:
    """
    Elasticsearch-native fuzzy full-text retrieval.
    """

    query_clean = _prepare_query(query, preprocessor)

    if not query_clean:
        return []

    hits = index.fuzzy_search(
        query=query_clean,
        k=k,
    )

    return _format_hits(hits)
