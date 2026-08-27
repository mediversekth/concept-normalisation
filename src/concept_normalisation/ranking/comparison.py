"""
Build one flat comparison table: for the same query, what did every
matching method (SapBERT, BioLORD, BioLORD+AI-context, exact BM25,
BM25+fuzzy, Levenshtein, Jaccard) pick as its #1 (best) match? One row per
query, one set of columns per method, so you can scan across a row and see
at a glance whether the methods agree.

Every method in this project returns its matches in a slightly different
shape, because each one came from a different original script with its own
key names. This file's only job is: pull the #1 match out of each method's
list and rename its fields consistently -- no new matching logic happens
here.
"""

import pandas as pd


def _top_match(matches, id_key, text_key, score_key):
    """Grab the #1 (best-ranked) match out of one method's list of
    match-dicts. Every matcher already returns its matches best-first, so
    "top match" is always just matches[0]. Returns (None, None, None) if
    that method found nothing for this query.

    Uses len(matches) == 0 rather than `if not matches` because a column
    that's been saved to parquet and read back can come back as a numpy
    array instead of a plain Python list -- and numpy raises a ValueError
    on `if not <array with 2+ elements>` since it doesn't know whether you
    mean "any" or "all". len() works the same way for both a list and an
    array, so it sidesteps that entirely."""
    if matches is None or len(matches) == 0:
        return None, None, None
    top = matches[0]
    return top.get(id_key), top.get(text_key), top.get(score_key)


def build_comparison_table(semantic_data: pd.DataFrame, query_column: str = "test_name") -> pd.DataFrame:
    """
    Takes the semantic_data dataframe after every matching step has run
    (dense + syntactic) and returns a new, flat dataframe: one row per
    query, with the top match from every method as its own
    <method>_concept_id / <method>_match / <method>_score columns.

    Expects these columns to already exist on semantic_data -- this is the
    naming pipeline/semantic.py and pipeline/syntactic.py (and
    evaluation/consensus.py's DEFAULT_METHOD_COLUMNS) actually produce:
      - algorithm_1_matches      (SapBERT, run_algorithm_1)
      - algorithm_2_matches      (BioLORD, run_algorithm_2)
      - algorithm_ai_matches     (BioLORD on AI-generated context, run_algorithm_ai -- optional)
      - multi_match_matches      (Elasticsearch multi-match/BM25)
      - elastic_fuzzy_matches    (Elasticsearch fuzzy full-text)
      - jaccard_matches          (JaccardMatcher)
    """

    rows = []
    for _, row in semantic_data.iterrows():
        algo1_id, algo1_text, algo1_score = _top_match(
            row.get("algorithm_1_matches", []), "concept_id", "description", "similarity"
        )
        algo2_id, algo2_text, algo2_score = _top_match(
            row.get("algorithm_2_matches", []), "concept_id", "text", "similarity"
        )
        ai_id, ai_text, ai_score = _top_match(
            row.get("algorithm_ai_matches", []), "concept_id", "text", "similarity"
        )
        multi_id, multi_text, multi_score = _top_match(
            row.get("multi_match_matches", []), "concept_id", "term", "score",
        )

        fuzzy_id, fuzzy_text, fuzzy_score = _top_match(
            row.get("elastic_fuzzy_matches", []), "concept_id", "term", "score",
        )

        jaccard_id, jaccard_text, jaccard_score = _top_match(
            row.get("jaccard_matches", []), "concept_id", "term", "jaccard_score",
        )

        rows.append({
            "test_name": row[query_column],

            "algorithm_1_concept_id": algo1_id,
            "algorithm_1_match": algo1_text,
            "algorithm_1_score": algo1_score,

            "algorithm_2_concept_id": algo2_id,
            "algorithm_2_match": algo2_text,
            "algorithm_2_score": algo2_score,

            "algorithm_ai_concept_id": ai_id,
            "algorithm_ai_match": ai_text,
            "algorithm_ai_score": ai_score,

            "multi_match_concept_id": multi_id,
            "multi_match_match": multi_text,
            "multi_match_score": multi_score,

            "elastic_fuzzy_concept_id": fuzzy_id,
            "elastic_fuzzy_match": fuzzy_text,
            "elastic_fuzzy_score": fuzzy_score,

            "jaccard_concept_id": jaccard_id,
            "jaccard_match": jaccard_text,
            "jaccard_score": jaccard_score,
                    })

    return pd.DataFrame(rows)
