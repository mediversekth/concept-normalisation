"""
Standalone Jaccard matching over the SNOMED candidate table.

This method does not use Elasticsearch. It compares the cleaned query
against SNOMED candidate terms using character n-gram Jaccard similarity.


1. Computes each candidate's n-gram set once.
2. Builds an inverted index (ngram -> set of candidate row indices).
3. At query time, only scores the candidates that share at least one
   n-gram with the query -- for reasonably specific queries (3+ chars)
   against a large candidate table this is normally a tiny fraction of
   the full table, not all of it.

The inverted index can be persisted to parquet with `index_path=...`, so
on the *first* run it gets built and saved, and every run after that
just loads it back instead of rebuilding it.
"""

from heapq import nlargest
from pathlib import Path

import pandas as pd


def _ngrams(text: str, n: int) -> set[str]:
    """Character n-grams of `text`. Falls back to the whole string for
    text shorter than n, so short candidates/queries still get a
    (single-element) n-gram set instead of an empty one."""

    if len(text) < n:
        return {text} if text else set()

    return {text[i:i + n] for i in range(len(text) - n + 1)}


class JaccardMatcher:
    """
    Match queries to candidate terms using character n-gram Jaccard
    similarity, backed by a cached inverted n-gram index.

    Parameters
    ----------
    candidates:
        DataFrame containing the SNOMED candidate terms.

    term_column:
        Name of the column containing candidate text.

    id_column:
        Name of the column containing SNOMED concept IDs.

    preprocessor:
        Optional TextPreprocessor. The same preprocessing is applied to
        both queries and candidate terms.

    ngram_size:
        Size of the character n-grams used by Jaccard. A value of 3
        means that strings are compared using character trigrams.

    index_path:
        Optional path to a parquet file holding a premade inverted
        n-gram index for this candidate table. If the file exists, it
        is loaded instead of being rebuilt. If it doesn't exist yet,
        the index is built from `candidates` and then saved there, so
        the next JaccardMatcher created against the same candidate
        table (and same ngram_size) can skip straight to loading it.
        Pass None (the default) to build the index in memory only,
        without reading/writing anything to disk.
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        term_column: str = "description",
        id_column: str = "concept_id",
        preprocessor=None,
        ngram_size: int = 3,
        index_path: str | Path | None = None,
    ):
        required_columns = {term_column, id_column}
        missing_columns = required_columns - set(candidates.columns)

        if missing_columns:
            raise ValueError(
                f"Candidate table is missing columns: {sorted(missing_columns)}"
            )

        if ngram_size < 1:
            raise ValueError("ngram_size must be at least 1")

        self.candidates = candidates.reset_index(drop=True)
        self.term_column = term_column
        self.id_column = id_column
        self.preprocessor = preprocessor
        self.ngram_size = ngram_size
        self.index_path = Path(index_path) if index_path else None

        # Candidate text is cleaned once here rather than once per query.
        self.cleaned_terms = [
            self._clean(term)
            for term in self.candidates[self.term_column].astype(str)
        ]

        if self.index_path is not None and self.index_path.exists():
            print(f"Loading premade Jaccard n-gram index from {self.index_path}")
            self._load_index(self.index_path)
        else:
            self._build_index()

            if self.index_path is not None:
                self.save_index(self.index_path)

    def _clean(self, text) -> str:
        """Apply the configured text preprocessing."""

        if self.preprocessor is not None:
            return self.preprocessor.clean(text)

        return str(text).strip().lower()

    def _build_index(self) -> None:
        """Compute every candidate's n-gram set and the ngram -> candidate
        inverted index, once."""

        print(f"Building Jaccard n-gram index for {len(self.candidates):,} candidates...")

        self.candidate_ngrams: list[set[str]] = [
            _ngrams(term, self.ngram_size) for term in self.cleaned_terms
        ]

        inverted: dict[str, set[int]] = {}

        for candidate_index, ngrams in enumerate(self.candidate_ngrams):
            for ngram in ngrams:
                inverted.setdefault(ngram, set()).add(candidate_index)

        self.inverted_index = inverted

        print(f"Jaccard index: {len(inverted):,} distinct {self.ngram_size}-grams")

    def save_index(self, path: str | Path) -> None:
        """
        Persist the inverted n-gram index to parquet (one row per
        ngram/candidate_index pair) so it can be loaded back with
        `index_path=...` instead of rebuilt next time.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {"ngram": ngram, "candidate_index": candidate_index}
            for ngram, candidate_indices in self.inverted_index.items()
            for candidate_index in candidate_indices
        ]

        pd.DataFrame(rows).to_parquet(path)
        print(f"Saved Jaccard n-gram index to {path}")

    def _load_index(self, path: str | Path) -> None:
        """Load a premade inverted index from parquet. Candidate n-gram
        sets are cheaply rebuilt from the already-cleaned candidate text
        (no pairwise comparison happens here) since they're needed at
        query time to compute exact intersection/union sizes."""

        index_table = pd.read_parquet(path)

        inverted: dict[str, set[int]] = {
            ngram: set(group.tolist())
            for ngram, group in index_table.groupby("ngram")["candidate_index"]
        }

        self.inverted_index = inverted
        self.candidate_ngrams = [
            _ngrams(term, self.ngram_size) for term in self.cleaned_terms
        ]

    def match(self, query: str, k: int = 10) -> list[dict]:
        """
        Return the top-k candidates ranked by Jaccard similarity.

        Scores are returned on a 0-1 scale:
            1.0 = identical n-gram sets
            0.0 = no shared n-grams

        Only candidates that share at least one n-gram with the query are
        scored -- every other candidate has a Jaccard score of 0 anyway,
        so skipping them changes nothing about the ranking, just how long
        it takes to get there.
        """

        if k < 1:
            raise ValueError("k must be at least 1")

        query_clean = self._clean(query)

        if not query_clean:
            return []

        query_ngrams = _ngrams(query_clean, self.ngram_size)

        if not query_ngrams:
            return []

        candidate_pool: set[int] = set()

        for ngram in query_ngrams:
            candidate_pool.update(self.inverted_index.get(ngram, ()))

        if not candidate_pool:
            return []

        query_size = len(query_ngrams)

        def score(candidate_index: int) -> float:
            candidate_ngrams = self.candidate_ngrams[candidate_index]
            intersection = len(query_ngrams & candidate_ngrams)
            union = query_size + len(candidate_ngrams) - intersection
            return intersection / union if union else 0.0

        top_candidates = nlargest(
            min(k, len(candidate_pool)),
            ((score(candidate_index), candidate_index) for candidate_index in candidate_pool),
            key=lambda result: result[0],
        )

        matches = []

        for rank, (jaccard_score, candidate_index) in enumerate(top_candidates, start=1):
            candidate = self.candidates.iloc[candidate_index]

            matches.append(
                {
                    "rank": rank,
                    "concept_id": candidate[self.id_column],
                    "term": candidate[self.term_column],
                    "jaccard_score": float(jaccard_score),
                }
            )

        return matches
