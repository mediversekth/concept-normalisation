"""
Dense (embedding) index over a SNOMED candidate pool, with batched top-k
cosine-similarity search.

vectors -> top-k matches

"""

import numpy as np
import pandas as pd


class DenseIndex:
    """
    Holds one SNOMED candidate pool (e.g. all short terms, or short terms +
    definitions combined) and its embedding matrix, and answers the
    question: "given a query fingerprint, which candidates are closest?"
    """

    def __init__(self, candidates: pd.DataFrame, embeddings: np.ndarray, text_column: str = "description"):
        # Sanity check: one embedding row per candidate row, or something's misaligned.
        assert len(candidates) == embeddings.shape[0]

        # reset_index so "row i in candidates" always lines up with "row i in embeddings",
        # even if `candidates` was filtered/concatenated earlier and has gappy indices.
        self.candidates = candidates.reset_index(drop=True)

        # The raw embedding matrix: shape (n_candidates, embedding_dim), e.g. (300_000, 768).
        # Row i is the fingerprint for self.candidates.iloc[i].
        self.embeddings = embeddings

        # Pre-transpose once here, not on every search: matching a query batch against
        # every candidate is `query_embeddings @ embeddings.T`, so we flip the matrix
        # sideways up front (embedding_dim, n_candidates) and reuse this every call to
        # match_batch() instead of re-flipping a 300,000-row matrix each time.
        self.embeddings_T = embeddings.T

        # Which column of `candidates` holds the human-readable text for each row --
        # "description" for the short-terms pool, "text" for the combined pool.
        self.text_column = text_column

        print(f"Candidate pool: {len(self.candidates):,} rows")

    def match_batch(self, query_embeddings: np.ndarray, top_k: int = 5, query_batch_size: int = 50, extra_columns=None):
        """
        For every query fingerprint in query_embeddings, find the top_k closest
        candidate fingerprints in this pool. Processes queries in batches of
        query_batch_size so we're never holding an enormous similarity matrix
        (n_queries x n_candidates) in memory all at once.

        extra_columns: any extra candidate columns to copy into each match dict,
        e.g. ["source"] to say whether a BioLORD match came from a short term
        or a definition.

        Returns: a list with one entry per query, each entry a list of top_k
        match dicts, ordered best-to-worst by similarity.
        """
        extra_columns = extra_columns or []
        matches = []
        n_queries = query_embeddings.shape[0]

        # --- Walk through the queries in chunks of query_batch_size ---
        for start in range(0, n_queries, query_batch_size):
            end = min(start + query_batch_size, n_queries)
            batch = query_embeddings[start:end]  # shape: (batch_size, embedding_dim)

            # THE CORE MATH: matrix-multiply this batch of query fingerprints against
            # every candidate fingerprint in one shot. Because every fingerprint was
            # L2-normalized when it was created (see the embedders), this dot product
            # IS the cosine similarity -- 1.0 means "point in exactly the same
            # direction / same meaning", 0 means "unrelated".
            # Result shape: (batch_size, n_candidates) -- one row per query, one
            # column per candidate, e.g. (50, 300_000).
            similarities = batch @ self.embeddings_T

            # For each query we only want the best top_k out of n_candidates.
            # Fully sorting 300,000 numbers per query just to keep the top 5 would be
            # wasteful. np.argpartition instead does a cheap partial rearrangement:
            # it guarantees the top_k largest values end up in the first top_k
            # positions, but doesn't bother sorting *within* that slice.
            # (We negate `similarities` because argpartition finds the smallest
            # values by default, and we want the largest similarities.)
            top_k_idx = np.argpartition(-similarities, top_k, axis=1)[:, :top_k]

            # --- Now handle each query in this batch individually ---
            for row_i in range(similarities.shape[0]):
                idxs = top_k_idx[row_i]              # the (unsorted) top_k candidate indices for this query
                row_sims = similarities[row_i, idxs]  # their similarity scores

                # Now that we've narrowed down to just top_k candidates, sorting is
                # cheap (only top_k items, not n_candidates), so rank them properly
                # best-to-worst.
                order = np.argsort(-row_sims)
                idxs = idxs[order]
                row_sims = row_sims[order]

                # Look up the actual candidate rows for these top_k indices and
                # package each one into a readable dict: which concept it is, its
                # text, any extra columns (like "source"), and the similarity score.
                row_matches = [
                    {
                        "concept_id": self.candidates.iloc[i]["concept_id"],
                        self.text_column: self.candidates.iloc[i][self.text_column],
                        **{col: self.candidates.iloc[i][col] for col in extra_columns},
                        "similarity": float(sim),
                    }
                    for i, sim in zip(idxs, row_sims)
                ]
                matches.append(row_matches)

            # Just a progress print so long runs don't look frozen.
            if start % (query_batch_size * 5) == 0:
                print(f"  matched {end:,}/{n_queries:,} queries")

        return matches


def load_dense_index_from_parquet(parquet_path, embeddings_path, text_column: str = "description") -> DenseIndex:
    """
    Convenience loader: reads a candidates parquet + its matching embeddings
    .npy file off disk and builds a DenseIndex from them. Not a method on
    DenseIndex -- just a plain function, since it doesn't need `self` or `cls`,
    it just needs to hand back a DenseIndex.
    """
    candidates = pd.read_parquet(parquet_path).reset_index(drop=True)
    embeddings = np.load(embeddings_path)
    return DenseIndex(candidates, embeddings, text_column=text_column)


def build_combined_pool(short_terms: pd.DataFrame, short_term_embeddings: np.ndarray,
                         definitions: pd.DataFrame, definition_embeddings: np.ndarray) -> DenseIndex:
    """
    Same combined pool as semantic_analysis_pipeline.py step 6: short terms
    give near-total concept coverage since only a small fraction of concepts
    have a text definition at all, so short terms + definitions are stacked
    into one candidate pool with a "source" column marking which is which,
    and one DenseIndex is built over the combined result.
    """
    # Give both tables the same column name ("text") so they can be stacked
    # on top of each other, and tag each row with where it came from.
    short_terms_unified = short_terms.rename(columns={"description": "text"})[
        ["concept_id", "text"]
    ].copy()
    short_terms_unified["source"] = "short_term"

    definitions_unified = definitions.rename(columns={"definition": "text"})[
        ["concept_id", "text"]
    ].copy()
    definitions_unified["source"] = "definition"

    # Stack the two candidate tables into one long table...
    combined_candidates = pd.concat([short_terms_unified, definitions_unified], ignore_index=True)
    # ...and stack their embedding matrices the same way, so row i of
    # combined_embeddings still lines up with row i of combined_candidates.
    combined_embeddings = np.vstack([short_term_embeddings, definition_embeddings])

    assert len(combined_candidates) == combined_embeddings.shape[0]
    print(f"Combined BioLORD pool: {len(combined_candidates):,} rows "
          f"({len(short_terms):,} short terms + {len(definitions):,} definitions)")

    return DenseIndex(combined_candidates, combined_embeddings, text_column="text")
