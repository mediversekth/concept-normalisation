"""
Build final consensus candidates per query by combining votes from
all matching methods.

Each method contributes its top-k candidates. Votes are weighted by rank.

Hierarchically related SNOMED concepts are clustered together. For each
cluster, the consensus representative is chosen as:

1. A nearby common ancestor shared by the cluster members, when one exists.
2. Otherwise, the strongest exact concept in the cluster.

The final output keeps both:
- rank-1 final_concept_id
- a ranked top-k list of final consensus concepts
"""

import itertools

import pandas as pd


DEFAULT_METHOD_COLUMNS = {
    "algorithm_1": (
        "algorithm_1_matches",
        "concept_id",
        "description",
    ),
    "algorithm_2": (
        "algorithm_2_matches",
        "concept_id",
        "text",
    ),
    "algorithm_ai": (
        "algorithm_ai_matches",
        "concept_id",
        "text",
    ),
    "multi_match": (
        "multi_match_matches",
        "concept_id",
        "term",
    ),
    "elastic_fuzzy": (
        "elastic_fuzzy_matches",
        "concept_id",
        "term",
    ),
    "jaccard": (
        "jaccard_matches",
        "concept_id",
        "term",
    ),
}


def _top_k_matches(matches, id_key, text_key, k):
    """
    Return the top-k matches from one method as:

        [(concept_id, text), ...]

    Works with both Python lists and NumPy-array-like values after
    parquet round-trips.
    """

    if matches is None or len(matches) == 0:
        return []

    return [
        (match.get(id_key), match.get(text_key))
        for match in list(matches)[:k]
    ]


class _UnionFind:
    """
    Small union-find structure used to merge related SNOMED concepts
    into hierarchy clusters.
    """

    def __init__(self, items):
        self.parent = {
            item: item
            for item in items
        }

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[
                self.parent[item]
            ]
            item = self.parent[item]

        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_a] = root_b


def _find_cluster_common_ancestor(
    members,
    hierarchy,
    max_depth,
):
    """
    Find the best common ancestor shared by ALL concepts in a cluster.

    Returns None if:
    - there is no hierarchy
    - the cluster has fewer than 2 members
    - there is no common ancestor within max_depth

    The selected common ancestor minimizes total hierarchy distance
    from all cluster members.
    """

    if hierarchy is None:
        return None

    if len(members) < 2:
        return None

    ancestor_maps = []

    for concept_id in members:

        ancestors = hierarchy.get_ancestors_with_distance(
            concept_id,
            max_depth=max_depth,
            include_self=True,
        )

        ancestor_maps.append(ancestors)

    # Concepts shared by ALL member ancestor sets
    common_ids = set(
        ancestor_maps[0].keys()
    )

    for ancestor_map in ancestor_maps[1:]:
        common_ids &= set(
            ancestor_map.keys()
        )

    if not common_ids:
        return None

    ranked_common_ancestors = []

    for ancestor_id in common_ids:

        distances = [
            ancestor_map[ancestor_id]
            for ancestor_map in ancestor_maps
        ]

        ranked_common_ancestors.append({
            "concept_id": ancestor_id,
            "distances": distances,
            "total_distance": sum(distances),
            "max_distance": max(distances),
        })

    ranked_common_ancestors.sort(
        key=lambda x: (
            x["total_distance"],
            x["max_distance"],
        )
    )

    return ranked_common_ancestors[0]


def build_final_candidates(
    semantic_data: pd.DataFrame,
    hierarchy=None,
    top_k: int = 3,
    final_top_k: int = 5,
    max_depth: int = 3,
    method_columns=None,
    query_column: str = "test_name",
) -> pd.DataFrame:
    """
    Build ranked final consensus SNOMED candidates for every query.

    Parameters
    ----------
    semantic_data:
        DataFrame containing all individual method candidate lists.

    hierarchy:
        SnomedHierarchy or Neo4jHierarchy instance.

    top_k:
        Number of candidates each method contributes.

    final_top_k:
        Number of consensus candidates retained in final output.

    max_depth:
        Maximum hierarchy depth used for:
        - deciding whether concepts belong to the same hierarchy cluster
        - finding a common ancestor representing that cluster

    method_columns:
        Mapping describing method result columns and candidate field names.

    query_column:
        Column containing the original query.

    Returns
    -------
    pd.DataFrame
        One row per query containing rank-1 consensus and final top-k.
    """

    method_columns = (
        method_columns
        or DEFAULT_METHOD_COLUMNS
    )

    # Cache pairwise hierarchy checks across rows
    relation_cache = {}

    def related(a, b):
        if hierarchy is None:
            return False

        a = str(a)
        b = str(b)

        key = tuple(
            sorted((a, b))
        )

        if key not in relation_cache:
            relation_cache[key] = hierarchy.is_related(
                a,
                b,
                max_depth=max_depth,
            )

        return relation_cache[key]

    rows = []
    n = len(semantic_data)

    for i, (_, row) in enumerate(
        semantic_data.iterrows()
    ):

        # ========================================================
        # 1. Gather weighted top-k votes from every method
        # ========================================================

        candidates = {}

        for method, (
            column,
            id_key,
            text_key,
        ) in method_columns.items():

            if column not in semantic_data.columns:
                continue

            matches = _top_k_matches(
                row.get(column),
                id_key=id_key,
                text_key=text_key,
                k=top_k,
            )

            for rank, (
                concept_id,
                text,
            ) in enumerate(matches):

                if concept_id is None:
                    continue

                concept_id = str(
                    concept_id
                )

                # top_k=3:
                # rank 1 -> 1.00
                # rank 2 -> 0.67
                # rank 3 -> 0.33
                weight = (
                    top_k - rank
                ) / top_k

                entry = candidates.setdefault(
                    concept_id,
                    {
                        "weight": 0.0,
                        "methods": set(),
                        "text": text,
                    },
                )

                entry["weight"] += weight
                entry["methods"].add(method)

        # ========================================================
        # 2. Handle rows with no candidates
        # ========================================================

        if not candidates:

            rows.append({
                "test_name": row[query_column],

                "final_concept_id": None,
                "final_match": None,
                "support_score": 0.0,
                "n_methods_agreeing": 0,
                "agreeing_methods": [],

                "final_candidate_ids": [],
                "final_candidate_matches": [],
                "final_candidate_scores": [],
                "final_candidate_sources": [],
            })

            if i % 20 == 0:
                print(
                    f"  final candidates: "
                    f"{i:,}/{n:,}"
                )

            continue

        # ========================================================
        # 3. Cluster hierarchically related concepts
        # ========================================================

        uf = _UnionFind(
            candidates.keys()
        )

        for concept_a, concept_b in itertools.combinations(
            candidates.keys(),
            2,
        ):

            if related(
                concept_a,
                concept_b,
            ):
                uf.union(
                    concept_a,
                    concept_b,
                )

        clusters = {}

        for concept_id in candidates:

            root = uf.find(
                concept_id
            )

            clusters.setdefault(
                root,
                [],
            ).append(
                concept_id
            )

        # ========================================================
        # 4. Score each cluster and build enriched candidate list
        # ========================================================

        ranked_candidates = []

        for root, members in clusters.items():

            cluster_score = sum(
                candidates[concept_id]["weight"]
                for concept_id in members
            )

            cluster_methods = set().union(
                *(
                    candidates[concept_id]["methods"]
                    for concept_id in members
                )
            )

            # ----------------------------------------------------
            # A. Keep EVERY original concept in the cluster
            # ----------------------------------------------------

            for concept_id in members:

                ranked_candidates.append({
                    "concept_id": concept_id,

                    "text": candidates[
                        concept_id
                    ]["text"],

                    # Support of the entire hierarchy cluster
                    "cluster_score": cluster_score,

                    # Direct support of this exact candidate
                    "concept_score": candidates[
                        concept_id
                    ]["weight"],

                    # Methods that contributed to the cluster
                    "methods": cluster_methods,

                    # This was actually returned by a matcher
                    "source": "exact_candidate",

                    # Which concepts form this cluster
                    "cluster_members": members,
                })

            # ----------------------------------------------------
            # B. ALSO add the common ancestor
            # ----------------------------------------------------

            common_ancestor = _find_cluster_common_ancestor(
                members=members,
                hierarchy=hierarchy,
                max_depth=max_depth,
            )

            if common_ancestor is not None:

                ancestor_id = str(
                    common_ancestor["concept_id"]
                )

                # Avoid adding it twice if the ancestor is already
                # one of the retrieved candidates
                if ancestor_id not in members:

                    ranked_candidates.append({
                        "concept_id": ancestor_id,

                        "text": (
                            f"Common ancestor {ancestor_id}"
                        ),

                        # Same cluster support
                        "cluster_score": cluster_score,

                        # It was not directly predicted by a method
                        "concept_score": 0.0,

                        "methods": cluster_methods,

                        "source": "common_ancestor",

                        "cluster_members": members,

                        "ancestor_total_distance": (
                            common_ancestor[
                                "total_distance"
                            ]
                        ),

                        "ancestor_max_distance": (
                            common_ancestor[
                                "max_distance"
                            ]
                        ),
                    })
        # ========================================================
        # 5. Rank cluster representatives
        # ========================================================

        ranked_candidates.sort(
            key=lambda candidate: (
                candidate["cluster_score"],
                candidate["concept_score"],
                len(candidate["methods"]),
            ),
            reverse=True,
        )

        # ========================================================
        # 6. Keep final top-k representatives
        # ========================================================

        final_ranked = (
            ranked_candidates[
                :final_top_k
            ]
        )

        winner = (
            final_ranked[0]
        )

        # ========================================================
        # 7. Save output
        # ========================================================

        rows.append({
            "test_name": row[query_column],

            # ----------------------------------------------------
            # Rank-1 final result
            # ----------------------------------------------------

            "final_concept_id": (
                winner["concept_id"]
            ),

            "final_match": (
                winner["text"]
            ),

            "support_score": round(
                winner["cluster_score"],
                3,
            ),

            "n_methods_agreeing": len(
                winner["methods"]
            ),

            "agreeing_methods": sorted(
                winner["methods"]
            ),

            # ----------------------------------------------------
            # Final top-k
            # ----------------------------------------------------

            "final_candidate_ids": [
                candidate["concept_id"]
                for candidate in final_ranked
            ],

            "final_candidate_matches": [
                candidate["text"]
                for candidate in final_ranked
            ],

            "final_candidate_scores": [
                round(
                    candidate["cluster_score"],
                    3,
                )
                for candidate in final_ranked
            ],

            # Tells you whether this output was an actual retrieved
            # candidate or a hierarchy-derived common ancestor.
            "final_candidate_sources": [
                candidate["source"]
                for candidate in final_ranked
            ],

            # Useful for debugging
            "final_candidate_cluster_members": [
                candidate["cluster_members"]
                for candidate in final_ranked
            ],

            "final_candidate_ancestor_distances": [
                {
                    "total": candidate.get(
                        "ancestor_total_distance"
                    ),
                    "max": candidate.get(
                        "ancestor_max_distance"
                    ),
                }
                for candidate in final_ranked
            ],
        })

        if i % 20 == 0:
            print(
                f"  final candidates: "
                f"{i:,}/{n:,}"
            )

    return pd.DataFrame(
        rows
    )
