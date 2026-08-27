"""
SNOMED CT hierarchy utilities.

SNOMED CT "Is a" relationships form a hierarchy where a concept can have
one or more parent concepts.

For example:

    Ventricular tachycardia
              |
              | Is a
              v
        Tachyarrhythmia
              |
              v
          Arrhythmia


For consensus between matching methods, we want to know whether two
candidate concepts are located in the same nearby area of the SNOMED
hierarchy.

Two concepts are therefore considered related when:

1. They are the exact same concept, OR
2. One is an ancestor of the other, OR
3. They share a common ancestor within `max_depth` hops.

Example:

              Cardiac arrhythmia
                 /          \
                /            \
        Candidate A       Candidate B

Candidate A and Candidate B are considered related because they share
the nearby common ancestor "Cardiac arrhythmia".

The max_depth parameter prevents concepts from being considered related
only because they eventually meet at an extremely broad SNOMED ancestor.
"""

from collections import deque

import pandas as pd

from concept_normalisation.config import (
    RELATIONSHIP_FILE,
    IS_A_TYPE_ID,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_CONCEPT_LABEL,
    NEO4J_CONCEPT_ID_PROP,
    NEO4J_ISA_REL_TYPE,
)


try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


# ============================================================
# RF2 / in-memory hierarchy
# ============================================================

class SnomedHierarchy:
    """
    SNOMED hierarchy backed by the RF2 Relationship Snapshot file.

    The RF2 "Is a" relationships are stored as:

        sourceId       = child concept
        destinationId  = parent concept

    After loading, self.parents looks like:

        child_concept_id -> {parent_1, parent_2, ...}
    """

    def __init__(
        self,
        relationship_file=RELATIONSHIP_FILE,
        is_a_type_id=IS_A_TYPE_ID,
    ):
        self.relationship_file = relationship_file
        self.is_a_type_id = str(is_a_type_id)

        # child concept_id -> direct parent concept_ids
        self.parents = {}

        self._loaded = False

    # ========================================================
    # Load hierarchy
    # ========================================================

    def load(self):
        """
        Load active SNOMED CT "Is a" relationships from the RF2
        Relationship Snapshot file.

        Only active relationships whose typeId corresponds to
        SNOMED's "Is a" relationship are retained.
        """

        if self._loaded:
            return self.parents

        print(
            f"Loading SNOMED relationships from "
            f"{self.relationship_file}..."
        )

        relationships = pd.read_csv(
            self.relationship_file,
            sep="\t",
            dtype=str,
        )

        is_a = relationships[
            (relationships["active"] == "1")
            & (
                relationships["typeId"]
                == self.is_a_type_id
            )
        ]

        print(
            f"Active 'Is a' relationships: "
            f"{len(is_a):,}"
        )

        self.parents = {}

        for source_id, destination_id in zip(
            is_a["sourceId"],
            is_a["destinationId"],
        ):
            # sourceId      = child
            # destinationId = parent

            self.parents.setdefault(
                str(source_id),
                set(),
            ).add(
                str(destination_id)
            )

        self._loaded = True

        return self.parents

    # ========================================================
    # Ancestors
    # ========================================================

    def get_ancestors(
        self,
        concept_id,
        max_depth=None,
    ) -> set:
        """
        Return all ancestors of a SNOMED concept.

        The starting concept itself is NOT included.

        Parameters
        ----------
        concept_id:
            SNOMED concept ID.

        max_depth:
            Maximum number of parent hops.

            Example:

                max_depth=1
                    -> direct parents only

                max_depth=2
                    -> parents + grandparents

                max_depth=None
                    -> walk all the way toward the root
        """

        ancestor_distances = (
            self.get_ancestors_with_distance(
                concept_id,
                max_depth=max_depth,
                include_self=False,
            )
        )

        return set(
            ancestor_distances.keys()
        )

    # ========================================================
    # Ancestors + distances
    # ========================================================

    def get_ancestors_with_distance(
        self,
        concept_id,
        max_depth=None,
        include_self=True,
    ) -> dict:
        """
        Return ancestors together with their shortest distance from
        the starting concept.

        Example:

            {
                "123": 0,   # starting concept
                "456": 1,   # parent
                "789": 2,   # grandparent
            }

        SNOMED concepts may have several parents, so breadth-first
        search is used to ensure that the stored distance is the
        shortest hierarchy distance.
        """

        if not self._loaded:
            self.load()

        concept_id = str(concept_id)

        distances = {}

        if include_self:
            distances[concept_id] = 0

        # Queue contains:
        #
        # (concept_id, distance_from_start)
        queue = deque([
            (concept_id, 0)
        ])

        # Prevent repeatedly traversing the same node
        visited = {
            concept_id
        }

        while queue:

            current, current_depth = queue.popleft()

            # Stop expanding once we reach max_depth
            if (
                max_depth is not None
                and current_depth >= max_depth
            ):
                continue

            for parent in self.parents.get(
                current,
                set(),
            ):

                parent = str(parent)

                next_depth = current_depth + 1

                # Store shortest known distance
                if (
                    parent not in distances
                    or next_depth < distances[parent]
                ):
                    distances[parent] = next_depth

                if parent not in visited:

                    visited.add(parent)

                    queue.append(
                        (
                            parent,
                            next_depth,
                        )
                    )

        # The caller may explicitly not want the original concept
        if not include_self:
            distances.pop(
                concept_id,
                None,
            )

        return distances

    # ========================================================
    # Common ancestors
    # ========================================================

    def find_common_ancestors(
        self,
        concept_a,
        concept_b,
        max_depth=3,
    ) -> list:
        """
        Find common ancestors of two SNOMED concepts.

        Both concepts themselves are included at distance 0. This is
        important because it naturally also handles ancestor/descendant
        relationships.

        Example:

                X
               / \\
              A   B

        gives:

            X:
                distance_a = 1
                distance_b = 1
                total_distance = 2

        Results are sorted so that the nearest common ancestor appears
        first.
        """

        concept_a = str(concept_a)
        concept_b = str(concept_b)

        ancestors_a = (
            self.get_ancestors_with_distance(
                concept_a,
                max_depth=max_depth,
                include_self=True,
            )
        )

        ancestors_b = (
            self.get_ancestors_with_distance(
                concept_b,
                max_depth=max_depth,
                include_self=True,
            )
        )

        common_ids = (
            set(ancestors_a)
            & set(ancestors_b)
        )

        results = []

        for ancestor_id in common_ids:

            distance_a = (
                ancestors_a[ancestor_id]
            )

            distance_b = (
                ancestors_b[ancestor_id]
            )

            results.append({
                "ancestor": ancestor_id,
                "distance_a": distance_a,
                "distance_b": distance_b,
                "total_distance": (
                    distance_a
                    + distance_b
                ),
            })

        # Nearest common ancestor first
        results.sort(
            key=lambda result: (
                result["total_distance"],
                max(
                    result["distance_a"],
                    result["distance_b"],
                ),
            )
        )

        return results

    # ========================================================
    # Best / nearest common ancestor
    # ========================================================

    def get_nearest_common_ancestor(
        self,
        concept_a,
        concept_b,
        max_depth=3,
    ):
        """
        Return only the nearest common ancestor.

        Returns None when no common ancestor exists within max_depth.

        Otherwise returns:

            {
                "ancestor": "...",
                "distance_a": ...,
                "distance_b": ...,
                "total_distance": ...
            }
        """

        common = self.find_common_ancestors(
            concept_a,
            concept_b,
            max_depth=max_depth,
        )

        if not common:
            return None

        return common[0]

    # ========================================================
    # Consensus relationship check
    # ========================================================

    def is_related(
        self,
        concept_a,
        concept_b,
        max_depth=3,
    ) -> bool:
        """
        Return True when two concepts are hierarchically related.

        Concepts count as related when:

        1. They are identical, OR
        2. One is an ancestor of the other, OR
        3. They share a common ancestor within max_depth.

        Because each starting concept is included at distance 0 in
        find_common_ancestors(), cases 1 and 2 are naturally covered
        by the same common-ancestor logic.
        """

        if concept_a is None or concept_b is None:
            return False

        concept_a = str(concept_a)
        concept_b = str(concept_b)

        if concept_a == concept_b:
            return True

        common = self.find_common_ancestors(
            concept_a,
            concept_b,
            max_depth=max_depth,
        )

        return len(common) > 0


# ============================================================
# Neo4j hierarchy
# ============================================================

class Neo4jHierarchy:
    """
    SNOMED hierarchy backed by Neo4j.

    This provides the same is_related() interface as SnomedHierarchy,
    but checks for a nearby common ancestor directly in Neo4j.

    The expected graph direction is:

        child -[:ISA]-> parent

    which matches the RF2 representation used by SnomedHierarchy.
    """

    def __init__(
        self,
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        concept_label=NEO4J_CONCEPT_LABEL,
        concept_id_prop=NEO4J_CONCEPT_ID_PROP,
        isa_rel_type=NEO4J_ISA_REL_TYPE,
    ):
        if GraphDatabase is None:
            raise ImportError(
                "pip install neo4j to use Neo4jHierarchy"
            )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(
                user,
                password,
            ),
        )

        self.concept_label = concept_label
        self.concept_id_prop = concept_id_prop
        self.isa_rel_type = isa_rel_type

    def close(self):
        self.driver.close()

    # ========================================================
    # Common ancestor lookup
    # ========================================================

    def find_common_ancestors(
        self,
        concept_a,
        concept_b,
        max_depth=3,
        limit=20,
    ) -> list:
        """
        Find common ancestors of two concepts in Neo4j.

        Results are ordered by the total path length from both
        candidates to the common ancestor.
        """

        concept_a = str(concept_a)
        concept_b = str(concept_b)

        if max_depth is None:
            depth_range = "*0.."
        else:
            depth_range = (
                f"*0..{int(max_depth)}"
            )

        query = f"""
        MATCH
            (a:{self.concept_label}
                {{{self.concept_id_prop}: $id_a}}),

            (b:{self.concept_label}
                {{{self.concept_id_prop}: $id_b}})

        MATCH
            p1 = (a)-[:{self.isa_rel_type}{depth_range}]->(common)

        MATCH
            p2 = (b)-[:{self.isa_rel_type}{depth_range}]->(common)

        WITH
            common,
            min(length(p1)) AS distance_a,
            min(length(p2)) AS distance_b

        RETURN
            common.{self.concept_id_prop} AS ancestor,
            distance_a,
            distance_b,
            distance_a + distance_b AS total_distance

        ORDER BY
            total_distance ASC,
            distance_a ASC,
            distance_b ASC

        LIMIT $limit
        """

        with self.driver.session() as session:

            records = session.run(
                query,
                id_a=concept_a,
                id_b=concept_b,
                limit=int(limit),
            )

            return [
                {
                    "ancestor": str(
                        record["ancestor"]
                    ),
                    "distance_a": int(
                        record["distance_a"]
                    ),
                    "distance_b": int(
                        record["distance_b"]
                    ),
                    "total_distance": int(
                        record["total_distance"]
                    ),
                }
                for record in records
            ]

    # ========================================================
    # Nearest common ancestor
    # ========================================================

    def get_nearest_common_ancestor(
        self,
        concept_a,
        concept_b,
        max_depth=3,
    ):
        """
        Return the nearest common ancestor of two concepts.
        """

        common = self.find_common_ancestors(
            concept_a,
            concept_b,
            max_depth=max_depth,
            limit=1,
        )

        if not common:
            return None

        return common[0]

    # ========================================================
    # Consensus relationship check
    # ========================================================

    def is_related(
        self,
        concept_a,
        concept_b,
        max_depth=3,
    ) -> bool:
        """
        Return True when the two concepts have a common ancestor within
        max_depth hops from each concept.
        """

        if concept_a is None or concept_b is None:
            return False

        concept_a = str(concept_a)
        concept_b = str(concept_b)

        if concept_a == concept_b:
            return True

        nearest = (
            self.get_nearest_common_ancestor(
                concept_a,
                concept_b,
                max_depth=max_depth,
            )
        )

        return nearest is not None
