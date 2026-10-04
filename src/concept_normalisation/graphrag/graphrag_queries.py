""

import json

# ==================================================================
# NEO4J CYPHER FOR RETRIEVING CONTEXT
# ==================================================================

def build_retrieval_query(use_score: bool = True) -> str:
    return f"""
RETURN
    node.sctid AS sctid,
    node.FSN AS fsn,
    node.embedding_text AS description,
    {"score," if use_score else ""}

    COLLECT {{
        MATCH (node)-[:ISA]->(parent:ObjectConcept)
        RETURN parent.FSN
        LIMIT 5
    }} AS parents,
    COLLECT {{
        MATCH (node)-[:ISA]->(parent:ObjectConcept)-[:ISA]->(grandparent:ObjectConcept)
        RETURN DISTINCT grandparent.FSN
        LIMIT 5
    }} AS grandparents,

    COLLECT {{
        MATCH (child:ObjectConcept)-[:ISA]->(node)
        RETURN DISTINCT child.FSN
        LIMIT 10
    }} AS children,

    COLLECT {{
        MATCH (node)-[:HAS_ROLE_GROUP]->(:RoleGroup)-[:FINDING_SITE]->(site:ObjectConcept)
        RETURN DISTINCT site.FSN
        LIMIT 3
    }} AS finding_sites,
    COLLECT {{
        MATCH (node)-[:HAS_ROLE_GROUP]->(:RoleGroup)-[:ASSOCIATED_MORPHOLOGY]->(morph:ObjectConcept)
        RETURN DISTINCT morph.FSN
        LIMIT 3
    }} AS morphologies,
    COLLECT {{
        MATCH (node)-[:HAS_ROLE_GROUP]->(:RoleGroup)-[:CAUSATIVE_AGENT]->(agent:ObjectConcept)
        RETURN DISTINCT agent.FSN
        LIMIT 3
    }} AS causative_agents,
    COLLECT {{
        MATCH (node)-[:HAS_ROLE_GROUP]->(:RoleGroup)-[:DUE_TO]->(cause:ObjectConcept)
        RETURN DISTINCT cause.FSN
        LIMIT 3
    }} AS due_to,
    COLLECT {{
        MATCH (node)-[:HAS_ROLE_GROUP]->(:RoleGroup)-[:CLINICAL_COURSE]->(course:ObjectConcept)
        RETURN DISTINCT course.FSN
        LIMIT 3
    }} AS clinical_course,
    COLLECT {{
        MATCH (node)-[:HAS_ROLE_GROUP]->(:RoleGroup)-[:INTERPRETS]->(interprets:ObjectConcept)
        RETURN DISTINCT interprets.FSN
        LIMIT 3
    }} AS interprets
"""

# ==================================================================
# LLM PROMPT FOR RERANKING GRAPHRAG CANDIDATES
# ==================================================================

def build_prompt(
    diagnosis: str,
    candidates: list[dict],
    max_llm_results: int = 5,
    use_score: bool = True,
) -> str:
    max_results = min(max_llm_results, len(candidates))

    score_fields = (
        """
            "base_score": "original base retriever score or null",
            "enriched_score": "original enriched retriever score or null",
            "rrf_score": "reciprocal rank fusion score"
        """
        if use_score
        else """
            "rrf_score": "reciprocal rank fusion score"
        """
    )

    return f"""
You are reranking SNOMED CT concepts for concept normalisation.

Diagnosis string:
{diagnosis}

Retrieved candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Each object in the retrieved candidates list is one candidate.
Concepts inside parents, grandparents, children, finding_sites,
morphologies, causative_agents, due_to, clinical_course, and interprets
are context only and are NOT candidates.

The candidates have already been combined using Reciprocal Rank Fusion (RRF).
The RRF score reflects how highly a candidate was ranked across the retrieval
methods.

The original base_score and enriched_score come from different retrieval
methods and are not necessarily directly comparable. Treat them as supporting
evidence only. A candidate with a lower raw retrieval score may still be the
best match if its meaning and graph context better match the diagnosis.

Select and rank the TOP {max_results} candidates from most likely to least
likely to represent the diagnosis string.

Rules:
- Return at most {max_results} candidates.
- Only return candidates from the retrieved candidates list.
- Do not return contextual concepts unless they also appear as a candidate.
- Do not simply preserve the RRF ranking.
- Consider the diagnosis string, candidate description, hierarchy,
  relationships, retrieval ranks, RRF score, and original retrieval scores.
- Prefer semantic and clinical correctness over retrieval score alone.
- Do not assume clinical information that is not present in the diagnosis string.
- Copy the exact sctid, fsn, and retrieval values from the candidate.
- Do not modify or recalculate any retrieval scores.

Return ONLY valid JSON with exactly this structure:

{{
    "results": [
        {{
            "sctid": "exact SCTID",
            "fsn": "exact FSN",
            "reason": "brief explanation of the ranking",
            {score_fields}
        }}
    ]
}}

Do not include markdown.
Do not include ```json.
Do not include any text before or after the JSON object.
""".strip()