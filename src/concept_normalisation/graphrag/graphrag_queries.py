""

import json

# ==================================================================
# NEO4J CYPHER FOR RETRIEVING CONTEXT
# ==================================================================

RETRIEVAL_QUERY = """
OPTIONAL MATCH (node)-[:ISA]->(parent:ObjectConcept)
OPTIONAL MATCH (parent)-[:ISA]->(grandparent:ObjectConcept)

OPTIONAL MATCH (child:ObjectConcept)-[:ISA]->(node)

OPTIONAL MATCH (node)-[:HAS_ROLE_GROUP]->(rg:RoleGroup)

OPTIONAL MATCH (rg)-[:FINDING_SITE]->(site:ObjectConcept)
OPTIONAL MATCH (rg)-[:ASSOCIATED_MORPHOLOGY]->(morph:ObjectConcept)
OPTIONAL MATCH (rg)-[:CAUSATIVE_AGENT]->(agent:ObjectConcept)
OPTIONAL MATCH (rg)-[:DUE_TO]->(cause:ObjectConcept)
OPTIONAL MATCH (rg)-[:CLINICAL_COURSE]->(course:ObjectConcept)
OPTIONAL MATCH (rg)-[:INTERPRETS]->(interprets:ObjectConcept)

RETURN
    node.sctid AS sctid,
    node.FSN AS fsn,
    node.embedding_text AS description,
    score,

    collect(DISTINCT parent.FSN) AS parents,
    collect(DISTINCT grandparent.FSN) AS grandparents,

    collect(DISTINCT child.FSN) AS children,

    collect(DISTINCT site.FSN) AS finding_sites,
    collect(DISTINCT morph.FSN) AS morphologies,
    collect(DISTINCT agent.FSN) AS causative_agents,
    collect(DISTINCT cause.FSN) AS due_to,
    collect(DISTINCT course.FSN) AS clinical_course,
    collect(DISTINCT interprets.FSN) AS interprets
"""

# ==================================================================
# LLM PROMPT FOR RERANKING GRAPHRAG CANDIDATES
# ==================================================================

def build_prompt(
        diagnosis: str, 
        candidates: list[dict], 
        max_llm_results: int = 5
    ) -> str:
    max_results = min(max_llm_results, len(candidates))

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

Select and rank the TOP {max_results} candidates from most likely to least
likely to represent the diagnosis string.

Rules:
- Return at most {max_results} candidates.
- Only return candidates from the retrieved candidates list.
- Do not return contextual concepts unless they also appear as a candidate.
- Do not simply preserve the retrieval order.
- Base the ranking on the candidate description, hierarchy, relationships, retrieval score, and diagnosis string.
- Do not assume clinical information that is not present in the diagnosis string.
- Copy the exact sctid, fsn, and original retrieval score for every returned candidate.

Return ONLY valid JSON with exactly this structure:

{{
    "results": [
        {{
            "sctid": "exact SCTID",
            "fsn": "exact FSN",
            "reason": "brief explanation of the ranking",
            "score": "original score of the retrieved concept"
        }}
    ]
}}

Do not include markdown.
Do not include ```json.
Do not include any text before or after the JSON object.
""".strip()