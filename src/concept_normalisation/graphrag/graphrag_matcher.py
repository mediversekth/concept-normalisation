
from concept_normalisation import config
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.generation import GraphRAG, RagTemplate
import json
import re

class GraphRAGMatcher:
    """A matcher for performing graph-based retrieval and reranking of SNOMED CT concepts."""

    def __init__(
        self,
        neo4j_uri=config.NEO4J_URI,
        neo4j_user=config.NEO4J_USER,
        neo4j_password=config.NEO4J_PASSWORD,
        neo4j_database=config.NEO4J_DATABASE,
        vector_index_name=config.GRAPHRAG_VECTOR_INDEX_NAME,
        fulltext_index_name=config.GRAPHRAG_FULLTEXT_INDEX_NAME,
        embedding_model=config.GRAPHRAG_EMBEDDING_MODEL_NAME,
        llm_model=config.GRAPHRAG_LLM_MODEL_NAME,
    ):
        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password),
        )

        self.embedder = SentenceTransformerEmbeddings(
            model=embedding_model,
        )

        self.llm = OllamaLLM(
            model_name=llm_model,
            model_params={
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 16000,
                },
                "format": "json",
            }
        )

        self.retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=self._retrieval_query(),
            embedder=self.embedder,
            neo4j_database=neo4j_database,
        )

        self.rag = GraphRAG(
            retriever=self.retriever,
            llm=self.llm,
            prompt_template=self._prompt_template(),
        )

    def _retrieval_query(self) -> str:
        return """
OPTIONAL MATCH (node)-[:ISA]->(parent:ObjectConcept)
OPTIONAL MATCH (parent)-[:ISA]->(grandparent:ObjectConcept)

OPTIONAL MATCH (child:ObjectConcept)-[:ISA]->(node)
OPTIONAL MATCH (grandchild:ObjectConcept)-[:ISA]->(child)

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
    collect(DISTINCT grandchild.FSN) as grandchildren,

    collect(DISTINCT site.FSN) AS finding_sites,
    collect(DISTINCT morph.FSN) AS morphologies,
    collect(DISTINCT agent.FSN) AS causative_agents,
    collect(DISTINCT cause.FSN) AS due_to,
    collect(DISTINCT course.FSN) AS clinical_course,
    collect(DISTINCT interprets.FSN) AS interprets
"""

    def _prompt_template(self):
        return RagTemplate(
            template="""
You are reranking SNOMED CT concepts for concept normalisation.

Diagnosis string:
{query_text}

Retrieved candidates:
{context}

Each top-level <Record ...> is one candidate.

Concepts inside parents, grandparents, children, grandchildren,
finding_sites, morphologies, causative_agents, due_to,
clinical_course, and interprets are context only and are NOT candidates.

Rank every top-level candidate from most likely to least likely to
represent the diagnosis string.

Rules:
- Return every top-level candidate exactly once.
- Do not add or remove candidates.
- Do not return contextual concepts unless they also appear as a top-level candidate.
- Do not simply preserve the retrieval order.
- Base the ranking on the candidate description, hierarchy, relationships, and diagnosis string.
- Do not assume clinical information that is not present in the diagnosis string.

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
    """,
            expected_inputs=["context", "query_text"],
        )

    def clean_query(self, text: str) -> str:
        """Do simple cleaning of query text"""
        text = str(text)
        # Remove '/', otherwise it won't be able to run the query
        text = text.replace("/", "or")
        text = re.sub(r"\s+", " ", text)
        return text


    def search(self, diagnosis: str, top_k: int = 3):
        result = self.rag.search(
            query_text=diagnosis,
            retriever_config={
                "top_k": top_k,
            },
            return_context=True,
        )

        try:
            # Parse LLM output to json, parse score as double.
            # Check that LLM outputs same amount of concepts as candidates provided by retriever.
            answer = json.loads(result.answer)
            matches = answer.get("results", [])

            retrieved_count = len(result.retriever_result.items)
            returned_count = len(matches)

            if returned_count != retrieved_count:
                raise ValueError(
                    f"LLM returned {returned_count} candidates, "
                    f"but retriever returned {retrieved_count}"
                )

            for match in matches:
                if "score" in match and match["score"] is not None:
                    match["score"] = float(match["score"])

        except json.JSONDecodeError as exc:
            # Something went wrong when parsing LLM output as JSON
            print(
                f"WARNING: GraphRAG was unable to parse returned JSON "
                f"for query {diagnosis!r}"
            )
            print(result.answer)

            answer = {
                "error": exc.msg,
                "raw_answer": result.answer,
            }
            matches = []

        except ValueError as exc:
            print(
                f"WARNING: GraphRAG returned invalid results "
                f"for query {diagnosis!r}: {exc}"
            )

            answer = {
                "error": str(exc),
                "raw_answer": result.answer,
            }
            matches = []

        return {
            "answer": answer,
            "matches": matches,
            "context": result.retriever_result,
        }

    def close(self):
        self.driver.close()