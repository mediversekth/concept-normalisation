
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
                "temperature": 0.0,
            },
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
You are performing SNOMED CT concept normalisation.

Diagnosis string:
{query_text}

Candidate SNOMED concepts retrieved from the terminology:
{context}

Select the candidate that most closely represents the diagnosis string.

Use:
- the candidate name
- synonyms / definition
- the SNOMED hierarchy
- retrieval relevance

Do not invent a SNOMED concept that is not present in the candidates.

Return the candidates ranked from most likely to least likely.

Use:
- the candidate name
- synonyms / definition
- the SNOMED hierarchy
- retrieval relevance

Do not invent a SNOMED concept that is not present in the candidates.

Return ONLY valid JSON.

The response must have exactly this structure:

{{
    "results": [
        {{
            "sctid": "SNOMED concept ID",
            "fsn": "Fully specified name",
            "reason": "Brief explanation",
            "score": score,
        }}
    ]
}}

Do not include markdown.
Do not include ```json.
Do not include any text before or after the JSON object.

{examples}
""",
            expected_inputs=["context", "query_text", "examples"],
        )

    def clean_query(self, text: str) -> str:
        """Do simple cleaning of query text"""
        text = str(text)
        # Remove '/', otherwise it won't be able to run the query
        text = text.replace("/", "or")
        text = re.sub(r"\s+", " ", text)
        return text


    def search(self, diagnosis: str, top_k: int = 3):
        """
        Search for SNOMED CT concepts that match the given diagnosis string.
        
        Parse the LLM output as JSON and return the results along with the retrieval context.

        Returns:
        {
            "answer": List[Dict[str, Any]],  # Parsed JSON from the LLM, key-value pairs are "sctid", "fsn", and "reason".
            "context": Any,                  # Retrieval context from the retriever
        }
        """
        result = self.rag.search(
            query_text=diagnosis,
            retriever_config={
                "top_k": top_k,
            },
            return_context=True,
        )

        try:
            answer = json.loads(result.answer)
        except json.JSONDecodeError:
            answer = {
                "error": "LLM returned invalid JSON",
                "raw_answer": result.answer,
            }

        return {
            "answer": answer,
            "matches": answer.get("results", []),
            "context": result.retriever_result,
        }

    def close(self):
        self.driver.close()