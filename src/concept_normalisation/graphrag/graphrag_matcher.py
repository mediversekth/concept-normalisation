
from pathlib import Path
from datetime import datetime

from concept_normalisation import config
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.types import RetrieverResultItem
import json
import re
from concept_normalisation.graphrag.graphrag_queries import (
    RETRIEVAL_QUERY,
    build_prompt
)

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
        log_dir=config.DATA_DIR,
        table_name=""
    ):
        self.log_dir=log_dir
        self.table_name=table_name

        self.driver = GraphDatabase.driver(
            neo4j_uri, 
            auth=(neo4j_user, neo4j_password),
        )

        self.embedder = SentenceTransformerEmbeddings(
            model=embedding_model,
        )

        self.llm = OllamaLLM(
            model_name=llm_model,
            host=config.GRAPHRAG_DEFAULT_HOST,
            model_params={
                "options": {
                    "temperature": 0.0,
                    "num_ctx": config.GRAPHRAG_DEFAULT_CONTEXT_SIZE,
                    "seed": config.GRAPHRAG_DEFAULT_SEED,
                },
                "format": "json",
                "keep_alive": "30m",
            },
            timeout=600
        )

        self.retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=RETRIEVAL_QUERY,
            embedder=self.embedder,
            result_formatter=self._result_formatter,
            neo4j_database=neo4j_database,
        )


    @staticmethod
    def _result_formatter(record) -> RetrieverResultItem:
        """Keep a structured, serialisable copy of every retriever candidate."""
        candidate = record.data()

        if candidate.get("score") is not None:
            candidate["score"] = float(candidate["score"])

        return RetrieverResultItem(
            content=json.dumps(candidate, ensure_ascii=False, default=str),
            metadata={
                "sctid": candidate.get("sctid"),
                "fsn": candidate.get("fsn"),
                "score": candidate.get("score"),
            },
        )


    @staticmethod
    def _candidate_from_item(item: RetrieverResultItem) -> dict:
        """Convert a RetrieverResultItem back to a plain dictionary."""
        if isinstance(item.content, dict):
            return dict(item.content)

        try:
            return json.loads(item.content)
        except (TypeError, json.JSONDecodeError):
            return {
                "content": item.content,
                "metadata": item.metadata,
            }


    def _clean_query(self, text: str) -> str:
        """Do simple cleaning of query text"""
        text = str(text)
        # Remove '/', otherwise it won't be able to run the query
        text = text.replace("/", "or")
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    def search(
            self, 
            diagnosis: str, 
            top_k: int = 10,
            min_score: float | None = None
        ):
        """
        Retrieve candidates, optionally filter weak retrievals, then ask the LLM
        to rerank at most the top five.

        The complete retriever output is preserved before filtering or LLM handling.
        """

        cleaned_diagnosis = self._clean_query(diagnosis)

        # Step 1: Retrieve candidates, will also retrieve context
        retriever_results = self.retriever.search(
            query_text=cleaned_diagnosis,
            top_k=top_k
        )
        candidates = [
            self._candidate_from_item(item)
            for item in retriever_results.items
        ]

        # Step 2: If a min_score is given, filter out results below min_score
        if min_score is None:
            llm_candidates = list(candidates)
        else:
            llm_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("score") is None
                or float(candidate["score"]) >= min_score
            ]

        # Nothing survived retrieval/filtering, so there is nothing to rerank.
        if not llm_candidates:
            return {
                "answer": {"results": []},
                "matches": [],
                "retrieved_candidates": candidates,
                "llm_candidates": [],
                "context": retriever_results,
                "top_k": top_k,
                "min_score": min_score,
            }

        # Step 3: LLM reranking. It sees all surviving candidates but should return
        # no more than configured maximum number of candidates.
        prompt = build_prompt(cleaned_diagnosis, llm_candidates, config.GRAPHRAG_DEFAULT_MAX_LLM_RESULTS)
        response = self.llm.invoke(prompt)

        try:
            # Load response as json, retain only MAX RESULTS, 
            answer = json.loads(response.content)
            matches = answer.get("results", [])

            if not isinstance(matches, list):
                raise ValueError("LLM did not return 'results' field as a list")

            if len(matches) > config.GRAPHRAG_DEFAULT_MAX_LLM_RESULTS:
                print(
                    f"WARNING [GRAPHRAG]: LLM returned {len(matches)} for "
                    f"query '{cleaned_diagnosis}'. Keeping only first {config.GRAPHRAG_DEFAULT_MAX_LLM_RESULTS}"
                )
                matches = matches[:config.GRAPHRAG_DEFAULT_MAX_LLM_RESULTS]

            for match in matches:
                if "score" in match and match["score"] is not None:
                    match["score"] = float(match["score"])

            self._log_candidates(cleaned_diagnosis, candidates=candidates, llm_candidates=llm_candidates, matches=matches)

        except json.JSONDecodeError as exc:
            print(f"ERROR [GRAPHRAG]: Unable to parse json for query '{cleaned_diagnosis}'")

            answer = {
                "error": exc.msg,
                "raw_answer": response.content
            }
            matches = []
            self._log_error(diagnosis=cleaned_diagnosis, answer=answer)

        except (TypeError, ValueError) as exc:
            print(f"ERROR [GRAPHRAG]: Returned invalid results for query '{cleaned_diagnosis}'")

            answer = {
                "error": str(exc),
                "raw_answer": response.content,
            }
            matches = []
            self._log_error(diagnosis=cleaned_diagnosis, answer=answer)

        return {
            "answer": answer,
            "matches": matches,
            "retrieved_candidates": candidates,
            "llm_candidates": llm_candidates,
            "context": retriever_results,
            "top_k": top_k,
            "min_score": min_score,
        }

    def close(self):
        self.driver.close()

    def _log_error(self, diagnosis, answer):
        logs_dir = Path(self.log_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        with (logs_dir / "graphrag_errors.jsonl").open("a",encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "diagnosis": diagnosis,
                    "error": answer.get("error"),
                    "raw_answer": answer.get("raw_answer"),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    def _log_candidates(self, diagnosis, candidates, llm_candidates, matches):
        logs_dir = Path(self.log_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        with (logs_dir / f"{self.table_name}_graphrag_candidates.jsonl").open("a",encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "diagnosis": diagnosis,
                    "candidates": candidates,
                    "llm_candidates": llm_candidates,
                    "matches": matches,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
        