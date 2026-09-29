
from pathlib import Path
from datetime import datetime
from typing import Literal

from concept_normalisation import config
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
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

        self.base_retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            retrieval_query=RETRIEVAL_QUERY,
            embedder=self.embedder,
            result_formatter=self._result_formatter,
            neo4j_database=neo4j_database,
        )

        self.enriched_retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name=f"{vector_index_name}_enriched",
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
        base_candidates, _ = self._retrieve_candidates(
            retriever=self.base_retriever, diagnosis=cleaned_diagnosis, top_k=top_k
        )
        enriched_candidates, _ = self._retrieve_candidates(
            retriever=self.enriched_retriever, diagnosis=cleaned_diagnosis, top_k=top_k
        )

        # Fuse candidates by SCTID
        fused_candidates: dict[str, dict] = {}
        fused_candidates = self._fuse_candidates(fused_candidates, base_candidates, "base")
        fused_candidates = self._fuse_candidates(fused_candidates, enriched_candidates, "enriched")
        candidates = list(fused_candidates.values())

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
            matches = []
            self._log_error(
                msg=f"Unable to parse json for query '{cleaned_diagnosis}'", 
                diagnosis=cleaned_diagnosis, error=str(exc), answer=response.content
            )

        except (TypeError, ValueError) as exc:
            matches = []
            self._log_error(
                msg=f"Returned invalid results for query '{cleaned_diagnosis}'", 
                diagnosis=cleaned_diagnosis, error=str(exc), answer=response.content
            )

        return {
            "answer": answer,
            "matches": matches,
            "retrieved_candidates": candidates,
            "llm_candidates": llm_candidates,
            "top_k": top_k,
            "min_score": min_score,
        }

    def _fuse_candidates(
        self,
        fused: dict[str, dict],
        candidates: list[dict],
        method: Literal["base", "enriched"],
    ) -> dict[str, dict]:
        """Fuse one list of candidates into fused results by SCTID."""

        if method not in ("base", "enriched"):
            raise ValueError(f"Invalid method: {method}")

        score_key = f"{method}_score"

        for candidate in candidates:
            sctid = candidate.get("sctid")
            if sctid is None:
                continue

            sctid = str(sctid)
            if sctid in fused:
                fused[sctid][score_key] = candidate.get("score")

                if method not in fused[sctid]["found_by"]:
                    fused[sctid]["found_by"].append(method)
            else:
                fused[sctid] = {
                    **candidate,
                    "base_score": candidate.get("score") if method == "base" else None,
                    "enriched_score": candidate.get("score") if method == "enriched" else None,
                    "found_by": [method],
                }

            scores = [
                fused[sctid].get("base_score"),
                fused[sctid].get("enriched_score"),
            ]

            fused[sctid]["score"] = max(
                score for score in scores if score is not None
            )

        return fused

    def _retrieve_candidates(
        self,
        retriever: HybridCypherRetriever,
        diagnosis: str,
        top_k: int,
    ) -> tuple[list[dict], object]:
        """Retrieve candidates from one retriever and convert them to dictionaries."""
        result = retriever.search(
            query_text=diagnosis,
            top_k=top_k,
        )

        candidates = [
            self._candidate_from_item(item)
            for item in result.items
        ]

        return candidates, result

    def close(self):
        self.driver.close()

    # ===================================================================
    # LOG HELPERS
    # ===================================================================

    def _log_error(self, msg, diagnosis, error, answer):
        print(f"ERROR [GRAPHRAG]: {msg}")

        logs_dir = Path(self.log_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        with (logs_dir / "graphrag_errors.jsonl").open("a",encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "diagnosis": diagnosis,
                    "error": error,
                    "raw_answer": answer,
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
                    "nr_of_candidates": len(candidates),
                    "diagnosis": diagnosis,
                    "candidates": candidates,
                    "llm_candidates": llm_candidates,
                    "matches": matches,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
        