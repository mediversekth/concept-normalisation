"""
Elasticsearch index and retrieval methods for SNOMED CT.

The index supports:

1. Standard multi-match BM25 retrieval.
2. Typo-tolerant Elasticsearch full-text retrieval using fuzziness.
"""

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError, bulk

from concept_normalisation.config import ELASTIC_INDEX_NAME, ELASTIC_URL


class ElasticBM25Index:
    def __init__(
        self,
        url: str = ELASTIC_URL,
        index_name: str = ELASTIC_INDEX_NAME,
    ):
        self.index_name = index_name
        self.es = Elasticsearch(url)

    def create_index(self) -> bool:
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        mapping = {
            "mappings": {
                "properties": {
                    "conceptId": {"type": "keyword"},
                    "term": {"type": "text"},
                    "definition": {"type": "text"},
                    "typeId": {"type": "keyword"},
                }
            }
        }

        self.es.indices.create(
            index=self.index_name,
            mappings=mapping["mappings"],
        )

        return self.es.indices.exists(index=self.index_name)

    def generate_documents(self, df: pd.DataFrame):
        """
        Yield Elasticsearch bulk-indexing actions one at a time.
        """

        for _, row in df.iterrows():
            yield {
                "_index": self.index_name,
                "_source": {
                    "conceptId": str(row["conceptId"]),
                    "term": str(row["term"]),
                    "definition": (
                        str(row["definition"])
                        if pd.notna(row["definition"])
                        else ""
                    ),
                    "typeId": str(row["typeId"]),
                },
            }

    def bulk_load(
        self,
        df: pd.DataFrame,
        chunk_size: int = 5000,
    ) -> None:
        try:
            bulk(
                self.es,
                self.generate_documents(df),
                chunk_size=chunk_size,
            )
        except BulkIndexError as error:
            print(f"Failed documents: {len(error.errors)}")
            print(error.errors[0])
            raise

    def multi_match_search(
        self,
        query: str,
        k: int = 10,
    ) -> list[dict]:
        """
        Standard non-fuzzy BM25 search across term and definition.

        The term field receives a larger boost because matching a SNOMED
        description is more important than matching only its definition.
        """

        response = self.es.search(
            index=self.index_name,
            size=k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": [
                        "term^3",
                        "definition",
                    ],
                    "type": "best_fields",
                }
            },
        )

        return response["hits"]["hits"]

    def fuzzy_search(
        self,
        query: str,
        k: int = 10,
        fuzziness: str = "AUTO",
        prefix_length: int = 1,
        max_expansions: int = 50,
    ) -> list[dict]:
        """
        Elasticsearch fuzzy full-text search.

        This uses multi_match with fuzziness so that multi-word test names
        are analyzed into separate tokens before fuzzy matching.
        """

        response = self.es.search(
            index=self.index_name,
            size=k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": [
                        "term^3",
                        "definition",
                    ],
                    "type": "best_fields",
                    "fuzziness": fuzziness,
                    "prefix_length": prefix_length,
                    "max_expansions": max_expansions,
                }
            },
        )

        return response["hits"]["hits"]