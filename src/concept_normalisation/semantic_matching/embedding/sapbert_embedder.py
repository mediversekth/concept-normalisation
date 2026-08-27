"""
Wraps the SapBERT model (raw transformers, mean-pooled): embeds SNOMED CT
short terms (chunked + resumable, for building the candidate pool) and
query text (one string at a time, for matching against it).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from concept_normalisation.config import (
    SAPBERT_MODEL_NAME, SAPBERT_CHUNK_DIR, SHORT_TERMS_SAPBERT_EMB,
)
from src.concept_normalisation.utils import get_device


class SapBertEmbedder:
    def __init__(
        self,
        model_name: str = SAPBERT_MODEL_NAME,
        max_length: int = 64,
        query_max_length: int = 128,
        batch_size: int = 64,
        device: torch.device = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.query_max_length = query_max_length
        self.batch_size = batch_size

        self.device = device or get_device()

        print("Using device:", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def embed_batch(self, texts, max_length: int = None) -> np.ndarray:
        max_length = max_length or self.max_length
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        emb = self.mean_pooling(output, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def embed_single(self, text) -> np.ndarray:
        """Same as semantic_analysis_pipeline.py's get_sapbert_embedding --
        used with e.g. semantic_data["baseline"].apply(embedder.embed_single).
        Uses query_max_length (128), not the short-term max_length (64)."""
        embedding = self.embed_batch([text], max_length=self.query_max_length)
        return embedding.squeeze()

    def embed_parquet_chunked(
        self,
        input_parquet_path,
        text_column: str = "description",
        output_path: Path = None,
        chunk_dir: Path = None,
    ) -> np.ndarray:
        """Same resumable chunked-embedding workflow as embed_sapbert.py's
        main script: embed every row of a parquet file's text_column in
        batch_size chunks, skipping chunks that already exist on disk, then
        stitch everything back together in order and save the full matrix."""
        output_path = Path(output_path or SHORT_TERMS_SAPBERT_EMB)
        chunk_dir = Path(chunk_dir or SAPBERT_CHUNK_DIR)
        chunk_dir.mkdir(exist_ok=True)

        short_terms = pd.read_parquet(input_parquet_path)
        texts = short_terms[text_column].astype(str).tolist()
        n = len(texts)
        n_chunks = (n + self.batch_size - 1) // self.batch_size
        print(f"{n:,} texts -> {n_chunks:,} chunks of {self.batch_size}")

        for chunk_idx in range(n_chunks):
            chunk_path = chunk_dir / f"chunk_{chunk_idx:06d}.npy"
            if chunk_path.exists():
                continue  # already done in a previous run -- skip

            start = chunk_idx * self.batch_size
            end = min(start + self.batch_size, n)
            batch_embeddings = self.embed_batch(texts[start:end])
            np.save(chunk_path, batch_embeddings)

            if chunk_idx % 50 == 0:
                print(f"  chunk {chunk_idx:,}/{n_chunks:,}")

        # --- Stitch all chunks back together in order ---
        print("Concatenating chunks...")
        all_chunks = [np.load(chunk_dir / f"chunk_{i:06d}.npy") for i in range(n_chunks)]
        embeddings = np.vstack(all_chunks)
        assert embeddings.shape[0] == n, "Row count mismatch -- a chunk is missing or corrupt"

        np.save(output_path, embeddings)
        print("Final embedding matrix shape:", embeddings.shape)
        print(f"Saved to {output_path}")
        print(f"Row i of that file corresponds to row i of {input_parquet_path}")

        # Once you've confirmed the final file is good, the per-chunk files in
        # chunk_dir can be deleted to save disk space.
        return embeddings
