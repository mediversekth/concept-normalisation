"""
Wraps the BioLORD SentenceTransformer model: embeds SNOMED CT short terms
and definitions (chunked + resumable, for building the candidate pool)
and query/context text (direct encode() calls, for matching against it).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from concept_normalisation.config import (
    BIOLORD_MODEL_NAME, DEFAULT_BIOLORD_BATCH_SIZE,
    BIOLORD_CHUNK_DIR, BIOLORD_SHORT_TERMS_CHUNK_DIR,
    DEFINITIONS_BIOLORD_EMB, SHORT_TERMS_BIOLORD_EMB,
)


class BioLordEmbedder:
    def __init__(self, model_name=BIOLORD_MODEL_NAME, batch_size=DEFAULT_BIOLORD_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts, batch_size: int = None, show_progress_bar: bool = True) -> np.ndarray:
        """Direct (non-chunked) encoding, as used in
        semantic_analysis_pipeline.py step 4 for the MIMIC context_texts."""
        batch_size = batch_size or self.batch_size
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

    def embed_in_chunks(self, texts, chunk_dir, output_path) -> np.ndarray:
        """Same resumable chunked-embedding workflow as embed_biolord's
        embed_in_chunks(): skip chunks already on disk, then stitch."""
        chunk_dir = Path(chunk_dir)
        chunk_dir.mkdir(exist_ok=True)

        n = len(texts)
        n_chunks = (n + self.batch_size - 1) // self.batch_size
        print(f"{n:,} texts -> {n_chunks:,} chunks of {self.batch_size}")

        for chunk_idx in range(n_chunks):
            chunk_path = chunk_dir / f"chunk_{chunk_idx:06d}.npy"
            if chunk_path.exists():
                continue  # already done -- resumable

            start = chunk_idx * self.batch_size
            end = min(start + self.batch_size, n)
            batch_embeddings = self.model.encode(
                texts[start:end], normalize_embeddings=True
            )
            np.save(chunk_path, batch_embeddings)

            if chunk_idx % 50 == 0:
                print(f"  chunk {chunk_idx:,}/{n_chunks:,}")

        print("Concatenating chunks...")
        all_chunks = [np.load(chunk_dir / f"chunk_{i:06d}.npy") for i in range(n_chunks)]
        embeddings = np.vstack(all_chunks)
        assert embeddings.shape[0] == n

        np.save(output_path, embeddings)
        print(f"Final embedding matrix shape: {embeddings.shape}")
        print(f"Saved to {output_path}")
        return embeddings

    def embed_definitions(self, definitions_parquet_path, chunk_dir=None, output_path=None):
        """--- 1. Definitions --- (embed_biolord, unchanged branch logic)"""
        definitions = pd.read_parquet(definitions_parquet_path)

        if len(definitions) == 0:
            print(
                "concepts_definitions.parquet is empty -- no text definitions were "
                "found in your graph (see step 10's output). Nothing to embed. "
                "If you want definitions, you'd need to either re-import SNOMED CT "
                "with the optional TextDefinition file included, or pull "
                "definitions from another source (e.g. UMLS) and align them by "
                "concept_id yourself."
            )
            return None

        print("\n=== Embedding SNOMED text definitions ===")
        def_texts = definitions["definition"].astype(str).tolist()
        embeddings = self.embed_in_chunks(
            def_texts,
            chunk_dir=chunk_dir or BIOLORD_CHUNK_DIR,
            output_path=output_path or DEFINITIONS_BIOLORD_EMB,
        )
        print("Row i of that file corresponds to row i of concepts_definitions.parquet")
        return embeddings

    def embed_short_terms(self, short_terms_parquet_path, chunk_dir=None, output_path=None):
        """--- 2. Short terms (FSN + synonyms) ---
        BioLORD is trained specifically to align concept NAMES with their
        DEFINITIONS in the same space (contrastive name<->definition pairs),
        so short terms are a natural fit for this model, not just a fallback
        for coverage. This gives BioLORD's own search something to match
        against for the ~97% of concepts that have no text definition at all."""
        short_terms = pd.read_parquet(short_terms_parquet_path)

        print("\n=== Embedding SNOMED short terms (FSN + synonyms) ===")
        short_term_texts = short_terms["description"].astype(str).tolist()
        embeddings = self.embed_in_chunks(
            short_term_texts,
            chunk_dir=chunk_dir or BIOLORD_SHORT_TERMS_CHUNK_DIR,
            output_path=output_path or SHORT_TERMS_BIOLORD_EMB,
        )
        print("Row i of that file corresponds to row i of concepts_short_terms.parquet")
        print("(same row alignment as short_terms_sapbert_embeddings.npy -- same rows, different model)")
        return embeddings
