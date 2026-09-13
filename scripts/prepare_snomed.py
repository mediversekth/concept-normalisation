import sys
from pathlib import Path

# Allow this script to be run directly from the repository root with:
# python scripts/prepare_snomed.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_normalisation.data_prep.snomed_extraction import SnomedExtractor
from concept_normalisation.semantic_matching.embedding.sapbert_embedder import SapBertEmbedder
from concept_normalisation.semantic_matching.embedding.biolord_embedder import BioLordEmbedder
from concept_normalisation import config

def main():
    extractor = SnomedExtractor()
    extractor.run()  # writes short terms + definitions parquet
    extractor.extract_isa_relationships()  # writes the is-a hierarchy parquet

    SapBertEmbedder().embed_parquet_chunked(config.SHORT_TERMS_PARQUET)

    biolord = BioLordEmbedder()
    biolord.embed_short_terms(config.SHORT_TERMS_PARQUET)
    biolord.embed_definitions(config.DEFINITIONS_PARQUET)


if __name__ == "__main__":
    main()
