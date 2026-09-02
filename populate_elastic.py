from concept_normalisation.data_prep.snomed_extraction import SnomedExtractor
from concept_normalisation.syntactic_matching.elastic_bm25 import ElasticBM25Index

def main():
    print("Preparing SNOMED documents...")
    extractor = SnomedExtractor()
    documents = extractor.build_bm25_documents()

    print(f"Documents to index: {len(documents):,}")

    # Connect to ElasticSearch
    elastic = ElasticBM25Index()

    print("Creating index...")
    elastic.create_index()
    print("Created index!")

    print("Uploading documents...")
    elastic.bulk_load(documents)
    print("Uploading done!")

    print("Done!")


if __name__ == "__main__":
    main()