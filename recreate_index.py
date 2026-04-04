"""
Recreate search indices — creates both legal-docs-external and legal-docs-internal
"""
import os
import logging
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

print(f"Loading from: {env_path}")
print(f"Endpoint: {os.environ.get('SEARCH_ENDPOINT')}")

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField,
    SearchFieldDataType, VectorSearch, HnswAlgorithmConfiguration,
    VectorSearchProfile, SemanticConfiguration, SemanticPrioritizedFields,
    SemanticField, SemanticSearch,
)


def recreate_search_index(search_endpoint: str, search_key: str, index_name: str, vector_dimensions: int = 1536):
    """Delete and recreate a search index with full hybrid search capability"""
    credential = AzureKeyCredential(search_key)
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    # Delete existing index if present
    try:
        index_client.delete_index(index_name)
        logging.info(f"Deleted existing index: {index_name}")
    except Exception as e:
        logging.info(f"Index may not exist yet: {e}")

    # Field definitions
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        # FIX: SearchableField (not SimpleField) so partial name search works via BM25
        SearchableField(name="blob_path", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="file_name", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="folder_path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="total_chunks", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="source_container", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name="default-vector-config",
        ),
    ]

    # HNSW vector search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="default-hnsw", kind="hnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="default-vector-config",
                algorithm_configuration_name="default-hnsw"
            )
        ]
    )

    # Semantic search configuration
    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        )
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    result = index_client.create_index(index)
    logging.info(f"Created index: {result.name} with {vector_dimensions}d vectors + semantic config")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    search_endpoint = os.environ.get("SEARCH_ENDPOINT")
    search_key = os.environ.get("SEARCH_KEY")

    ext_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    int_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")

    print(f"\nCreating external index: {ext_index}")
    recreate_search_index(search_endpoint, search_key, ext_index)

    print(f"\nCreating internal index: {int_index}")
    recreate_search_index(search_endpoint, search_key, int_index)

    print("\nBoth indices created successfully!")
    print("Next steps:")
    print("  1. python index_internal.py   (fast — run first)")
    print("  2. python index_to_search.py  (long — run after internal is done)")
