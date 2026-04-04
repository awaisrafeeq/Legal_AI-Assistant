"""
Sprint 2: Hybrid Search Index Pipeline
Reads OCR text from blob storage, chunks, generates embeddings, uploads to Azure AI Search
"""
import os
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tiktoken import encoding_for_model
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    container_sas_url: str
    search_endpoint: str
    search_key: str
    search_index: str
    openai_endpoint: str
    openai_key: str
    openai_embedding_deployment: str
    input_prefix: str  # Where OCR text files are (e.g., "extracted-text")
    max_files: Optional[int]
    max_chunk_size: int
    chunk_overlap: int


def load_config() -> Config:
    load_dotenv()
    return Config(
        container_sas_url=os.environ["CONTAINER_SAS_URL"],
        search_endpoint=os.environ["SEARCH_ENDPOINT"],
        search_key=os.environ["SEARCH_KEY"],
        # FIX: reads SEARCH_INDEX_EXTERNAL — external indexer targets external index only
        search_index=os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external"),
        openai_endpoint=os.environ["OPENAI_ENDPOINT"],
        openai_key=os.environ["OPENAI_KEY"],
        openai_embedding_deployment=os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT", "Embeddings"),
        input_prefix=os.environ.get("OUTPUT_PREFIX", "extracted-text"),
        max_files=int(os.environ["MAX_FILES"]) if os.environ.get("MAX_FILES") else None,
        max_chunk_size=int(os.environ.get("MAX_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "50")),
    )


def split_container_sas_url(container_sas_url: str) -> Tuple[str, str, str]:
    p = urlparse(container_sas_url)
    if not p.scheme or not p.netloc:
        raise ValueError("CONTAINER_SAS_URL must be a full https URL")
    if not p.query:
        raise ValueError("CONTAINER_SAS_URL must include SAS query string")
    container_name = p.path.strip("/").split("/")[-1]
    if not container_name:
        raise ValueError("CONTAINER_SAS_URL path must end with container name")
    account_url = f"{p.scheme}://{p.netloc}"
    return account_url, container_name, p.query


def get_encoding():
    """Get tiktoken encoding for text-embedding-ada-002 (cl100k_base)"""
    return encoding_for_model("text-embedding-ada-002")


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split text into overlapping chunks by token count"""
    enc = get_encoding()
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        # FIX: skip tiny trailing chunks — not semantically useful, waste API calls
        if len(chunk_tokens) < 10:
            break

        # FIX: renamed from chunk_text to decoded_chunk — was shadowing the function name
        decoded_chunk = enc.decode(chunk_tokens)
        chunks.append(decoded_chunk)

        # Move forward by chunk size minus overlap
        start += max_tokens - overlap_tokens
        if start >= len(tokens):
            break
        # Prevent infinite loop if overlap >= max_tokens
        if start <= 0:
            start = end

    return chunks


def generate_doc_id(blob_path: str, chunk_index: int) -> str:
    """Generate unique document ID from blob path and chunk index"""
    hash_input = f"{blob_path}:{chunk_index}"
    return hashlib.md5(hash_input.encode()).hexdigest()


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(6))
def generate_embeddings(client: AzureOpenAI, texts: List[str], deployment: str) -> List[List[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI"""
    response = client.embeddings.create(
        model=deployment,
        input=texts
    )
    return [item.embedding for item in response.data]


def create_search_index_if_not_exists(
    endpoint: str,
    key: str,
    index_name: str,
    vector_dimensions: int = 1536
) -> bool:
    """Create Azure AI Search index with hybrid search + semantic configuration"""
    credential = AzureKeyCredential(key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    try:
        index_client.get_index(index_name)
        logger.info(f"Index '{index_name}' already exists")
        return False
    except Exception:
        pass

    # FIX: SearchableField (not SimpleField) for blob_path, file_name, folder_path
    # SimpleField(searchable=True) does NOT do full-text search — only SearchableField does
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
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

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[VectorSearchProfile(name="default-vector-config", algorithm_configuration_name="hnsw-algo")],
    )

    # FIX: added SemanticConfiguration — enables semantic re-ranking on content field
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

    index_client.create_index(index)
    logger.info(f"Created index '{index_name}' with {vector_dimensions}d vector search + semantic config")
    return True


# FIX: search_client passed in — created once in main(), not re-instantiated every batch
def index_documents_batch(
    search_client: SearchClient,
    documents: List[Dict[str, Any]]
) -> None:
    """Upload documents to Azure AI Search in batch"""
    result = search_client.upload_documents(documents=documents)
    success_count = sum(1 for r in result if r.succeeded)
    logger.info(f"Indexed {success_count}/{len(documents)} documents")


def extract_metadata_from_path(blob_path: str, input_prefix: str) -> Dict[str, str]:
    """Extract file name and folder path from blob path"""
    relative_path = blob_path[len(input_prefix):].lstrip("/")

    parts = relative_path.split("/")
    # FIX: strip .txt extension — makes file_name consistent with index_internal.py
    file_name = parts[-1].replace(".txt", "") if parts else relative_path.replace(".txt", "")
    folder_path = "/".join(parts[:-1]) if len(parts) > 1 else ""

    return {
        "file_name": file_name,
        "folder_path": folder_path,
    }


def extract_source_path_from_content(content: str) -> Optional[str]:
    """
    process_all_formats.py writes a header at the top of every OCR file:
        Source: original/path/to/file.pdf
        Type: PDF
        ============================================================

    If this header is present, return the original source path so we can
    reconstruct the correct folder_path / file_name for the search index.
    Returns None for plain OCR files (ocr_to_blob.py style) that have no header.
    """
    first_line = content.split("\n", 1)[0].rstrip()
    if first_line.startswith("Source: "):
        return first_line[len("Source: "):].rstrip()
    return None


def strip_metadata_header(content: str) -> str:
    """Remove the Source:/Type:/=== header added by process_all_formats.py before indexing."""
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("=" * 10):
            return "\n".join(lines[i + 1:]).lstrip("\n")
    return content


def main() -> None:
    cfg = load_config()

    account_url, container, sas = split_container_sas_url(cfg.container_sas_url)
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    openai_client = AzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_key,
        api_version="2024-02-01",
    )

    create_search_index_if_not_exists(
        cfg.search_endpoint,
        cfg.search_key,
        cfg.search_index,
        vector_dimensions=1536
    )

    # FIX: SearchClient created ONCE and reused — was being re-instantiated every 1000 docs
    search_client = SearchClient(
        endpoint=cfg.search_endpoint,
        index_name=cfg.search_index,
        credential=AzureKeyCredential(cfg.search_key)
    )

    # List all .txt files in the extracted-text folder (non-internal only)
    txt_files = []
    prefix = cfg.input_prefix.strip("/") + "/" if cfg.input_prefix else ""

    logger.info(f"Listing blobs with prefix: {prefix or '(root)'}")
    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        # Skip internal folder — handled by index_internal.py
        if blob.name.lower().endswith(".txt") and "/internal/" not in blob.name:
            txt_files.append(blob.name)
            if cfg.max_files and len(txt_files) >= cfg.max_files:
                break

    if not txt_files:
        logger.warning(f"No .txt files found with prefix: {prefix}")
        return

    logger.info(f"Found {len(txt_files)} external text files to index")

    # FIX: batch_size 16 → 512
    # With ~422 chunks per large file: was 27 API calls/file, now 1 API call/file
    # Reduces ~588,000 unnecessary API calls across full dataset
    batch_size = 512
    documents_buffer = []

    for blob_path in tqdm(txt_files, desc="Indexing external documents"):
        try:
            blob_client = container_client.get_blob_client(blob_path)
            content = blob_client.download_blob().readall().decode("utf-8")

            if not content.strip():
                logger.warning(f"Empty content in {blob_path}, skipping")
                continue

            # If the OCR file has a "Source: ..." header (written by process_all_formats.py),
            # extract the original document path from it so folder_path / file_name are correct.
            # Files from ocr_to_blob.py have no such header and are handled by the path alone.
            meta = extract_metadata_from_path(blob_path, cfg.input_prefix)
            source_path = extract_source_path_from_content(content)
            if source_path:
                parts = source_path.split("/")
                meta = {
                    "file_name": parts[-1],
                    "folder_path": "/".join(parts[:-1]),
                }
                content = strip_metadata_header(content)

            chunks = chunk_text(content, cfg.max_chunk_size, cfg.chunk_overlap)
            total_chunks = len(chunks)

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                embeddings = generate_embeddings(
                    openai_client,
                    batch_chunks,
                    cfg.openai_embedding_deployment
                )

                for idx, (chunk_content, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = i + idx

                    doc = {
                        "id": generate_doc_id(blob_path, chunk_index),
                        "content": chunk_content,
                        "content_vector": embedding,
                        "blob_path": blob_path,
                        "file_name": meta["file_name"],
                        "folder_path": meta["folder_path"],
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        # FIX: always "legal-documents" — external indexer never tags anything as internal
                        "source_container": "legal-documents",
                    }
                    documents_buffer.append(doc)

                    if len(documents_buffer) >= 1000:
                        index_documents_batch(search_client, documents_buffer)
                        documents_buffer = []

        except Exception as e:
            logger.error(f"Error processing {blob_path}: {e}")
            continue

    if documents_buffer:
        index_documents_batch(search_client, documents_buffer)

    logger.info("External indexing complete!")


if __name__ == "__main__":
    main()
