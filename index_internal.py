"""
Index only internal OCR results — targets legal-docs-internal index
"""
import os
import sys
import hashlib
import logging
sys.path.insert(0, r'e:\Jeff')

from index_to_search import (
    load_config, create_search_index_if_not_exists,
    generate_embeddings, chunk_text, index_documents_batch
)
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from tqdm import tqdm
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()


def extract_metadata_from_path(blob_path: str, input_prefix: str) -> dict:
    """Extract metadata from blob path — strips .txt extension"""
    parts = blob_path.replace("extracted-text/internal/", "").split("/")
    # FIX: consistent with index_to_search.py — strip .txt
    file_name = parts[-1].replace(".txt", "") if parts else ""
    folder_path = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return {
        "file_name": file_name,
        "folder_path": folder_path,
        "blob_path": blob_path
    }


def main_internal():
    cfg = load_config()

    p = urlparse(cfg.container_sas_url)
    account_url = f"{p.scheme}://{p.netloc}"
    sas = p.query

    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container = p.path.strip('/').split('/')[-1]
    container_client = blob_service.get_container_client(container)

    openai_client = AzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_key,
        api_version="2024-02-01",
    )

    # FIX: targets internal index — reads SEARCH_INDEX_INTERNAL env var
    internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")

    create_search_index_if_not_exists(
        cfg.search_endpoint,
        cfg.search_key,
        internal_index,
        vector_dimensions=1536
    )

    # FIX: SearchClient created once and reused
    search_client = SearchClient(
        endpoint=cfg.search_endpoint,
        index_name=internal_index,
        credential=AzureKeyCredential(cfg.search_key)
    )

    # List only internal .txt files
    txt_files = []
    prefix = "extracted-text/internal/"

    logging.info(f"Listing blobs with prefix: {prefix}")
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if blob.name.lower().endswith(".txt"):
            txt_files.append(blob.name)

    if not txt_files:
        logging.warning("No .txt files found in extracted-text/internal/ folder")
        return

    logging.info(f"Found {len(txt_files)} internal text files to index")

    # Uses same batch_size=512 from index_to_search for consistency
    batch_size = 512
    documents_buffer = []

    for blob_path in tqdm(txt_files, desc="Indexing internal documents"):
        try:
            blob_client = container_client.get_blob_client(blob_path)
            content = blob_client.download_blob().readall().decode("utf-8")

            if not content.strip():
                logging.warning(f"Empty content in {blob_path}, skipping")
                continue

            chunks = chunk_text(content, cfg.max_chunk_size, cfg.chunk_overlap)
            total_chunks = len(chunks)

            meta = extract_metadata_from_path(blob_path, cfg.input_prefix)

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                embeddings = generate_embeddings(
                    openai_client,
                    batch_chunks,
                    cfg.openai_embedding_deployment
                )

                for idx, (chunk_content, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = i + idx

                    # FIX: was using hash() which is random per Python restart
                    # → created duplicate documents on every re-run (silent index corruption)
                    # Now uses deterministic MD5 — same file always gets same ID
                    doc_id = hashlib.md5(
                        f"internal:{blob_path}:{chunk_index}".encode()
                    ).hexdigest()

                    doc = {
                        "id": doc_id,
                        "content": chunk_content,
                        "content_vector": embedding,
                        "blob_path": blob_path,
                        "file_name": meta["file_name"],
                        "folder_path": meta["folder_path"],
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        # FIX: was "internal" — now matches what rag_backend.py checks
                        "source_container": "legal-documents-internal",
                    }
                    documents_buffer.append(doc)

                    if len(documents_buffer) >= 1000:
                        index_documents_batch(search_client, documents_buffer)
                        documents_buffer = []

        except Exception as e:
            logging.error(f"Error processing {blob_path}: {e}")
            continue

    if documents_buffer:
        index_documents_batch(search_client, documents_buffer)

    logging.info("Internal indexing complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main_internal()
