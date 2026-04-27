"""
Incremental indexer for external OCR outputs.

Uses the external OCR success log to index only newly OCR'd files into the
already-deployed Azure Search external index, while matching the live schema.
"""
import logging
import hashlib
import os
from urllib.parse import urlparse

import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

from index_to_search import (
    chunk_text,
    generate_embeddings,
    index_documents_batch,
    load_config,
    extract_source_path_from_content,
    strip_metadata_header,
)


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_container_sas_url(container_sas_url: str, container_override: str = "") -> tuple[str, str, str]:
    parsed = urlparse(container_sas_url)
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    container_name = container_override or (path_parts[-1] if path_parts else "")
    if not container_name:
        raise ValueError(
            "Container name not found in CONTAINER_SAS_URL. "
            "Set CONTAINER_NAME=legal-documents or use a container-level SAS URL."
        )
    account_url = f"{parsed.scheme}://{parsed.netloc}"
    return account_url, container_name, parsed.query


def fetch_live_field_names(endpoint: str, key: str, index_name: str) -> set[str]:
    url = endpoint.rstrip("/") + f"/indexes/{index_name}?api-version=2024-07-01"
    response = requests.get(url, headers={"api-key": key}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return {field["name"] for field in payload.get("fields", [])}


def generate_doc_id(blob_path: str, chunk_index: int) -> str:
    return hashlib.md5(f"{blob_path}:{chunk_index}".encode()).hexdigest()


def ocr_blob_path_from_success_entry(entry: str) -> str:
    entry = entry.strip()
    if not entry:
        return ""
    if entry.startswith("extracted-text/"):
        return entry
    return f"extracted-text/{entry}.txt"


def extract_metadata_from_blob_or_source(blob_path: str, content: str) -> dict:
    source_path = extract_source_path_from_content(content)
    if source_path:
        parts = source_path.split("/")
        return {
            "file_name": parts[-1],
            "folder_path": "/".join(parts[:-1]),
            "blob_path": blob_path,
            "content": strip_metadata_header(content),
        }

    relative_path = blob_path[len("extracted-text/"):] if blob_path.startswith("extracted-text/") else blob_path
    parts = relative_path.split("/")
    return {
        "file_name": parts[-1].replace(".txt", "") if parts else relative_path.replace(".txt", ""),
        "folder_path": "/".join(parts[:-1]) if len(parts) > 1 else "",
        "blob_path": blob_path,
        "content": content,
    }


def main():
    cfg = load_config()
    index_name = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    success_list_file = os.environ.get("SUCCESS_LIST_FILE", "logs/ocr_success_list.txt")

    if not os.path.exists(success_list_file):
        raise FileNotFoundError(f"Success list file not found: {success_list_file}")

    with open(success_list_file, encoding="utf-8", errors="ignore") as fh:
        success_entries = [line.strip() for line in fh if line.strip()]

    target_blob_paths = [ocr_blob_path_from_success_entry(entry) for entry in success_entries]
    logger.info(f"Loaded {len(target_blob_paths)} OCR-success entries from {success_list_file}")

    live_fields = fetch_live_field_names(cfg.search_endpoint, cfg.search_key, index_name)
    logger.info(f"Live index '{index_name}' has {len(live_fields)} fields")

    container_override = os.environ.get("CONTAINER_NAME", "legal-documents")
    account_url, container_name, sas = split_container_sas_url(cfg.container_sas_url, container_override)
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container_name)

    openai_client = AzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_key,
        api_version="2024-02-01",
    )

    search_client = SearchClient(
        endpoint=cfg.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(cfg.search_key),
    )

    documents_buffer = []
    indexed_files = 0
    missing_blobs = 0

    for blob_path in tqdm(target_blob_paths, desc="Indexing new external OCR files"):
        try:
            blob_client = container_client.get_blob_client(blob_path)
            try:
                content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
            except Exception:
                logger.warning(f"OCR blob missing or unreadable, skipping: {blob_path}")
                missing_blobs += 1
                continue

            if not content.strip():
                logger.warning(f"Empty OCR content, skipping: {blob_path}")
                continue

            meta = extract_metadata_from_blob_or_source(blob_path, content)
            chunks = chunk_text(meta["content"], cfg.max_chunk_size, cfg.chunk_overlap)
            total_chunks = len(chunks)
            if not chunks:
                logger.warning(f"No chunks generated, skipping: {blob_path}")
                continue

            for start in range(0, len(chunks), 512):
                batch_chunks = chunks[start:start + 512]
                embeddings = generate_embeddings(
                    openai_client,
                    batch_chunks,
                    cfg.openai_embedding_deployment,
                )

                for idx, (chunk_content, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = start + idx
                    doc = {
                        "id": generate_doc_id(blob_path, chunk_index),
                        "content": chunk_content,
                        "blob_path": meta["blob_path"],
                        "file_name": meta["file_name"],
                        "folder_path": meta["folder_path"],
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "source_container": "legal-documents",
                        "content_vector": embedding,
                        "document_type": "",
                        "document_subtype": "",
                        "persons": [],
                        "organizations": [],
                        "projects": [],
                        "key_dates": [],
                        "key_amounts": [],
                        "summary": "",
                    }
                    documents_buffer.append({k: v for k, v in doc.items() if k in live_fields})

                    if len(documents_buffer) >= 1000:
                        index_documents_batch(search_client, documents_buffer)
                        documents_buffer = []

            indexed_files += 1

        except Exception as e:
            logger.error(f"Failed to index {blob_path}: {e}")

    if documents_buffer:
        index_documents_batch(search_client, documents_buffer)

    logger.info(
        f"Incremental external indexing complete | indexed_files={indexed_files} | "
        f"missing_ocr_blobs={missing_blobs}"
    )


if __name__ == "__main__":
    main()
