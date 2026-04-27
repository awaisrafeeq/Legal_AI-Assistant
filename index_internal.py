"""
Index only internal OCR results — targets legal-docs-internal index
"""
import os
import sys
import hashlib
import logging
from datetime import datetime, timezone
sys.path.insert(0, r'e:\Jeff')

from index_to_search import (
    load_config, create_search_index_if_not_exists,
    generate_embeddings, chunk_text, index_documents_batch,
    extract_source_path_from_content, strip_metadata_header,
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


def _resolve_account_and_container(sas_url: str, container_override: str = ""):
    p = urlparse(sas_url)
    account_url = f"{p.scheme}://{p.netloc}"
    path_parts = [part for part in p.path.strip("/").split("/") if part]
    container = container_override or (path_parts[0] if path_parts else "")
    return account_url, p.query, container


def ocr_blob_path_from_source_entry(entry: str, output_prefix: str = "extracted-text") -> str:
    prefix = output_prefix.strip("/")
    if prefix:
        return f"{prefix}/internal/{entry}.txt"
    return f"internal/{entry}.txt"


def parse_modified_since(value: str):
    raw = (value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main_internal():
    cfg = load_config()
    list_file = os.environ.get("OCR_SOURCE_LIST_FILE", "").strip()
    output_prefix = os.environ.get("OUTPUT_PREFIX", "extracted-text")
    modified_since = parse_modified_since(os.environ.get("MODIFIED_SINCE", ""))

    container_override = os.environ.get("CONTAINER_NAME", "legal-documents")
    account_url, sas, container = _resolve_account_and_container(
        cfg.container_sas_url,
        container_override=container_override,
    )
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
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

    txt_files = []
    if list_file:
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"OCR source list file not found: {list_file}")
        with open(list_file, encoding="utf-8", errors="ignore") as fh:
            source_entries = [line.strip() for line in fh if line.strip()]
        txt_files = [ocr_blob_path_from_source_entry(entry, output_prefix) for entry in source_entries]
        logging.info(
            f"Loaded {len(txt_files)} target OCR blobs from source list: {list_file}"
        )
    else:
        prefix = f"{output_prefix.strip('/')}/internal/"
        logging.info(f"Listing blobs with prefix: {prefix}")
        for blob in container_client.list_blobs(name_starts_with=prefix):
            if not blob.name.lower().endswith(".txt"):
                continue
            if modified_since and getattr(blob, "last_modified", None):
                blob_modified = blob.last_modified.astimezone(timezone.utc)
                if blob_modified < modified_since:
                    continue
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

            # If OCR file has Source: header (from updated ocr_internal.py),
            # extract original path for correct metadata and strip header before chunking
            meta = extract_metadata_from_path(blob_path, cfg.input_prefix)
            source_path = extract_source_path_from_content(content)
            if source_path:
                parts = source_path.split("/")
                meta = {
                    "file_name": parts[-1],
                    "folder_path": "/".join(parts[:-1]),
                    "blob_path": blob_path,
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
