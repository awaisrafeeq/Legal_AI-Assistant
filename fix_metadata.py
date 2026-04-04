"""
Fix metadata (folder_path, file_name) in the external search index.
Does NOT regenerate embeddings — only updates metadata fields via merge.
Fixes broken source URLs caused by process_all_formats.py flattening folder paths.
"""
import os
import hashlib
import logging
from typing import Optional
from urllib.parse import urlparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


def split_container_sas_url(url: str):
    p = urlparse(url)
    account_url = f"{p.scheme}://{p.netloc}"
    container = p.path.strip("/").split("/")[-1]
    return account_url, container, p.query


def generate_doc_id(blob_path: str, chunk_index: int) -> str:
    hash_input = f"{blob_path}:{chunk_index}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def extract_source_from_content(content: str) -> Optional[str]:
    """Extract original source path from process_all_formats.py header."""
    first_line = content.split("\n", 1)[0].strip()
    if first_line.startswith("Source: "):
        return first_line[len("Source: "):].strip()
    return None


def extract_metadata_from_path(blob_path: str, input_prefix: str):
    """Fallback: derive metadata from blob path (for ocr_to_blob.py files)."""
    relative_path = blob_path[len(input_prefix):].lstrip("/")
    parts = relative_path.split("/")
    file_name = parts[-1].replace(".txt", "") if parts else relative_path.replace(".txt", "")
    folder_path = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return file_name, folder_path


def main():
    container_sas_url = os.environ["CONTAINER_SAS_URL"]
    search_endpoint = os.environ["SEARCH_ENDPOINT"]
    search_key = os.environ["SEARCH_KEY"]
    search_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    input_prefix = os.environ.get("OUTPUT_PREFIX", "extracted-text")

    account_url, container, sas = split_container_sas_url(container_sas_url)
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=search_index,
        credential=AzureKeyCredential(search_key)
    )

    # List all OCR text files
    prefix = input_prefix.strip("/") + "/" if input_prefix else ""
    txt_files = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if blob.name.endswith(".txt") and not blob.name.startswith("extracted-text/internal/"):
            txt_files.append(blob.name)

    logger.info(f"Found {len(txt_files)} text files to check metadata")

    fixed = 0
    skipped = 0
    batch = []
    batch_size = 1000

    for blob_path in tqdm(txt_files, desc="Fixing metadata"):
        try:
            # Read only first line to check for Source: header (fast - small range)
            blob_client = container_client.get_blob_client(blob_path)
            # Download just first 500 bytes to get the Source: line
            first_bytes = blob_client.download_blob(offset=0, length=500).readall().decode("utf-8", errors="ignore")

            source_path = extract_source_from_content(first_bytes)

            if source_path:
                # process_all_formats.py file — fix metadata from Source: header
                parts = source_path.split("/")
                file_name = parts[-1]
                folder_path = "/".join(parts[:-1])
            else:
                # ocr_to_blob.py file — derive from path (should already be correct)
                file_name, folder_path = extract_metadata_from_path(blob_path, input_prefix)
                skipped += 1
                continue  # These are already correct, skip

            # Find all chunk IDs for this blob and update them
            # We need to know how many chunks exist — search for them
            results = search_client.search(
                search_text="*",
                filter=f"blob_path eq '{blob_path}'",
                select=["id", "chunk_index", "file_name", "folder_path"],
                top=1000
            )

            for doc in results:
                # Only update if metadata is actually wrong
                if doc["file_name"] != file_name or doc["folder_path"] != folder_path:
                    batch.append({
                        "@search.action": "merge",
                        "id": doc["id"],
                        "file_name": file_name,
                        "folder_path": folder_path,
                    })

            if len(batch) >= batch_size:
                result = search_client.upload_documents(documents=batch)
                success = sum(1 for r in result if r.succeeded)
                fixed += success
                logger.info(f"Updated {success} documents")
                batch = []

        except Exception as e:
            logger.error(f"Error processing {blob_path}: {e}")
            continue

    # Flush remaining
    if batch:
        result = search_client.upload_documents(documents=batch)
        success = sum(1 for r in result if r.succeeded)
        fixed += success

    logger.info(f"Done! Fixed: {fixed} | Already correct: {skipped}")


if __name__ == "__main__":
    main()
