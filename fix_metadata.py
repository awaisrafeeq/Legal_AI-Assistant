"""
Fix metadata (folder_path, file_name) in the external search index.
Does NOT regenerate embeddings — only updates metadata fields via merge.
Handles:
1. process_all_formats.py files (have Source: header) — extract original path from header
2. ocr_to_blob.py files (no header) — derive path from blob path structure
3. Unicode NFC normalization for French accented characters
4. Leading spaces in folder names (e.g., "  Tableau de preuve...")
"""
import os
import sys
import hashlib
import logging
import unicodedata
from typing import Optional
from urllib.parse import urlparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("logs", exist_ok=True)

_log_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
_file_handler = logging.FileHandler("logs/fix_metadata.log", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

load_dotenv()


def split_container_sas_url(url: str):
    p = urlparse(url)
    account_url = f"{p.scheme}://{p.netloc}"
    container = p.path.strip("/").split("/")[-1]
    return account_url, container, p.query


def extract_source_from_content(content: str) -> Optional[str]:
    """Extract original source path from process_all_formats.py header.
    Preserves leading spaces — some folder names start with spaces."""
    first_line = content.split("\n", 1)[0].rstrip()
    if first_line.startswith("Source: "):
        return first_line[len("Source: "):]
    return None


def extract_metadata_from_path(blob_path: str, input_prefix: str):
    """Derive file_name and folder_path from the OCR blob path."""
    relative_path = blob_path[len(input_prefix):].lstrip("/")
    parts = relative_path.split("/")
    file_name = parts[-1].replace(".txt", "") if parts else relative_path.replace(".txt", "")
    folder_path = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return file_name, folder_path


def escape_odata(value: str) -> str:
    """Escape single quotes for OData filter expressions."""
    return value.replace("'", "''")


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
    errors = 0
    batch = []
    batch_size = 1000

    for blob_path in tqdm(txt_files, desc="Fixing metadata"):
        try:
            blob_client = container_client.get_blob_client(blob_path)
            first_bytes = blob_client.download_blob(offset=0, length=500).readall().decode("utf-8", errors="ignore")

            source_path = extract_source_from_content(first_bytes)

            if source_path:
                # process_all_formats.py file — use Source: header
                source_path = unicodedata.normalize("NFC", source_path)
                parts = source_path.split("/")
                file_name = parts[-1]
                folder_path = "/".join(parts[:-1])
            else:
                # ocr_to_blob.py file — derive from blob path
                file_name, folder_path = extract_metadata_from_path(blob_path, input_prefix)
                file_name = unicodedata.normalize("NFC", file_name)
                folder_path = unicodedata.normalize("NFC", folder_path)

            # Search for all chunks of this blob in the index
            safe_blob_path = escape_odata(blob_path)
            results = search_client.search(
                search_text="*",
                filter=f"blob_path eq '{safe_blob_path}'",
                select=["id", "file_name", "folder_path"],
                top=1000
            )

            for doc in results:
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
                logger.info(f"Batch updated: {success} documents")
                batch = []

        except Exception as e:
            errors += 1
            logger.error(f"Error: {blob_path[:80]}... | {e}")
            continue

    # Flush remaining
    if batch:
        result = search_client.upload_documents(documents=batch)
        success = sum(1 for r in result if r.succeeded)
        fixed += success

    logger.info(f"Done! Fixed: {fixed} | Skipped: {skipped} | Errors: {errors}")


if __name__ == "__main__":
    main()
