"""
Repair .msg/native-message PDF exports that were misclassified as email.

This does not re-OCR or re-index content. It only merges metadata in Azure AI
Search for chunks that are currently document_type="courriel" but clearly look
like SMS/native-message exports.
"""
import logging
import os
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fix_native_message_classification.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
SCAN_BATCH_SIZE = 1000


def normalize_metadata_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_index_field_names(search_endpoint: str, search_key: str, index_name: str) -> set:
    client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_key))
    index = client.get_index(index_name)
    return {field.name for field in index.fields}


def is_native_message_chunk(content: str, file_name: str = "", blob_path: str = "") -> bool:
    text = content or ""
    lower = text.lower()
    path_lower = f"{file_name} {blob_path}".lower()

    if "native messages" in lower or "native-message" in lower or ".msg" in path_lower:
        return True

    header = lower[:1500]
    has_message_headers = "from:" in header and "to:" in header and "date:" in header
    phone_count = len(re.findall(r"(?:\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", header))
    email_count = len(re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", header))

    # Real emails also have From/To/Date, so require phone-heavy/no-email shape.
    return has_message_headers and phone_count >= 2 and email_count == 0


def find_candidate_chunks(search_client: SearchClient) -> Dict[str, Dict]:
    groups: Dict[str, Dict] = {}
    skip = 0

    while True:
        results = search_client.search(
            search_text="*",
            filter="document_type eq 'courriel' or document_type eq 'Courriel' or document_type eq 'COURRIEL'",
            select=["id", "blob_path", "file_name", "folder_path", "content", "source_container"],
            top=SCAN_BATCH_SIZE,
            skip=skip,
        )

        count = 0
        for doc in results:
            count += 1
            content = doc.get("content", "") or ""
            file_name = doc.get("file_name", "") or ""
            blob_path = doc.get("blob_path", "") or ""
            if not is_native_message_chunk(content, file_name, blob_path):
                continue

            key = blob_path or f"{doc.get('source_container', '')}/{file_name}"
            group = groups.setdefault(
                key,
                {
                    "file_name": file_name,
                    "blob_path": blob_path,
                    "folder_path": doc.get("folder_path", "") or "",
                    "source_container": doc.get("source_container", "") or "",
                    "ids": [],
                    "sample": content[:1200],
                },
            )
            group["ids"].append(doc["id"])

        if count < SCAN_BATCH_SIZE:
            break
        skip += SCAN_BATCH_SIZE

    return groups


def build_patch_doc(chunk_id: str, supported_fields: set) -> Dict:
    metadata = {
        "document_type": "correspondance",
        "document_subtype": "message_texte",
        "summary": (
            "Export de messages texte ou de messagerie native avec en-têtes From/To/Date. "
            "Ce document n'est pas un courriel classique."
        ),
        "document_type_norm": normalize_metadata_text("correspondance"),
        "document_subtype_norm": normalize_metadata_text("message_texte"),
    }
    doc = {"@search.action": "merge", "id": chunk_id}
    doc.update({k: v for k, v in metadata.items() if k in supported_fields})
    return doc


def merge_patches(search_client: SearchClient, chunk_ids: List[str], supported_fields: set) -> int:
    updated = 0
    batch = [build_patch_doc(chunk_id, supported_fields) for chunk_id in chunk_ids]
    for i in range(0, len(batch), BATCH_SIZE):
        result = search_client.upload_documents(documents=batch[i:i + BATCH_SIZE])
        updated += sum(1 for item in result if item.succeeded)
    return updated


def process_index(label: str, index_name: str, search_endpoint: str, search_key: str) -> None:
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )
    supported_fields = get_index_field_names(search_endpoint, search_key, index_name)

    logger.info("[%s] scanning courriel chunks for native-message exports...", label)
    groups = find_candidate_chunks(search_client)
    total_chunks = sum(len(group["ids"]) for group in groups.values())
    logger.info("[%s] candidates: %s files, %s chunks", label, len(groups), total_chunks)

    updated = 0
    for group in tqdm(groups.values(), desc=f"Fixing {label} native messages"):
        updated += merge_patches(search_client, group["ids"], supported_fields)

    logger.info("[%s] updated chunks: %s", label, updated)


def main() -> int:
    search_endpoint = os.environ["SEARCH_ENDPOINT"]
    search_key = os.environ["SEARCH_KEY"]
    external_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")

    process_index("External", external_index, search_endpoint, search_key)
    process_index("Internal", internal_index, search_endpoint, search_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
