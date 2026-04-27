import logging
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

from fix_classification import (
    classify_document,
    get_index_field_names,
    merge_metadata_by_ids_filtered,
)


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_unclassified_internal_groups(search_client: SearchClient):
    groups = {}
    skip = 0
    batch = 1000
    while skip <= 100000:
        results = search_client.search(
            search_text="*",
            filter="source_container eq 'legal-documents-internal' and (document_type eq '' or document_type eq null)",
            select=["id", "blob_path", "file_name", "chunk_index", "content"],
            top=batch,
            skip=skip,
        )
        count = 0
        for doc in results:
            count += 1
            blob_path = (doc.get("blob_path") or "").strip()
            if not blob_path:
                continue
            entry = groups.setdefault(
                blob_path,
                {
                    "file_name": doc.get("file_name") or blob_path.split("/")[-1].removesuffix(".txt"),
                    "chunks": [],
                    "chunk_ids": [],
                },
            )
            entry["chunks"].append((doc.get("chunk_index", 0), doc.get("content", "")))
            entry["chunk_ids"].append(doc["id"])
        if count < batch:
            break
        skip += batch
    return groups


def main():
    search_endpoint = os.environ["SEARCH_ENDPOINT"]
    search_key = os.environ["SEARCH_KEY"]
    internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")
    openai_endpoint = os.environ["OPENAI_ENDPOINT"]
    openai_key = os.environ["OPENAI_KEY"]
    gpt_deployment = os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat")
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=internal_index,
        credential=AzureKeyCredential(search_key),
    )
    supported_fields = get_index_field_names(search_endpoint, search_key, internal_index)
    gpt_client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-02-01",
    )

    groups = find_unclassified_internal_groups(search_client)
    logger.info(f"Found {len(groups)} unclassified internal OCR files from search index")
    classified = 0
    skipped = 0

    for blob_path, group in tqdm(groups.items(), total=len(groups), desc="Classifying internal files"):
        try:
            ordered_chunks = [content for _, content in sorted(group["chunks"], key=lambda item: item[0])]
            content = "\n\n".join(chunk for chunk in ordered_chunks if chunk)
            if not content.strip():
                skipped += 1
                continue

            filename = group["file_name"]
            metadata = classify_document(gpt_client, gpt_deployment, filename, content)
            if not metadata:
                skipped += 1
                continue

            chunk_ids = group["chunk_ids"]
            if not chunk_ids:
                skipped += 1
                continue

            merge_metadata_by_ids_filtered(search_client, chunk_ids, metadata, supported_fields)
            classified += 1
        except Exception as e:
            logger.error(f"Failed to classify {blob_path}: {e}")

    logger.info(f"Incremental internal classification complete | classified={classified} | skipped={skipped}")


if __name__ == "__main__":
    main()
