"""
Document Classification & Entity Extraction Pipeline.

Reads full OCR text for every document, sends to GPT-4 for structured
classification, then merges metadata into Azure AI Search indices.

Extracts:
  - document_type, document_subtype
  - persons, organizations, projects
  - key_dates, key_amounts
  - summary (French, 3-5 lines)

Run once to enrich existing indices. Safe to re-run (skips already classified).
"""
import os
import sys
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from functools import partial

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SimpleField,
    SearchFieldDataType,
    SearchField,
)
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError
from tiktoken import encoding_for_model
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/classify_documents.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Token counting ────────────────────────────────────────────────────
_enc = encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ── Config ────────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS = 100_000   # Leave room for prompt + response in 128K window (system prompt ~10k + margin)
BATCH_SIZE = 500               # Index merge batch size
MAX_WORKERS = 5                # Parallel GPT classification workers (reduced to avoid 429 rate limits)
SAVE_INTERVAL = 50             # Save progress every N documents


# ── Classification prompt ─────────────────────────────────────────────
CLASSIFICATION_PROMPT = """You are a legal document analyst for a Quebec real estate financing fraud case.
You will receive the FULL text of a legal document (OCR-extracted). Analyze the ENTIRE document carefully.

Your task: extract structured metadata from this document.

Return a JSON object with these fields:

{{
  "document_type": "<primary type>",
  "document_subtype": "<specific subtype or null>",
  "persons": ["<full name>", ...],
  "organizations": ["<company/org name>", ...],
  "projects": ["<project/property name or location>", ...],
  "key_dates": ["<YYYY-MM-DD or descriptive date>", ...],
  "key_amounts": ["<amount as stated>", ...],
  "summary": "<3-5 line summary in French>"
}}

DOCUMENT TYPE must be exactly one of:
  déclaration, interrogatoire, courriel, facture, contrat, acte_notarié,
  jugement, ordonnance, requête, bail, hypothèque, évaluation, rapport,
  relevé, certificat, résolution, mandat, procuration, cession,
  mise_en_demeure, procès_verbal, cahier_de_preuve, remise_volontaire,
  divulgation, bordereau, preuve, pièce, plan, photo, états_financiers,
  convention, offre, lettre, chèque, reçu, notes, transcription, autre

DOCUMENT SUBTYPE examples:
  - déclaration → sous_serment, solennelle, affidavit, assermentation
  - interrogatoire → préalable, contre_interrogatoire, examen
  - courriel → investisseur, interne, notaire, comptable, partenaire
  - contrat → prêt, vente, service, location, cession
  - acte_notarié → prêt, vente, hypothèque, cession_créance, cautionnement
  - rapport → évaluation, expert, police, financier, dépôt
  - relevé → bancaire, compte, carte_crédit
  - facture → honoraires, construction, service_professionnel
  Use null if no clear subtype.

RULES:
- Extract ALL person names mentioned anywhere in the document (full names when possible)
- Extract ALL organization/company names (include Quebec company numbers like "9303-4197 Québec inc.")
- For projects: extract property addresses, project names (Brompton, St-Lin, Rawdon, Couvent, etc.), lot numbers
- For dates: extract the most important dates (document date, transaction dates, court dates)
- For amounts: extract dollar amounts with context (e.g., "700 000 $", "$1,144.00", "250 000$")
- Summary must be in French, factual, and cover the key facts of the document
- If the document contains multiple types (e.g., email forwarding a contract), use the PRIMARY document type
- If the document is an email ABOUT a topic (plan, bail, cession), classify as "courriel" not the topic

IMPORTANT: Return ONLY valid JSON. No markdown, no explanation, no code blocks."""


def build_classification_messages(filename: str, content: str) -> list:
    """Build messages for GPT classification call."""
    return [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": f"Filename: {filename}\n\nFull document content:\n\n{content}"}
    ]


def build_chunked_classification_messages(filename: str, contents: List[str], part_num: int, total_parts: int) -> list:
    """Build messages for a single part of a large document."""
    return [
        {"role": "system", "content": CLASSIFICATION_PROMPT + f"\n\nNOTE: This is part {part_num}/{total_parts} of a large document. Extract all metadata you can find in this part."},
        {"role": "user", "content": f"Filename: {filename}\n\nDocument content (part {part_num}/{total_parts}):\n\n{contents[part_num - 1]}"}
    ]


# ── GPT classification ────────────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    reraise=True
)
def call_gpt(client: AzureOpenAI, deployment: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def classify_document(client: AzureOpenAI, deployment: str, filename: str, content: str) -> Optional[Dict]:
    """Classify a single document. Handles large docs by splitting."""
    tokens = count_tokens(content)

    try:
        if tokens <= MAX_CONTEXT_TOKENS:
            # Single call — full document
            messages = build_classification_messages(filename, content)
            raw = call_gpt(client, deployment, messages)
            return json.loads(raw)
        else:
            # Split into parts
            logger.info(f"Large document ({tokens} tokens), splitting: {filename[:80]}")
            parts = split_content(content, MAX_CONTEXT_TOKENS)
            all_results = []

            for i, part in enumerate(parts, 1):
                messages = build_chunked_classification_messages(filename, parts, i, len(parts))
                raw = call_gpt(client, deployment, messages)
                all_results.append(json.loads(raw))

            # Merge results from all parts
            return merge_classifications(all_results)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {filename[:80]}: {e}")
        return None
    except Exception as e:
        logger.error(f"Classification failed for {filename[:80]}: {e}")
        return None


def split_content(content: str, max_tokens: int) -> List[str]:
    """Split content into parts that each fit within max_tokens."""
    tokens = _enc.encode(content)
    parts = []
    for i in range(0, len(tokens), max_tokens):
        part_tokens = tokens[i:i + max_tokens]
        parts.append(_enc.decode(part_tokens))
    return parts


def merge_classifications(results: List[Dict]) -> Dict:
    """Merge classification results from multiple parts of a large document."""
    merged = {
        "document_type": results[0].get("document_type", "autre"),
        "document_subtype": results[0].get("document_subtype"),
        "persons": [],
        "organizations": [],
        "projects": [],
        "key_dates": [],
        "key_amounts": [],
        "summary": results[0].get("summary", ""),
    }

    seen_persons = set()
    seen_orgs = set()
    seen_projects = set()
    seen_dates = set()
    seen_amounts = set()

    for r in results:
        for p in r.get("persons", []):
            if p and p.lower() not in seen_persons:
                seen_persons.add(p.lower())
                merged["persons"].append(p)
        for o in r.get("organizations", []):
            if o and o.lower() not in seen_orgs:
                seen_orgs.add(o.lower())
                merged["organizations"].append(o)
        for pr in r.get("projects", []):
            if pr and pr.lower() not in seen_projects:
                seen_projects.add(pr.lower())
                merged["projects"].append(pr)
        for d in r.get("key_dates", []):
            if d and d not in seen_dates:
                seen_dates.add(d)
                merged["key_dates"].append(d)
        for a in r.get("key_amounts", []):
            if a and a not in seen_amounts:
                seen_amounts.add(a)
                merged["key_amounts"].append(a)

    return merged


# ── Index schema update ───────────────────────────────────────────────

NEW_FIELDS = [
    SearchableField(name="document_type", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
    SearchableField(name="document_subtype", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
    SearchField(name="persons",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True, searchable=True),
    SearchField(name="organizations",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True, searchable=True),
    SearchField(name="projects",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True, searchable=True),
    SearchField(name="key_dates",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True, searchable=True),
    SearchField(name="key_amounts",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True, searchable=True),
    SearchableField(name="summary", type=SearchFieldDataType.String),
]


def update_index_schema(endpoint: str, key: str, index_name: str):
    """Add classification fields to an existing index."""
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    try:
        index = index_client.get_index(index_name)
    except Exception as e:
        logger.error(f"Cannot get index '{index_name}': {e}")
        return False

    existing_names = {f.name for f in index.fields}
    fields_to_add = [f for f in NEW_FIELDS if f.name not in existing_names]

    if not fields_to_add:
        logger.info(f"Index '{index_name}' already has all classification fields")
        return True

    logger.info(f"Adding {len(fields_to_add)} new fields to '{index_name}': {[f.name for f in fields_to_add]}")
    index.fields.extend(fields_to_add)
    index_client.create_or_update_index(index)
    logger.info(f"Index '{index_name}' schema updated successfully")
    return True


# ── Index update (merge metadata into chunks) ─────────────────────────

def update_chunks_metadata(search_client: SearchClient, blob_path: str, metadata: Dict) -> int:
    """Find all chunks for a blob_path and merge classification metadata."""
    safe_path = blob_path.replace("'", "''")
    results = search_client.search(
        search_text="*",
        filter=f"blob_path eq '{safe_path}'",
        select=["id"],
        top=1000
    )

    batch = []
    for doc in results:
        update = {
            "@search.action": "merge",
            "id": doc["id"],
            "document_type": metadata.get("document_type", "autre"),
            "document_subtype": metadata.get("document_subtype") or "",
            "persons": metadata.get("persons", []),
            "organizations": metadata.get("organizations", []),
            "projects": metadata.get("projects", []),
            "key_dates": metadata.get("key_dates", []),
            "key_amounts": metadata.get("key_amounts", []),
            "summary": metadata.get("summary", ""),
        }
        batch.append(update)

    if not batch:
        return 0

    # Upload in sub-batches
    updated = 0
    for i in range(0, len(batch), BATCH_SIZE):
        sub = batch[i:i + BATCH_SIZE]
        result = search_client.upload_documents(documents=sub)
        updated += sum(1 for r in result if r.succeeded)

    return updated


# ── Helpers ───────────────────────────────────────────────────────────

def split_sas_url(url: str) -> Tuple[str, str, str]:
    p = urlparse(url)
    account_url = f"{p.scheme}://{p.netloc}"
    container = p.path.strip("/").split("/")[-1]
    return account_url, container, p.query


def strip_ocr_header(content: str) -> Tuple[str, Optional[str]]:
    """Strip Source:/Type:/=== header. Returns (clean_content, source_path)."""
    source_path = None
    first_line = content.split("\n", 1)[0].rstrip()
    if first_line.startswith("Source: "):
        source_path = first_line[len("Source: "):]
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("=" * 10):
                content = "\n".join(lines[i + 1:]).lstrip("\n")
                break
    return content, source_path


def get_filename_from_blob(blob_path: str, ocr_prefix: str) -> str:
    """Extract original filename from OCR blob path."""
    relative = blob_path
    if ocr_prefix and relative.startswith(ocr_prefix):
        relative = relative[len(ocr_prefix):].lstrip("/")
    if relative.endswith(".txt"):
        relative = relative[:-4]
    parts = relative.split("/")
    return parts[-1] if parts else relative


# ── Progress tracking ─────────────────────────────────────────────────

PROGRESS_FILE = "logs/classify_progress.json"
FAILED_FILE = "logs/failed_documents.json"

def load_progress() -> set:
    """Load set of already-classified blob paths."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


def load_failed() -> set:
    """Load set of failed blob paths for retry."""
    if os.path.exists(FAILED_FILE):
        with open(FAILED_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


def save_failed(failed: set):
    """Save failed files to disk."""
    with open(FAILED_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(failed), f)


def save_progress(done: set):
    """Save progress to disk."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(done), f)


# ── Parallel processing ───────────────────────────────────────────────

_progress_lock = Lock()
_stats_lock = Lock()


def process_single_document(
    blob_path: str,
    container_client,
    ocr_prefix: str,
    gpt_client: AzureOpenAI,
    gpt_deployment: str,
) -> Optional[Tuple[str, Dict, int]]:
    """
    Process a single document end-to-end.
    Returns: (blob_path, metadata, chunk_count) or None if failed.
    """
    try:
        # 1. Download OCR text
        blob_client = container_client.get_blob_client(blob_path)
        raw_content = blob_client.download_blob().readall().decode("utf-8", errors="ignore")

        if not raw_content.strip():
            return blob_path, None, 0  # Empty document, mark as done

        # 2. Strip header, get original filename
        content, source_path = strip_ocr_header(raw_content)
        if source_path:
            filename = source_path.split("/")[-1]
        else:
            filename = get_filename_from_blob(blob_path, ocr_prefix)

        # 3. Classify with GPT
        metadata = classify_document(gpt_client, gpt_deployment, filename, content)

        if metadata is None:
            return blob_path, None, -1  # Failed classification

        return blob_path, metadata, 0  # Success (chunk count will be updated later)

    except Exception as e:
        logger.error(f"Error processing {blob_path[:80]}: {e}")
        return blob_path, None, -1  # Failed


# ── Main pipeline ─────────────────────────────────────────────────────

def process_container(
    label: str,
    ocr_sas_url: str,
    ocr_prefix: str,
    search_client: SearchClient,
    gpt_client: AzureOpenAI,
    gpt_deployment: str,
    done_set: set,
    failed_set: set = None,
    retry_failed_only: bool = False,
):
    """Process all OCR files in a container prefix using parallel workers."""
    account_url, container, sas = split_sas_url(ocr_sas_url)
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    # List all .txt files under the OCR prefix
    prefix = ocr_prefix.strip("/") + "/" if ocr_prefix else ""
    txt_files = []
    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        if blob.name.endswith(".txt"):
            txt_files.append(blob.name)

    logger.info(f"[{label}] Found {len(txt_files)} OCR files")

    # Filter logic based on retry mode
    if retry_failed_only and failed_set:
        # Only retry previously failed files
        remaining = [f for f in txt_files if f in failed_set]
        logger.info(f"[{label}] Retry mode: {len(remaining)} failed files to retry")
    else:
        # Normal mode: filter out already done
        remaining = [f for f in txt_files if f not in done_set]
        logger.info(f"[{label}] Already classified: {len(txt_files) - len(remaining)} | Remaining: {len(remaining)}")

    if not remaining:
        logger.info(f"[{label}] No files to process")
        return

    classified = 0
    failed = 0
    chunks_updated = 0
    batch_updates = []
    current_failed = set()  # Track failures in this run

    # Create partial function with fixed arguments
    process_fn = partial(
        process_single_document,
        container_client=container_client,
        ocr_prefix=ocr_prefix,
        gpt_client=gpt_client,
        gpt_deployment=gpt_deployment,
    )

    logger.info(f"[{label}] Starting parallel processing with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_blob = {executor.submit(process_fn, blob_path): blob_path for blob_path in remaining}

        # Process completed tasks as they finish
        for future in tqdm(as_completed(future_to_blob), total=len(remaining), desc=f"Classifying {label}"):
            blob_path = future_to_blob[future]
            
            try:
                result = future.result()
                
                if result is None:
                    # Processing failed
                    with _stats_lock:
                        failed += 1
                    with _progress_lock:
                        current_failed.add(blob_path)
                    continue

                blob_path, metadata, status = result

                if status == 0 and metadata is None:
                    # Empty document - mark as done, remove from failed if present
                    with _progress_lock:
                        done_set.add(blob_path)
                        if failed_set is not None and blob_path in failed_set:
                            failed_set.discard(blob_path)
                    continue

                if metadata is None:
                    # Classification failed - track for retry
                    with _stats_lock:
                        failed += 1
                    with _progress_lock:
                        current_failed.add(blob_path)
                    continue

                # Success - prepare batch update
                with _stats_lock:
                    classified += 1
                    
                    # Remove from failed set if retry succeeded
                    with _progress_lock:
                        if failed_set is not None and blob_path in failed_set:
                            failed_set.discard(blob_path)
                    
                    # Create index update batch
                    safe_path = blob_path.replace("'", "''")
                    results = search_client.search(
                        search_text="*",
                        filter=f"blob_path eq '{safe_path}'",
                        select=["id"],
                        top=1000
                    )

                    for doc in results:
                        update = {
                            "@search.action": "merge",
                            "id": doc["id"],
                            "document_type": metadata.get("document_type", "autre"),
                            "document_subtype": metadata.get("document_subtype") or "",
                            "persons": metadata.get("persons", []),
                            "organizations": metadata.get("organizations", []),
                            "projects": metadata.get("projects", []),
                            "key_dates": metadata.get("key_dates", []),
                            "key_amounts": metadata.get("key_amounts", []),
                            "summary": metadata.get("summary", ""),
                        }
                        batch_updates.append(update)

                    # Batch upload when threshold reached
                    if len(batch_updates) >= BATCH_SIZE:
                        _upload_batch(search_client, batch_updates)
                        batch_updates = []

                # Mark as done and save progress periodically
                with _progress_lock:
                    done_set.add(blob_path)
                    if classified % SAVE_INTERVAL == 0:
                        save_progress(done_set)
                        logger.info(f"[{label}] Progress: {classified} classified, {chunks_updated} chunks updated, {failed} failed")

            except Exception as e:
                logger.error(f"[{label}] Error processing {blob_path[:80]}: {e}")
                with _stats_lock:
                    failed += 1
                with _progress_lock:
                    current_failed.add(blob_path)

    # Upload any remaining batch updates
    if batch_updates:
        _upload_batch(search_client, batch_updates)

    # Save progress and failed files
    save_progress(done_set)
    if failed_set is not None:
        # Update global failed set with current failures and save
        failed_set.update(current_failed)
        save_failed(failed_set)
        logger.info(f"[{label}] DONE — Classified: {classified} | Chunks updated: {chunks_updated} | Failed: {failed} | Failed set size: {len(failed_set)}")
    else:
        logger.info(f"[{label}] DONE — Classified: {classified} | Chunks updated: {chunks_updated} | Failed: {failed}")


def _upload_batch(search_client: SearchClient, batch: List[Dict]) -> int:
    """Upload a batch of documents to the search index."""
    try:
        result = search_client.upload_documents(documents=batch)
        succeeded = sum(1 for r in result if r.succeeded)
        logger.info(f"Batch upload: {succeeded}/{len(batch)} documents updated")
        return succeeded
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        return 0


def main():
    # Check for retry mode
    retry_failed_only = "--retry" in sys.argv or "-r" in sys.argv
    
    # Config
    search_endpoint = os.environ["SEARCH_ENDPOINT"]
    search_key = os.environ["SEARCH_KEY"]
    external_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")
    container_sas = os.environ["CONTAINER_SAS_URL"]
    openai_endpoint = os.environ["OPENAI_ENDPOINT"]
    openai_key = os.environ["OPENAI_KEY"]
    gpt_deployment = os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat")

    # GPT client
    gpt_client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-02-01",
    )

    # Step 1: Update index schemas
    logger.info("=" * 60)
    logger.info("STEP 1: Updating index schemas")
    logger.info("=" * 60)
    update_index_schema(search_endpoint, search_key, external_index)
    update_index_schema(search_endpoint, search_key, internal_index)

    # Step 2: Load progress and failed files
    done_set = load_progress()
    failed_set = load_failed()
    logger.info(f"Loaded progress: {len(done_set)} already classified")
    logger.info(f"Failed files to retry: {len(failed_set)}")
    
    if retry_failed_only:
        logger.info("*** RETRY MODE: Only processing previously failed files ***")

    # Step 3: Process external documents
    logger.info("=" * 60)
    logger.info("STEP 2: Classifying EXTERNAL documents")
    logger.info("=" * 60)

    ext_search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=external_index,
        credential=AzureKeyCredential(search_key)
    )
    process_container(
        label="External",
        ocr_sas_url=container_sas,
        ocr_prefix="extracted-text",
        search_client=ext_search_client,
        gpt_client=gpt_client,
        gpt_deployment=gpt_deployment,
        done_set=done_set,
        failed_set=failed_set,
        retry_failed_only=retry_failed_only,
    )

    # Step 4: Process internal documents
    logger.info("=" * 60)
    logger.info("STEP 3: Classifying INTERNAL documents")
    logger.info("=" * 60)

    int_search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=internal_index,
        credential=AzureKeyCredential(search_key)
    )
    process_container(
        label="Internal",
        ocr_sas_url=container_sas,
        ocr_prefix="extracted-text/internal",
        search_client=int_search_client,
        gpt_client=gpt_client,
        gpt_deployment=gpt_deployment,
        done_set=done_set,
        failed_set=failed_set,
        retry_failed_only=retry_failed_only,
    )

    logger.info("=" * 60)
    logger.info("ALL DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
