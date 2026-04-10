"""
Fix classification gaps: re-classify unclassified documents and properly merge metadata.

Handles:
1. Files where GPT classification succeeded but index merge failed (Unicode/quote issues)
2. Files that failed classification (429/500/400 errors)
3. Uses search-based approach to find unclassified chunks instead of OData blob_path filter

Run: python fix_classification.py
"""
import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
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
        logging.FileHandler("logs/fix_classification.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

_enc = encoding_for_model("gpt-4")
MAX_CONTEXT_TOKENS = 100_000
MAX_WORKERS = 3  # Lower to avoid 429s
BATCH_SIZE = 500
CACHE_FILE = "logs/classification_cache.json"

# ── Same classification prompt from classify_documents.py ────────────
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


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def split_content(content: str, max_tokens: int) -> List[str]:
    tokens = _enc.encode(content)
    parts = []
    for i in range(0, len(tokens), max_tokens):
        parts.append(_enc.decode(tokens[i:i + max_tokens]))
    return parts


def merge_classifications(results: List[Dict]) -> Dict:
    merged = {
        "document_type": results[0].get("document_type", "autre"),
        "document_subtype": results[0].get("document_subtype"),
        "persons": [], "organizations": [], "projects": [],
        "key_dates": [], "key_amounts": [],
        "summary": results[0].get("summary", ""),
    }
    seen = {"persons": set(), "organizations": set(), "projects": set(),
            "key_dates": set(), "key_amounts": set()}
    for r in results:
        for field in seen:
            for val in r.get(field, []):
                if val and val.lower() not in seen[field]:
                    seen[field].add(val.lower())
                    merged[field].append(val)
    return merged


def strip_ocr_header(content: str) -> Tuple[str, Optional[str]]:
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


# ── GPT call with retry ──────────────────────────────────────────────
@retry(
    wait=wait_exponential(multiplier=3, min=10, max=180),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    reraise=True
)
def call_gpt(client: AzureOpenAI, deployment: str, messages: list) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def repair_json(raw: str) -> Optional[Dict]:
    """Try to repair truncated JSON from GPT."""
    # Try as-is first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try closing open strings and brackets
    fixed = cleaned.rstrip()
    # Close any open string
    if fixed.count('"') % 2 != 0:
        fixed += '"'
    # Close open arrays
    open_brackets = fixed.count('[') - fixed.count(']')
    fixed += ']' * max(0, open_brackets)
    # Close open objects
    open_braces = fixed.count('{') - fixed.count('}')
    fixed += '}' * max(0, open_braces)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: extract what we can with regex
    import re
    doc_type = re.search(r'"document_type"\s*:\s*"([^"]*)"', raw)
    summary = re.search(r'"summary"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw)
    return {
        "document_type": doc_type.group(1) if doc_type else "autre",
        "document_subtype": None,
        "persons": re.findall(r'"persons"\s*:\s*\[(.*?)\]', raw, re.DOTALL) and
                   [p.strip(' "') for p in re.findall(r'"([^"]+)"', re.search(r'"persons"\s*:\s*\[(.*?)\]', raw, re.DOTALL).group(1))] if re.search(r'"persons"\s*:\s*\[', raw) else [],
        "organizations": [],
        "projects": [],
        "key_dates": [],
        "key_amounts": [],
        "summary": summary.group(1) if summary else f"Classification partielle: {raw[:100]}",
    }


def classify_document(client: AzureOpenAI, deployment: str, filename: str, content: str) -> Optional[Dict]:
    tokens = count_tokens(content)
    try:
        if tokens <= MAX_CONTEXT_TOKENS:
            messages = [
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"Filename: {filename}\n\nFull document content:\n\n{content}"}
            ]
            raw = call_gpt(client, deployment, messages)
            return repair_json(raw)
        else:
            parts = split_content(content, MAX_CONTEXT_TOKENS)
            all_results = []
            for i, part in enumerate(parts, 1):
                messages = [
                    {"role": "system", "content": CLASSIFICATION_PROMPT + f"\n\nNOTE: This is part {i}/{len(parts)} of a large document."},
                    {"role": "user", "content": f"Filename: {filename}\n\nDocument content (part {i}/{len(parts)}):\n\n{part}"}
                ]
                raw = call_gpt(client, deployment, messages)
                parsed = repair_json(raw)
                if parsed:
                    all_results.append(parsed)
            return merge_classifications(all_results) if all_results else None
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {filename[:80]}: {e}")
        return None
    except Exception as e:
        err_str = str(e)
        if "content" in err_str.lower() and "filter" in err_str.lower():
            logger.warning(f"Content filter triggered for {filename[:80]}, using fallback")
            return {
                "document_type": "autre",
                "document_subtype": "contenu_filtré",
                "persons": [], "organizations": [], "projects": [],
                "key_dates": [], "key_amounts": [],
                "summary": f"Document filtré par le système de sécurité. Nom du fichier: {filename}"
            }
        if "maximum context length" in err_str.lower():
            # Token overflow — truncate content and retry
            logger.warning(f"Context overflow for {filename[:80]}, truncating to 90K tokens")
            truncated = _enc.decode(_enc.encode(content)[:90000])
            try:
                messages = [
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": f"Filename: {filename}\n\nFull document content (truncated):\n\n{truncated}"}
                ]
                raw = call_gpt(client, deployment, messages)
                return repair_json(raw)
            except Exception as e2:
                logger.error(f"Retry after truncation also failed for {filename[:80]}: {e2}")
                return None
        if "500" in err_str:
            # Server error — classify from filename only as last resort
            logger.warning(f"Server error for {filename[:80]}, classifying from filename only")
            try:
                messages = [
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": f"Filename: {filename}\n\nThe document content could not be processed. Classify based on filename only."}
                ]
                raw = call_gpt(client, deployment, messages)
                return repair_json(raw)
            except Exception:
                pass
        logger.error(f"Classification failed for {filename[:80]}: {e}")
        return None


# ── Cache management ─────────────────────────────────────────────────
def load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# ── Find unclassified chunk IDs ──────────────────────────────────────
def find_unclassified_chunks(search_client: SearchClient) -> Dict[str, List[str]]:
    """
    Find all chunks without document_type, grouped by blob_path.
    Returns: {blob_path: [chunk_id1, chunk_id2, ...]}
    """
    blob_to_chunks = {}
    skip = 0
    batch = 1000

    while skip <= 100000:
        try:
            results = search_client.search(
                search_text="*",
                filter="document_type eq '' or document_type eq null",
                select=["id", "blob_path"],
                top=batch,
                skip=skip
            )
            count = 0
            for doc in results:
                count += 1
                bp = doc.get("blob_path", "")
                if bp not in blob_to_chunks:
                    blob_to_chunks[bp] = []
                blob_to_chunks[bp].append(doc["id"])

            if count < batch:
                break
            skip += batch
        except Exception as e:
            logger.error(f"Error scanning at skip={skip}: {e}")
            break

    return blob_to_chunks


def merge_metadata_by_ids(search_client: SearchClient, chunk_ids: List[str], metadata: Dict) -> int:
    """Merge metadata directly by chunk IDs — avoids OData blob_path filter issues."""
    batch = []
    for cid in chunk_ids:
        batch.append({
            "@search.action": "merge",
            "id": cid,
            "document_type": metadata.get("document_type", "autre"),
            "document_subtype": metadata.get("document_subtype") or "",
            "persons": metadata.get("persons", []),
            "organizations": metadata.get("organizations", []),
            "projects": metadata.get("projects", []),
            "key_dates": metadata.get("key_dates", []),
            "key_amounts": metadata.get("key_amounts", []),
            "summary": metadata.get("summary", ""),
        })

    updated = 0
    for i in range(0, len(batch), BATCH_SIZE):
        sub = batch[i:i + BATCH_SIZE]
        try:
            result = search_client.upload_documents(documents=sub)
            updated += sum(1 for r in result if r.succeeded)
        except Exception as e:
            logger.error(f"Batch merge failed: {e}")

    return updated


def split_sas_url(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}", p.path.strip("/").split("/")[-1], p.query


_lock = Lock()


def process_one(blob_path, chunk_ids, container_client, gpt_client, gpt_deployment, cache):
    """Classify one file and return (blob_path, metadata, chunk_ids) or None."""
    # Check cache first
    if blob_path in cache:
        return blob_path, cache[blob_path], chunk_ids

    try:
        blob_client = container_client.get_blob_client(blob_path)
        raw = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
        if not raw.strip():
            meta = {"document_type": "autre", "document_subtype": "vide",
                    "persons": [], "organizations": [], "projects": [],
                    "key_dates": [], "key_amounts": [],
                    "summary": "Document vide."}
            with _lock:
                cache[blob_path] = meta
            return blob_path, meta, chunk_ids

        content, source_path = strip_ocr_header(raw)
        if source_path:
            filename = source_path.split("/")[-1]
        else:
            filename = blob_path.split("/")[-1]
            if filename.endswith(".txt"):
                filename = filename[:-4]

        metadata = classify_document(gpt_client, gpt_deployment, filename, content)
        if metadata:
            with _lock:
                cache[blob_path] = metadata
            return blob_path, metadata, chunk_ids
        else:
            return None

    except Exception as e:
        logger.error(f"Error: {blob_path[:80]}: {e}")
        return None


def process_index(label, index_name, ocr_sas_url, search_endpoint, search_key,
                  gpt_client, gpt_deployment, cache):
    """Find and fix all unclassified chunks in an index."""
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )

    logger.info(f"[{label}] Scanning for unclassified chunks...")
    blob_to_chunks = find_unclassified_chunks(search_client)
    total_files = len(blob_to_chunks)
    total_chunks = sum(len(v) for v in blob_to_chunks.values())
    logger.info(f"[{label}] Found {total_files} files with {total_chunks} unclassified chunks")

    if not blob_to_chunks:
        return

    account_url, container, sas = split_sas_url(ocr_sas_url)
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    # Check how many are already in cache (just need merge)
    cached = sum(1 for bp in blob_to_chunks if bp in cache)
    need_gpt = total_files - cached
    logger.info(f"[{label}] {cached} in cache (merge only), {need_gpt} need GPT classification")

    classified = 0
    merged_chunks = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for bp, cids in blob_to_chunks.items():
            f = executor.submit(process_one, bp, cids, container_client,
                                gpt_client, gpt_deployment, cache)
            futures[f] = bp

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fixing {label}"):
            try:
                result = future.result()
                if result is None:
                    failed += 1
                    continue

                bp, metadata, chunk_ids = result
                # Merge into index by chunk IDs
                updated = merge_metadata_by_ids(search_client, chunk_ids, metadata)
                merged_chunks += updated
                classified += 1

                # Save cache periodically
                if classified % 50 == 0:
                    with _lock:
                        save_cache(cache)
                    logger.info(f"[{label}] Progress: {classified}/{total_files} classified, "
                                f"{merged_chunks} chunks merged, {failed} failed")

            except Exception as e:
                logger.error(f"[{label}] Future error: {e}")
                failed += 1

    # Final save
    save_cache(cache)
    logger.info(f"[{label}] DONE — Classified: {classified} | Chunks merged: {merged_chunks} | Failed: {failed}")


def main():
    search_endpoint = os.environ["SEARCH_ENDPOINT"]
    search_key = os.environ["SEARCH_KEY"]
    external_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")
    container_sas = os.environ["CONTAINER_SAS_URL"]
    openai_endpoint = os.environ["OPENAI_ENDPOINT"]
    openai_key = os.environ["OPENAI_KEY"]
    gpt_deployment = os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat")

    gpt_client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-02-01",
    )

    cache = load_cache()
    logger.info(f"Loaded classification cache: {len(cache)} entries")

    logger.info("=" * 60)
    logger.info("FIXING EXTERNAL INDEX")
    logger.info("=" * 60)
    process_index("External", external_index, container_sas, search_endpoint, search_key,
                  gpt_client, gpt_deployment, cache)

    logger.info("=" * 60)
    logger.info("FIXING INTERNAL INDEX")
    logger.info("=" * 60)
    process_index("Internal", internal_index, container_sas, search_endpoint, search_key,
                  gpt_client, gpt_deployment, cache)

    logger.info("=" * 60)
    logger.info("ALL DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
