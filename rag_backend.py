import prompts


from cgitb import lookup
import os
import json
import logging
from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass, field
import unicodedata
from contextlib import asynccontextmanager


from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from datetime import datetime, timedelta


from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict

from sentence_transformers import CrossEncoder
from logging.handlers import TimedRotatingFileHandler

os.makedirs("logs", exist_ok=True)

_log_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
_file_handler = TimedRotatingFileHandler(
    "logs/rag_backend.log", when="midnight", backupCount=30, encoding="utf-8"
)
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    search_endpoint: str
    search_key: str
    search_index: str
    search_index_external: str
    search_index_internal: str
    min_search_score: float
    cross_encoder_min_score: float
    openai_endpoint: str
    openai_key: str
    openai_chat_deployment: str
    openai_embedding_deployment: str
    container_sas_url: str
    internal_container_sas_url: str
    acs_connection_string: str
    acs_sender_email: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            search_endpoint=os.environ["SEARCH_ENDPOINT"],
            search_key=os.environ["SEARCH_KEY"],
            search_index=os.environ.get("SEARCH_INDEX", "legal-docs-index"),
            search_index_external=os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external"),
            search_index_internal=os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal"),
            min_search_score=float(os.environ.get("MIN_SEARCH_SCORE", "0.15")),
            cross_encoder_min_score=float(os.environ.get("CROSS_ENCODER_MIN_SCORE", "0.5")),
            openai_endpoint=os.environ["OPENAI_ENDPOINT"],
            openai_key=os.environ["OPENAI_KEY"],
            openai_chat_deployment=os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat"),
            openai_embedding_deployment=os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT", "Embeddings"),
            container_sas_url=os.environ["CONTAINER_SAS_URL"],
            internal_container_sas_url=os.environ.get("INTERNAL_CONTAINER_SAS_URL", ""),
            acs_connection_string=os.environ.get("ACS_CONNECTION_STRING", ""),
            acs_sender_email=os.environ.get("ACS_SENDER_EMAIL", ""),
        )


# ============================================================================
# MODELS
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    content: str
    file_name: str
    folder_path: str
    blob_path: str
    chunk_index: int
    score: float
    source_container: Optional[str] = None
    source_url: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    target_language: Optional[str] = "auto"
    history: Optional[List[ChatMessage]] = []
    source_mode: Optional[str] = "all"  # "all", "internal_only", "external_only"
    allowed_files: Optional[List[str]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    conversation_id: str
    detected_query_language: Optional[str] = None


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str, llm: AzureChatOpenAI) -> str:
    """
    Detect the language of the user's query.
    Returns a language code: 'en', 'fr', 'ur', 'ar', 'es', etc.
    Used to respond in the same language the user wrote in.
    """
    # Fast path: check for Urdu/Arabic unicode characters
    urdu_arabic_chars = sum(
        1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
    )
    if urdu_arabic_chars > 2:
        urdu_specific = sum(1 for c in text if c in 'ے،ہھگڈڑ')
        return "ur" if urdu_specific > 0 else "ar"

    # Fast path: check for French indicators
    french_indicators = [
        'le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'dans',
        'pour', 'avec', 'qui', 'que', 'du', 'au', 'en', 'il', 'elle'
    ]
    text_lower = text.lower()
    french_word_count = sum(
        1 for word in french_indicators if f" {word} " in f" {text_lower} "
    )
    if french_word_count >= 2:
        return "fr"

    # Fallback: use LLM
    prompt = prompts.get_language_detection_prompt(text)
    response = llm.invoke([("human", prompt)])
    result = response.content.strip().lower().replace("'", "").replace('"', '')[:5]

    import re
    match = re.search(r'\b([a-z]{2})\b', result)
    return match.group(1) if match else "en"


def translate_text(llm: AzureChatOpenAI, text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using LLM, preserving legal terminology"""
    if source_lang == target_lang:
        return text

    lang_names = {"en": "English", "fr": "French"}
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    prompt = prompts.get_translation_prompt(source_name, target_name, text)
    response = llm.invoke([("human", prompt)])
    return response.content.strip()


# ============================================================================
# AZURE CLIENTS
# ============================================================================

class AzureClients:
    def __init__(self, config: Config):
        self.config = config

        # Dual-index search clients
        self.search_client_external = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.search_index_external,
            credential=AzureKeyCredential(config.search_key)
        )
        self.search_client_internal = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.search_index_internal,
            credential=AzureKeyCredential(config.search_key)
        )

        from urllib.parse import urlparse
        parsed = urlparse(config.container_sas_url)
        account_url = f"{parsed.scheme}://{parsed.netloc}"
        sas_token = parsed.query
        self.container_name = parsed.path.strip('/').split('/')[-1]
        self.blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)

        self.openai_client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.openai_key,
            api_version="2024-02-01",
        )

        # Cross-encoder for re-ranking search results
        logger.info("Loading cross-encoder model for re-ranking...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        logger.info("Cross-encoder model loaded.")

        self.llm = AzureChatOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.openai_key,
            azure_deployment=config.openai_chat_deployment,
            api_version="2024-02-01",
            temperature=0.1,
        )


# ============================================================================
# RAG STATE
# ============================================================================

class RAGState(TypedDict, total=False):
    query: str
    query_lang: str
    rewritten_query: str
    query_embedding: List[float]
    search_results: List[Dict]
    winning_source: str
    context: str
    answer: str
    sources: List[Dict]
    conversation_history: List[Dict]
    source_mode: str
    allowed_files: List[str]
    lookup_mode: str
    retrieval_attempts: int
    retrieval_sufficient: bool
    sub_queries: List[str]
    sub_results: List[Dict]
    query_variants: List[str]
    query_intent: str  # "discovery" or "answer"
    discovery_filters: Dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def send_email_acs(config, recipient_email: str, subject: str, body: str) -> bool:
    """
    Send an email via Azure Communication Services.
    Returns True if sent successfully, False otherwise.
    """
    if not config.acs_connection_string or not config.acs_sender_email:
        logger.warning("ACS not configured — skipping email send")
        return False
    try:
        from azure.communication.email import EmailClient
        client = EmailClient.from_connection_string(config.acs_connection_string)
        message = {
            "senderAddress": config.acs_sender_email,
            "recipients": {"to": [{"address": recipient_email}]},
            "content": {
                "subject": subject,
                "plainText": body,
            },
        }
        poller = client.begin_send(message)
        result = poller.result()
        logger.info(f"Email sent to {recipient_email} | status: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {e}")
        return False


def extract_email_intent(query: str):
    """
    Check if the user wants to send an email.
    Returns (recipient_email, True) if email intent found, (None, False) otherwise.
    """
    import re
    email_match = re.search(r'[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}', query)
    if not email_match:
        return None, False
    email_keywords = ['email', 'send', 'mail', 'envoyer', 'envoie', 'bhejo', 'bhej']
    if any(kw in query.lower() for kw in email_keywords):
        return email_match.group(0), True
    return None, False


def generate_embedding(azure_clients: AzureClients, text: str) -> List[float]:
    """Generate embedding for query text"""
    response = azure_clients.openai_client.embeddings.create(
        model=azure_clients.config.openai_embedding_deployment,
        input=text
    )
    return response.data[0].embedding


def generate_sas_url(
    azure_clients: AzureClients,
    blob_path: str,
    source_container: str = "",
    folder_path: str = "",
    file_name: str = ""
) -> str:
    """
    Generate a direct download URL for the original PDF.

    Internal:
    - OCR text is indexed from extracted-text/internal/.../file.pdf.txt
    - Original PDF is expected in the INTERNAL container at .../file.pdf

    External:
    - Use the exact indexed blob_path when available.
      This avoids BlobNotFound caused by reconstructing path from folder_path/file_name.
    """
    try:
        from urllib.parse import urlparse, quote

        is_internal = source_container == "legal-documents-internal"

        if is_internal and azure_clients.config.internal_container_sas_url:
            sas_url = azure_clients.config.internal_container_sas_url

            if blob_path.startswith("extracted-text/internal/"):
                blob_file_path = blob_path[len("extracted-text/internal/"):]
            else:
                blob_file_path = file_name or blob_path.split("/")[-1]

            if blob_file_path.endswith(".txt"):
                blob_file_path = blob_file_path[:-4]

        else:
            sas_url = azure_clients.config.container_sas_url

            if folder_path and file_name:
                blob_file_path = f"{folder_path}/{file_name}"
            elif blob_path:
                blob_file_path = blob_path
                if blob_file_path.startswith("extracted-text/"):
                    blob_file_path = blob_file_path[len("extracted-text/"):]
                if blob_file_path.endswith(".txt"):
                    blob_file_path = blob_file_path[:-4]
            else:
                blob_file_path = file_name or ""

        parsed = urlparse(sas_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        container_name = parsed.path.strip("/").split("/")[-1]
        sas_token = parsed.query

        unicode_form = "NFC" if is_internal else "NFD"
        blob_file_path = unicodedata.normalize(unicode_form, blob_file_path)
        encoded_path = quote(blob_file_path, safe="/")
        url = f"{base_url}/{container_name}/{encoded_path}?{sas_token}"
        logger.info(f"SAS URL generated | container={source_container} | path={blob_file_path}")
        return url

    except Exception as e:
        logger.warning(f"Failed to generate URL for {blob_path}: {e}")
        return None


# ============================================================================
# SEARCH & RE-RANKING
# ============================================================================

def _search_one_index(
    search_client: SearchClient,
    query: str,
    query_embedding: List[float],
    top_k: int,
    allowed_files: Optional[List[str]] = None,
    odata_filter: Optional[str] = None
) -> List[Dict]:
    """
    Run hybrid search (BM25 + vector) on a single index.
    Returns raw results without SAS URLs (filled later for the winning source only).
    """
    search_query = query

    if allowed_files:
        file_hint = " ".join(allowed_files)
        search_query = f"{query} {file_hint}"

    vector_query = {
        "kind": "vector",
        "vector": query_embedding,
        "fields": "content_vector",
        "k": top_k * 2,
        "exhaustive": True
    }

    results = search_client.search(
        search_text=search_query,
        vector_queries=[vector_query],
        filter=odata_filter if odata_filter else None,
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        select=[
            "content", "file_name", "folder_path",
            "blob_path", "chunk_index", "source_container",
            "document_type", "document_subtype", "persons",
            "organizations", "projects", "key_dates",
            "key_amounts", "summary"
        ],
        top=top_k * 2
    )

    results_list = [
        {
            "content": r["content"],
            "file_name": r["file_name"],
            "folder_path": r["folder_path"],
            "blob_path": r["blob_path"],
            "chunk_index": r["chunk_index"],
            "source_container": r.get("source_container", "unknown"),
            "score": (r.get("@search.reranker_score") or 0) / 4.0
                    if r.get("@search.reranker_score") is not None
                    else r.get("@search.score", 0),
            "source_url": None,
        }
        for r in results
    ]

    # Filter to allowed files if specified
    if allowed_files:
        allowed_terms = [x.strip().lower() for x in allowed_files if x.strip()]
        results_list = [
            r for r in results_list
            if any(term in r.get("file_name", "").strip().lower() for term in allowed_terms)
        ]

    return results_list


def rerank_results(
    cross_encoder: CrossEncoder,
    query: str,
    results: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    Re-rank search results using a cross-encoder model.
    Scores each (query, chunk) pair jointly for much better precision
    than independent BM25/vector scores.
    """
    if not results:
        return results

    pairs = [(query, r["content"][:512]) for r in results]
    scores = cross_encoder.predict(pairs)

    for result, score in zip(results, scores):
        result["cross_encoder_score"] = float(score)

    reranked = sorted(results, key=lambda r: r["cross_encoder_score"], reverse=True)

    logger.info(
        f"Cross-encoder re-ranking: {len(results)} → top {top_n} | "
        f"best={reranked[0]['cross_encoder_score']:.4f}, "
        f"worst kept={reranked[min(top_n, len(reranked)) - 1]['cross_encoder_score']:.4f}"
    )

    return reranked[:top_n]


def hybrid_search_isolated(
    azure_clients: AzureClients,
    query: str,
    query_embedding: List[float],
    top_k: int = 7,
    source_mode: str = "all",
    allowed_files: Optional[List[str]] = None,
    odata_filter: Optional[str] = None
) -> tuple:
    """
    Search BOTH indices separately, pick the winner by highest score.

    CRITICAL BEHAVIOUR:
    - External index searched independently
    - Internal index searched independently
    - GPT-4 receives context from ONE winning source only — never a mix
    - If both sources score below min_search_score → return ([], "none")
      → no GPT-4 call → no hallucination possible

    Pipeline order:
    1. Search both indices (BM25 + vector)
    2. Re-rank with cross-encoder (sets cross_encoder_score)
    3. Filter by cross-encoder minimum threshold
    4. Filter by Azure MIN_SCORE threshold
    5. Pick winning source by best cross-encoder score

    Returns: (results: List[Dict], winning_source: str)
    """
    MIN_SCORE = azure_clients.config.min_search_score
    CROSS_ENCODER_MIN = azure_clients.config.cross_encoder_min_score
    RERANK_TOP_N = 15

    if allowed_files:
        MIN_SCORE = 0.01  # Reduced but not zero — prevents total garbage

    allowed_files_set = set(allowed_files or [])

    external_results = []
    internal_results = []

    # --- Step 1: Search both indices (with optional metadata filter) ---
    if source_mode in ("all", "external_only"):
        external_results = _search_one_index(
            azure_clients.search_client_external,
            query, query_embedding, top_k,
            allowed_files=list(allowed_files_set) if allowed_files_set else None,
            odata_filter=odata_filter
        )

    if source_mode in ("all", "internal_only"):
        internal_results = _search_one_index(
            azure_clients.search_client_internal,
            query, query_embedding, top_k,
            allowed_files=list(allowed_files_set) if allowed_files_set else None,
            odata_filter=odata_filter
        )

    # --- Step 2: Re-rank with cross-encoder (this SETS cross_encoder_score) ---
    if external_results:
        logger.info(f"Re-ranking {len(external_results)} external results with cross-encoder...")
        external_results = rerank_results(
            azure_clients.cross_encoder, query, external_results, top_n=RERANK_TOP_N
        )
    if internal_results:
        logger.info(f"Re-ranking {len(internal_results)} internal results with cross-encoder...")
        internal_results = rerank_results(
            azure_clients.cross_encoder, query, internal_results, top_n=RERANK_TOP_N
        )

    # --- Step 3: Filter by cross-encoder score (AFTER re-ranking set the scores) ---
    if external_results:
        external_results = [
            r for r in external_results
            if r.get("cross_encoder_score", 0) >= CROSS_ENCODER_MIN
        ]
    if internal_results:
        internal_results = [
            r for r in internal_results
            if r.get("cross_encoder_score", 0) >= CROSS_ENCODER_MIN
        ]

    logger.info(
        f"After cross-encoder filtering (threshold={CROSS_ENCODER_MIN}): "
        f"{len(external_results)} external, {len(internal_results)} internal"
    )

    # --- Step 4: Check Azure score thresholds ---
    def _best_score(results):
        return max(
            (r.get("cross_encoder_score", r["score"]) for r in results),
            default=0.0
        )

    ext_best = _best_score(external_results)
    int_best = _best_score(internal_results)

    ext_azure_best = max((r["score"] for r in external_results), default=0.0)
    int_azure_best = max((r["score"] for r in internal_results), default=0.0)

    logger.info(
        f"Re-ranked scores — External best: {ext_best:.4f} | Internal best: {int_best:.4f}"
    )
    logger.info(
        f"Azure scores — External: {ext_azure_best:.4f} | "
        f"Internal: {int_azure_best:.4f} | Threshold: {MIN_SCORE}"
    )

    if ext_azure_best < MIN_SCORE and int_azure_best < MIN_SCORE:
        logger.warning("Both sources below confidence threshold — returning no results")
        return [], "none"

    # Filter by MIN_SCORE (on original Azure scores), then sort by cross-encoder score
    filtered_internal = sorted(
        [r for r in internal_results if r["score"] >= MIN_SCORE],
        key=lambda r: r.get("cross_encoder_score", r["score"]),
        reverse=True
    )
    filtered_external = sorted(
        [r for r in external_results if r["score"] >= MIN_SCORE],
        key=lambda r: r.get("cross_encoder_score", r["score"]),
        reverse=True
    )

    # --- Step 5: Merge both sources, sort by cross-encoder score, take top N ---
    merged = filtered_internal + filtered_external
    if not merged:
        return [], "none"

    merged.sort(
        key=lambda r: r.get("cross_encoder_score", r.get("score", 0)),
        reverse=True
    )
    winning = merged[:RERANK_TOP_N]

    # Determine winning_source label from what actually made it into top N
    sources_in_winning = set(r.get("source_container", "") for r in winning)
    if "legal-documents-internal" in sources_in_winning and len(sources_in_winning) > 1:
        winning_source = "both"
    elif "legal-documents-internal" in sources_in_winning:
        winning_source = "legal-documents-internal"
    else:
        winning_source = "legal-documents"

    # Generate SAS URLs for each result based on its own container
    for r in winning:
        r["source_url"] = generate_sas_url(
            azure_clients,
            r["blob_path"],
            r.get("source_container", ""),
            r.get("folder_path", ""),
            r.get("file_name", "")
        )

    logger.info(
        f"Final: {len(winning)} chunks | Source: {winning_source} | "
        f"Best CE: {winning[0].get('cross_encoder_score', 0):.4f} | "
        f"Best Azure: {winning[0]['score']:.4f}"
    )

    return winning, winning_source


# ============================================================================
# DISCOVERY SEARCH (metadata-filtered, high-recall)
# ============================================================================

def _build_person_variations(person: str) -> List[str]:
    """
    Generate name variations for fuzzy matching.
    E.g., 'Yves Blach' → ['Yves Blach', 'Yves Blache', 'Yves Blachi', 'Blach', 'Blache']
    """
    import unicodedata as _ud

    variations = set()
    variations.add(person)

    # Split into parts
    parts = person.strip().split()
    if len(parts) >= 2:
        first_name = parts[0]
        last_name = " ".join(parts[1:])

        # Last name only (catches "Blache" when searching "Blach")
        variations.add(last_name)

        # Common French name endings: add/remove trailing 'e'
        if last_name.endswith('e'):
            variations.add(f"{first_name} {last_name[:-1]}")  # Blache → Blach
            variations.add(last_name[:-1])
        else:
            variations.add(f"{first_name} {last_name}e")  # Blach → Blache
            variations.add(f"{last_name}e")

    # Strip accents version
    nfkd = _ud.normalize("NFKD", person)
    no_accents = "".join(c for c in nfkd if not _ud.combining(c))
    if no_accents != person:
        variations.add(no_accents)

    return list(variations)


def _build_odata_filter(filters: Dict[str, Any]) -> str:
    """Build OData filter string from discovery filter dict."""
    parts = []

    doc_type = filters.get("document_type")
    if doc_type:
        escaped = doc_type.replace("'", "''")
        parts.append(f"document_type eq '{escaped}'")

    person = filters.get("person")
    if person:
        # Build OR filter with all name variations for fuzzy matching
        person_variations = _build_person_variations(person)
        person_clauses = []
        for v in person_variations:
            escaped = v.replace("'", "''")
            person_clauses.append(f"persons/any(p: p eq '{escaped}')")
        if len(person_clauses) == 1:
            parts.append(person_clauses[0])
        else:
            parts.append(f"({' or '.join(person_clauses)})")

    org = filters.get("organization")
    if org:
        escaped = org.replace("'", "''")
        parts.append(f"organizations/any(o: o eq '{escaped}')")

    project = filters.get("project")
    if project:
        escaped = project.replace("'", "''")
        parts.append(f"projects/any(p: p eq '{escaped}')")

    return " and ".join(parts) if parts else ""


def _build_fuzzy_person_filters(person: str) -> List[str]:
    """
    Generate multiple OData filter variations for a person name to handle
    accent differences, case, and partial matches.
    """
    variations = _build_person_variations(person)

    filters = []
    for v in variations:
        escaped = v.replace("'", "''")
        filters.append(f"persons/any(p: p eq '{escaped}')")

    return filters


def discovery_search(
    azure_clients: AzureClients,
    query: str,
    filters: Dict[str, Any],
    source_mode: str = "all",
    max_results: int = 50
) -> List[Dict]:
    """
    High-recall search using metadata filters from classification.
    Returns document-level results (deduped by file), not chunk-level.
    """
    odata_filter = _build_odata_filter(filters)

    # Also build fuzzy person filters for broader recall
    person = filters.get("person")
    person_filters = _build_fuzzy_person_filters(person) if person else []

    keyword = filters.get("keyword")
    search_text = keyword if keyword else query

    all_results = []

    def _run_filtered_search(search_client: SearchClient, filter_str: str, container_label: str):
        """Run a single filtered search and collect results."""
        try:
            results = search_client.search(
                search_text=search_text,
                filter=filter_str if filter_str else None,
                query_type="semantic",
                semantic_configuration_name="semantic-config",
                select=[
                    "content", "file_name", "folder_path",
                    "blob_path", "chunk_index", "source_container",
                    "document_type", "document_subtype", "persons",
                    "organizations", "projects", "key_dates",
                    "key_amounts", "summary"
                ],
                top=max_results
            )
            for r in results:
                all_results.append({
                    "content": r.get("content", ""),
                    "file_name": r.get("file_name", ""),
                    "folder_path": r.get("folder_path", ""),
                    "blob_path": r.get("blob_path", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "source_container": r.get("source_container", container_label),
                    "score": r.get("@search.score", 0),
                    "document_type": r.get("document_type", ""),
                    "document_subtype": r.get("document_subtype", ""),
                    "persons": r.get("persons", []),
                    "organizations": r.get("organizations", []),
                    "projects": r.get("projects", []),
                    "key_dates": r.get("key_dates", []),
                    "key_amounts": r.get("key_amounts", []),
                    "summary": r.get("summary", ""),
                    "source_url": None,
                })
        except Exception as e:
            logger.error(f"Discovery search error on {container_label}: {e}")

    # Search with primary OData filter
    if source_mode in ("all", "external_only"):
        _run_filtered_search(azure_clients.search_client_external, odata_filter, "legal-documents")
    if source_mode in ("all", "internal_only"):
        _run_filtered_search(azure_clients.search_client_internal, odata_filter, "legal-documents-internal")

    # If person filter exists, also try fuzzy variations (broader recall)
    if person_filters and odata_filter:
        for pf in person_filters:
            # Combine person variation with other filters (excluding the original person filter)
            other_parts = []
            doc_type = filters.get("document_type")
            if doc_type:
                other_parts.append(f"document_type eq '{doc_type.replace(chr(39), chr(39)*2)}'")
            project = filters.get("project")
            if project:
                other_parts.append(f"projects/any(p: p eq '{project.replace(chr(39), chr(39)*2)}')")

            fuzzy_filter = " and ".join([pf] + other_parts) if other_parts else pf

            if source_mode in ("all", "external_only"):
                _run_filtered_search(azure_clients.search_client_external, fuzzy_filter, "legal-documents")
            if source_mode in ("all", "internal_only"):
                _run_filtered_search(azure_clients.search_client_internal, fuzzy_filter, "legal-documents-internal")

    # Only do a text-only fallback if OData-filtered search returned 0 results
    if odata_filter and len(all_results) == 0:
        # Build a keyword-rich search text from the filters
        filter_keywords = []
        if filters.get("document_type"):
            filter_keywords.append(filters["document_type"])
        if person:
            filter_keywords.append(person)
        if filters.get("organization"):
            filter_keywords.append(filters["organization"])
        if filters.get("project"):
            filter_keywords.append(filters["project"])
        fallback_text = " ".join(filter_keywords + ([keyword] if keyword else []))

        logger.info(f"Discovery: OData filter returned 0 results, falling back to text search: {fallback_text}")

        # Use the keyword-rich text (not generic search_text) for fallback
        def _run_fallback_search(search_client: SearchClient, container_label: str):
            try:
                results = search_client.search(
                    search_text=fallback_text,
                    filter=None,
                    query_type="semantic",
                    semantic_configuration_name="semantic-config",
                    select=[
                        "content", "file_name", "folder_path",
                        "blob_path", "chunk_index", "source_container",
                        "document_type", "document_subtype", "persons",
                        "organizations", "projects", "key_dates",
                        "key_amounts", "summary"
                    ],
                    top=max_results
                )
                for r in results:
                    all_results.append({
                        "content": r.get("content", ""),
                        "file_name": r.get("file_name", ""),
                        "folder_path": r.get("folder_path", ""),
                        "blob_path": r.get("blob_path", ""),
                        "chunk_index": r.get("chunk_index", 0),
                        "source_container": r.get("source_container", container_label),
                        "score": r.get("@search.score", 0),
                        "document_type": r.get("document_type", ""),
                        "document_subtype": r.get("document_subtype", ""),
                        "persons": r.get("persons", []),
                        "organizations": r.get("organizations", []),
                        "projects": r.get("projects", []),
                        "key_dates": r.get("key_dates", []),
                        "key_amounts": r.get("key_amounts", []),
                        "summary": r.get("summary", ""),
                        "source_url": None,
                    })
            except Exception as e:
                logger.error(f"Discovery fallback search error on {container_label}: {e}")

        if source_mode in ("all", "external_only"):
            _run_fallback_search(azure_clients.search_client_external, "legal-documents")
        if source_mode in ("all", "internal_only"):
            _run_fallback_search(azure_clients.search_client_internal, "legal-documents-internal")

    # Deduplicate by file (keep best chunk per file)
    best_by_file = {}
    for r in all_results:
        file_key = (r.get("source_container", ""), r.get("blob_path", ""), r.get("file_name", ""))
        existing = best_by_file.get(file_key)
        if existing is None or r.get("score", 0) > existing.get("score", 0):
            best_by_file[file_key] = r

    deduped = sorted(best_by_file.values(), key=lambda r: r.get("score", 0), reverse=True)

    # Generate SAS URLs
    for r in deduped:
        r["source_url"] = generate_sas_url(
            azure_clients,
            r["blob_path"],
            r.get("source_container", ""),
            r.get("folder_path", ""),
            r.get("file_name", "")
        )

    logger.info(
        f"Discovery search: {len(all_results)} raw → {len(deduped)} unique files | "
        f"filter: {odata_filter or '(text only)'}"
    )

    return deduped


# ============================================================================
# GRAPH NODES
# ============================================================================

def _parse_query_variants(response_text: str) -> List[str]:
    """Parse V1:/V2:/V3: prefixed lines from LLM response into a list of query strings."""
    variants = []
    for line in response_text.strip().splitlines():
        line = line.strip()
        for prefix in ("V1:", "V2:", "V3:"):
            if line.upper().startswith(prefix):
                variant = line[len(prefix):].strip()
                if variant:
                    variants.append(variant)
                break
    return variants


def rewrite_query_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Rewrite user query into multiple French keyword search variations.
    Generates 3 query variants to maximize recall via multi-query retrieval.
    Also detects original query language so the answer is returned in the same language.
    """
    query = state.get("query", "")
    history = state.get("conversation_history", [])

    query_lang = detect_language(query, azure_clients.llm)
    logger.info(f"Detected query language: {query_lang}")

    allowed_files = state.get("allowed_files", [])
    if allowed_files:
        logger.info("Allowed files present — skipping aggressive rewrite")
        return {
            "query": query,
            "query_lang": query_lang,
            "rewritten_query": query,
            "query_variants": [query],
            "conversation_history": history,
            "source_mode": state.get("source_mode", "all"),
            "allowed_files": allowed_files
        }

    history_text = ""
    if history:
        last_turns = history[-4:]
        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in last_turns]
        )
        history_text = f"\nRecent conversation:\n{history_text}\n"

    ql = query.lower()

    query_type = "general legal"
    if any(word in ql for word in [
        "income", "revenue", "profit", "financial",
        "creditors", "bilan", "benefit", "net income"
    ]):
        query_type = "financial"
    elif any(word in ql for word in [
        "email", "courriel", "sender", "recipient",
        "subject", "message", "sent"
    ]):
        query_type = "email"
    elif any(word in ql for word in [
        "invoice", "register", "supplier", "mutation",
        "project", "facture"
    ]):
        query_type = "invoice/register"

    lookup_mode = "document" if any(
        term in ql for term in [
            "pdf", "pdfs", "pièce", "piece", "document original",
            "documents originaux", "fichier", "fichiers", "source",
            "sources", "lien", "liens", "trouve les pdf",
            "retourne les fichiers", "original file", "original files",
            "exclude interrogatoires", "exclure les interrogatoires"
        ]
    ) else "answer"

    # Generate 3 query variations
    prompt = prompts.get_multi_query_rewrite_prompt(history_text, query_type, query)
    response = azure_clients.llm.invoke([("human", prompt)])
    variants = _parse_query_variants(response.content)

    # Fallback: if parsing fails, use single rewrite
    if len(variants) < 2:
        logger.warning("Multi-query parsing failed — falling back to single rewrite")
        fallback_prompt = prompts.get_rewrite_query_prompt(history_text, query_type, query)
        fallback_resp = azure_clients.llm.invoke([("human", fallback_prompt)])
        variants = [fallback_resp.content.strip()]

    logger.info(f"Query variants ({len(variants)}): {[v[:50] for v in variants]}")

    # --- Combined intent detection + filter extraction (single GPT call) ---
    query_intent = "answer"
    discovery_filters = {}

    combined_prompt = prompts.get_combined_intent_filter_prompt(query, history_text)
    combined_resp = azure_clients.llm.invoke([("human", combined_prompt)])
    try:
        raw = combined_resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)

        # Extract intent
        if parsed.get("intent", "").upper() == "DISCOVERY":
            query_intent = "discovery"

        # Extract filters (remove intent key and null values)
        discovery_filters = {k: v for k, v in parsed.items() if v is not None and k != "intent"}
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse combined intent+filters: {e} | raw: {combined_resp.content}")
        # Fallback: try separate intent detection
        try:
            intent_prompt = prompts.get_discovery_intent_prompt(query)
            intent_resp = azure_clients.llm.invoke([("human", intent_prompt)])
            if "DISCOVERY" in intent_resp.content.strip().upper():
                query_intent = "discovery"
        except Exception:
            pass

    logger.info(f"Query intent: {query_intent.upper()} | filters: {discovery_filters}")

    return {
        "query": query,
        "query_lang": query_lang,
        "rewritten_query": variants[0],
        "query_variants": variants,
        "conversation_history": history,
        "lookup_mode": lookup_mode,
        "source_mode": state.get("source_mode", "all"),
        "allowed_files": state.get("allowed_files", []),
        "query_intent": query_intent,
        "discovery_filters": discovery_filters,
    }


def discovery_retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Discovery mode: use metadata filters (document_type, persons, etc.)
    for high-recall document listing. Returns document-level results.
    """
    query = state.get("query", "")
    filters = state.get("discovery_filters", {})
    source_mode = state.get("source_mode", "all")

    results = discovery_search(
        azure_clients, query, filters,
        source_mode=source_mode, max_results=50
    )

    # Determine winning source
    sources_in_results = set(r.get("source_container", "") for r in results)
    if "legal-documents-internal" in sources_in_results and len(sources_in_results) > 1:
        winning_source = "both"
    elif "legal-documents-internal" in sources_in_results:
        winning_source = "legal-documents-internal"
    elif results:
        winning_source = "legal-documents"
    else:
        winning_source = "none"

    logger.info(f"Discovery retrieve: {len(results)} documents found | source: {winning_source}")

    return {
        **state,
        "search_results": results,
        "winning_source": winning_source,
    }


def discovery_generate_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Generate a document listing response for discovery queries.
    Lists all matching documents with their metadata instead of extracting answers.
    """
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    query_lang = state.get("query_lang", "en")
    winning_source = state.get("winning_source", "unknown")
    filters = state.get("discovery_filters", {})

    if not search_results:
        filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items()) if filters else "none"
        return {
            "answer": (
                f"No documents found matching your criteria (filters: {filter_desc}).\n\n"
                "This may mean:\n"
                "- The relevant documents have not been classified yet\n"
                "- The document type or person name may be spelled differently\n"
                "- Try a broader search with fewer filters"
            ),
            "sources": [],
            "context": "",
            "winning_source": "none"
        }

    # Build a structured document listing
    lang_names = {
        "en": "English", "fr": "French", "ur": "Urdu",
        "ar": "Arabic", "es": "Spanish"
    }
    resp_lang = lang_names.get(query_lang, "English")

    # Group by document_type for organized output
    doc_list_parts = []
    for i, r in enumerate(search_results[:30], 1):
        doc_type = r.get("document_type", "Unknown")
        subtype = r.get("document_subtype", "")
        file_name = r.get("file_name", "unknown")
        persons = r.get("persons", [])
        summary = r.get("summary", "")
        source_type = "🔴 Internal" if r.get("source_container") == "legal-documents-internal" else "📗 External"

        entry = f"{i}. **{file_name}**\n"
        entry += f"   Type: {doc_type}"
        if subtype:
            entry += f" — {subtype}"
        entry += f" | {source_type}\n"
        if persons:
            entry += f"   Persons: {', '.join(persons[:5])}\n"
        if summary:
            entry += f"   Summary: {summary[:150]}...\n" if len(summary) > 150 else f"   Summary: {summary}\n"

        doc_list_parts.append(entry)

    total = len(search_results)
    shown = min(total, 30)
    filter_desc = ", ".join(f"**{k}**: {v}" for k, v in filters.items()) if filters else "no specific filters"

    answer = f"Found **{total}** documents matching your criteria ({filter_desc}).\n\n"
    if shown < total:
        answer += f"Showing top {shown} results:\n\n"
    answer += "\n".join(doc_list_parts)

    return {
        "answer": answer,
        "sources": search_results[:30],
        "context": "",
        "winning_source": winning_source
    }


def retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Multi-query retrieval: run all query variants against the index,
    merge + deduplicate results, then let the cross-encoder re-ranker
    (inside hybrid_search_isolated) pick the best chunks.
    """
    query_variants = state.get("query_variants", [])
    fallback_query = state.get("rewritten_query") or state.get("query", "")

    if not query_variants:
        query_variants = [fallback_query]

    source_mode = state.get("source_mode", "all")
    allowed_files = state.get("allowed_files", [])

    all_results = []
    winning_sources = []

    # Build OData filter from discovery_filters for ANSWER mode too
    discovery_filters = state.get("discovery_filters", {})
    answer_odata_filter = _build_odata_filter(discovery_filters) if discovery_filters else None
    if answer_odata_filter:
        logger.info(f"ANSWER mode using metadata filter: {answer_odata_filter}")

    for i, variant in enumerate(query_variants):
        logger.info(
            f"Multi-query retrieval variant {i+1}/{len(query_variants)}: {variant[:60]}..."
        )
        variant_embedding = generate_embedding(azure_clients, variant)

        if answer_odata_filter:
            # Search WITH metadata filter (high precision)
            filtered_results, filtered_source = hybrid_search_isolated(
                azure_clients, variant, variant_embedding,
                top_k=20, source_mode=source_mode,
                allowed_files=allowed_files,
                odata_filter=answer_odata_filter
            )
            all_results.extend(filtered_results)
            winning_sources.append(filtered_source)

            # Only do unfiltered search on FIRST variant if filtered gave < 3 results
            if i == 0 and len(filtered_results) < 3:
                logger.info(f"Filtered search gave only {len(filtered_results)} results — adding unfiltered search")
                results, winning_source = hybrid_search_isolated(
                    azure_clients, variant, variant_embedding,
                    top_k=20, source_mode=source_mode,
                    allowed_files=allowed_files
                )
                all_results.extend(results)
                winning_sources.append(winning_source)
        else:
            # No filter available — normal unfiltered search
            results, winning_source = hybrid_search_isolated(
                azure_clients, variant, variant_embedding,
                top_k=20, source_mode=source_mode,
                allowed_files=allowed_files
            )
            all_results.extend(results)
            winning_sources.append(winning_source)

    # Deduplicate by blob_path + chunk_index, keeping the highest-scored version
    best_by_key = {}
    for r in all_results:
        key = (r.get("blob_path", ""), r.get("chunk_index", 0))
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = r
        else:
            r_score = r.get("cross_encoder_score", r.get("score", 0))
            e_score = existing.get("cross_encoder_score", existing.get("score", 0))
            if r_score > e_score:
                best_by_key[key] = r
    deduped = list(best_by_key.values())

    # Sort by cross-encoder score and take top results
    deduped.sort(
        key=lambda r: r.get("cross_encoder_score", r.get("score", 0)),
        reverse=True
    )
    search_results = deduped[:20]

    # Determine winning source from the best results
    source_counts = Counter(s for s in winning_sources if s != "none")
    winning_source = source_counts.most_common(1)[0][0] if source_counts else "none"

    logger.info(
        f"Multi-query retrieval: {len(query_variants)} variants → "
        f"{len(all_results)} raw → {len(deduped)} deduped → "
        f"{len(search_results)} final | source: {winning_source}"
    )

    return {
        "query": state.get("query", ""),
        "rewritten_query": fallback_query,
        "query_variants": query_variants,
        "query_embedding": generate_embedding(azure_clients, fallback_query),
        "search_results": search_results,
        "winning_source": winning_source,
        "conversation_history": state.get("conversation_history", []),
        "source_mode": source_mode,
        "allowed_files": allowed_files,
        "lookup_mode": state.get("lookup_mode", "answer"),
        "discovery_filters": state.get("discovery_filters", {}),
    }


MAX_RETRIEVAL_ATTEMPTS = 3


def decompose_query_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Check if the query needs to be decomposed into multiple sub-queries.
    For example: "compare contract A and contract B" → two separate retrievals.
    """
    query = state.get("query", "")

    prompt = prompts.get_decompose_query_prompt(query)
    response = azure_clients.llm.invoke([("human", prompt)])
    result = response.content.strip()

    sub_queries = []
    if result != "SINGLE":
        for line in result.splitlines():
            line = line.strip()
            if line.startswith("SUB:"):
                sub_queries.append(line[4:].strip())

    if len(sub_queries) < 2:
        sub_queries = []

    if sub_queries:
        logger.info(f"Query decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
    else:
        logger.info("Query does not need decomposition — proceeding as single query")

    return {
        **state,
        "sub_queries": sub_queries,
        "retrieval_attempts": 0,
    }


def should_decompose(state: RAGState) -> str:
    """Routing function: if sub-queries exist, go to multi-retrieve; otherwise single retrieve."""
    sub_queries = state.get("sub_queries", [])
    if len(sub_queries) >= 2:
        return "multi_retrieve"
    return "retrieve"


def multi_retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Handle decomposed queries: run a separate retrieval for each sub-query,
    then merge all results.
    """
    sub_queries = state.get("sub_queries", [])
    source_mode = state.get("source_mode", "all")
    allowed_files = state.get("allowed_files", [])

    all_results = []
    winning_sources = []

    for i, sq in enumerate(sub_queries):
        logger.info(f"Multi-retrieve sub-query {i+1}/{len(sub_queries)}: {sq[:60]}...")

        # Rewrite each sub-query to French keywords
        history_text = ""
        sq_lower = sq.lower()
        query_type = "general legal"
        if any(w in sq_lower for w in ["income", "revenue", "profit", "financial"]):
            query_type = "financial"
        elif any(w in sq_lower for w in ["email", "courriel", "sender", "recipient"]):
            query_type = "email"

        rewrite_prompt = prompts.get_rewrite_query_prompt(history_text, query_type, sq)
        rewrite_resp = azure_clients.llm.invoke([("human", rewrite_prompt)])
        rewritten_sq = rewrite_resp.content.strip()
        logger.info(f"Sub-query {i+1} rewritten: '{sq[:40]}' → '{rewritten_sq[:60]}'")

        sq_embedding = generate_embedding(azure_clients, rewritten_sq)
        results, winning_source = hybrid_search_isolated(
            azure_clients, rewritten_sq, sq_embedding,
            top_k=20, source_mode=source_mode,
            allowed_files=allowed_files
        )

        # Tag each result with its sub-query for context
        for r in results:
            r["sub_query"] = sq

        all_results.extend(results)
        winning_sources.append(winning_source)

    # Deduplicate by blob_path + chunk_index
    seen = set()
    deduped = []
    for r in all_results:
        key = (r.get("blob_path", ""), r.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Pick the most common winning source
    source_counts = Counter(s for s in winning_sources if s != "none")
    winning = source_counts.most_common(1)[0][0] if source_counts else "none"

    logger.info(f"Multi-retrieve: {len(deduped)} unique chunks from {len(sub_queries)} sub-queries")

    return {
        **state,
        "search_results": deduped,
        "winning_source": winning,
        "retrieval_attempts": 1,
        "retrieval_sufficient": True,
    }


def evaluate_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Self-evaluation node: check retrieval quality using both score-based
    pre-checks AND LLM judgment. If insufficient, trigger re-retrieval.
    """
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    attempts = state.get("retrieval_attempts", 0) + 1

    # Gate 1: No results at all
    if not search_results:
        logger.info(f"Evaluate: no results found (attempt {attempts})")
        return {
            **state,
            "retrieval_attempts": attempts,
            "retrieval_sufficient": attempts >= MAX_RETRIEVAL_ATTEMPTS,
        }

    # Gate 2: Score-based pre-check — don't waste an LLM call on garbage
    best_cross_score = max(
        (r.get("cross_encoder_score", 0) for r in search_results), default=0
    )
    ce_threshold = azure_clients.config.cross_encoder_min_score

    if best_cross_score < ce_threshold:
        logger.info(
            f"Evaluate: best cross-encoder score {best_cross_score:.4f} "
            f"below threshold {ce_threshold} — marking insufficient (attempt {attempts})"
        )
        return {
            **state,
            "retrieval_attempts": attempts,
            "retrieval_sufficient": attempts >= MAX_RETRIEVAL_ATTEMPTS,
        }

    # Gate 3: LLM-based evaluation for borderline cases
    chunks_summary = "\n".join(
        f"[Chunk {i+1}] File: {r.get('file_name', 'unknown')} | "
        f"Score: {r.get('cross_encoder_score', r.get('score', 0)):.4f}\n"
        f"{r.get('content', '')[:300]}..."
        for i, r in enumerate(search_results[:5])
    )

    prompt = prompts.get_evaluate_retrieval_prompt(query, chunks_summary)
    response = azure_clients.llm.invoke([("human", prompt)])
    evaluation = response.content.strip().upper()

    is_sufficient = "SUFFICIENT" in evaluation
    logger.info(f"Evaluate (attempt {attempts}): {evaluation} | sufficient={is_sufficient}")

    if not is_sufficient and attempts >= MAX_RETRIEVAL_ATTEMPTS:
        logger.warning(
            f"Max retrieval attempts ({MAX_RETRIEVAL_ATTEMPTS}) reached — "
            f"proceeding with best results"
        )
        is_sufficient = True

    return {
        **state,
        "retrieval_attempts": attempts,
        "retrieval_sufficient": is_sufficient,
    }


def should_retry_retrieval(state: RAGState) -> str:
    """Routing function: retry retrieval or proceed to generate."""
    if state.get("retrieval_sufficient", False):
        return "generate"
    return "retry_rewrite"


def retry_rewrite_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Rewrite the query with a different strategy when previous retrieval was insufficient.
    Generates 3 new variants using a completely different approach.
    """
    query = state.get("query", "")
    previous_variants = state.get("query_variants", [])
    attempts = state.get("retrieval_attempts", 1)

    previous_variants_text = "\n".join(f"- {v}" for v in previous_variants)
    prompt = prompts.get_multi_query_retry_prompt(query, previous_variants_text, attempts)
    response = azure_clients.llm.invoke([("human", prompt)])
    new_variants = _parse_query_variants(response.content)

    # Fallback to single rewrite if parsing fails
    if len(new_variants) < 2:
        logger.warning("Retry multi-query parsing failed — falling back to single rewrite")
        fallback_prompt = prompts.get_rewrite_retry_prompt(
            query, previous_variants[0] if previous_variants else "", attempts
        )
        fallback_resp = azure_clients.llm.invoke([("human", fallback_prompt)])
        new_variants = [fallback_resp.content.strip()]

    logger.info(
        f"Retry rewrite (attempt {attempts}): {len(new_variants)} new variants: "
        f"{[v[:50] for v in new_variants]}"
    )

    return {
        **state,
        "rewritten_query": new_variants[0],
        "query_variants": new_variants,
    }


# ============================================================================
# NOT-FOUND RESPONSE (shared across generate_node gates)
# ============================================================================

NOT_FOUND_RESPONSE = {
    "answer": (
        "I was unable to find relevant information in the available legal documents "
        "to answer this question accurately.\n\n"
        "This may mean:\n"
        "• The relevant document has not been indexed yet\n"
        "• The question refers to information not present in the loaded documents\n\n"
        "Please consult a qualified legal professional for authoritative guidance."
    ),
    "sources": [],
    "context": "",
    "winning_source": "none"
}


def generate_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Generate a structured, professional legal answer from a single trusted source.

    Guards:
    1. No results → return not-found
    2. All results below cross-encoder quality threshold → return not-found
    3. Lookup mode "document" → return file links instead of generated answer
    4. Normal mode → generate answer with GPT-4 + confidence signal
    """
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    history = state.get("conversation_history", [])
    winning_source = state.get("winning_source", "unknown")
    lookup_mode = state.get("lookup_mode", "answer")

    # --- Guard 1: No results at all ---
    if not search_results:
        logger.info("No results above confidence threshold — returning not-found response")
        return dict(NOT_FOUND_RESPONSE)

    # --- Guard 2: Filter out low-quality results before sending to GPT-4 ---
    ce_threshold = azure_clients.config.cross_encoder_min_score
    quality_results = [
        r for r in search_results
        if r.get("cross_encoder_score", r.get("score", 0)) >= ce_threshold
    ]

    if not quality_results:
        logger.info(
            f"All {len(search_results)} results below quality threshold "
            f"({ce_threshold}) — returning not-found response"
        )
        return dict(NOT_FOUND_RESPONSE)

    # Use only quality results from here on
    search_results = quality_results

    # --- Guard 3: Document lookup mode ---
    if lookup_mode == "document":
        seen = set()
        unique_sources = []

        for s in search_results:
            fname = s.get("file_name", "")
            lower_name = fname.lower()

            # Skip interrogatoire transcripts
            if "interrogatoire" in lower_name:
                continue

            key = (
                s.get("source_container", ""),
                s.get("blob_path", ""),
                fname
            )
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        if not unique_sources:
            return {
                "answer": (
                    "I could not find original source PDF files matching your request. "
                    "The available results appear to be transcript/reference documents "
                    "rather than the original source files."
                ),
                "sources": [],
                "context": "",
                "winning_source": "none"
            }

        return {
            "answer": "I found the most relevant source documents matching your request.",
            "sources": unique_sources[:10],
            "context": "",
            "winning_source": winning_source
        }

    # --- Normal mode: Generate answer with GPT-4 ---
    # Step 1: Select top unique files FIRST — these become the grounded source set.
    #         GPT will ONLY see chunks from these files, so every citation is displayable.
    #         Sort by cross-encoder score first to ensure best files win regardless of variant order.
    sorted_results = sorted(
        search_results,
        key=lambda r: r.get("cross_encoder_score", r.get("score", 0)),
        reverse=True
    )
    seen_files = set()
    seen_basenames = {}  # basename -> count, to limit near-duplicate files
    grounded_sources = []
    for r in sorted_results:
        key = (r.get("source_container", ""), r.get("blob_path", ""), r.get("file_name", ""))
        if key in seen_files:
            continue
        # Limit files with very similar basenames (e.g. 4 interrogatoire variants)
        # Allow max 2 files sharing the same first 30 chars of filename
        fname = r.get("file_name", "")
        base_prefix = fname[:30] if fname else ""
        prefix_count = seen_basenames.get(base_prefix, 0)
        if prefix_count >= 2:
            continue
        seen_files.add(key)
        seen_basenames[base_prefix] = prefix_count + 1
        grounded_sources.append(r)

    # Step 2: Build context ONLY from chunks belonging to grounded source files
    grounded_file_keys = set(
        (r.get("source_container", ""), r.get("blob_path", ""), r.get("file_name", ""))
        for r in grounded_sources
    )
    grounded_chunks = [
        r for r in search_results
        if (r.get("source_container", ""), r.get("blob_path", ""), r.get("file_name", ""))
        in grounded_file_keys
    ]

    context_parts = []
    for i, result in enumerate(grounded_chunks, 1):
        source_type = (
            "🔴 Internal (Confidential)"
            if result.get("source_container") == "legal-documents-internal"
            else "📗 External"
        )
        context_parts.append(
            f"[Source {i}] {source_type} | File: {result['file_name']}\n"
            f"{result['content'][:3000]}\n"
        )
    context = "\n---\n".join(context_parts)

    # Step 3: Build messages for GPT
    messages = []

    query_lang = state.get("query_lang", "en")
    lang_names = {
        "en": "English", "fr": "French", "ur": "Urdu",
        "ar": "Arabic", "es": "Spanish", "de": "German", "it": "Italian"
    }
    response_language = lang_names.get(query_lang, "English")

    system_prompt = prompts.get_system_prompt(response_language, context)
    messages.append(("system", system_prompt))

    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(("human", content))
            else:
                messages.append(("ai", content))

    messages.append(("human", query))

    response = azure_clients.llm.invoke(messages)
    answer = response.content

    # Extract and strip confidence signal before sending to user
    confidence = "CONFIDENT"
    for tag in ["[NOT_FOUND]", "[PARTIAL]", "[CONFIDENT]"]:
        if tag in answer:
            confidence = tag.strip("[]")
            answer = answer.replace(tag, "").strip()
            break

    logger.info(f"LLM confidence signal: {confidence}")
    logger.info(f"Generated answer ({len(answer)} chars) | grounded on {len(grounded_sources)} files")

    return {
        "answer": answer,
        "sources": grounded_sources,
        "context": context,
        "winning_source": winning_source
    }


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def should_route_intent(state: RAGState) -> str:
    """Route based on query intent: discovery queries skip decompose and go straight to filtered search."""
    if state.get("query_intent") == "discovery":
        return "discovery_retrieve"
    return "decompose_query"


def build_rag_graph(azure_clients: AzureClients):
    """
    Build the agentic RAG workflow graph.

    Flow:
      rewrite_query ─┬─ (DISCOVERY) → discovery_retrieve → discovery_generate → END
                      │
                      └─ (ANSWER) → decompose_query ─┬─ (SINGLE) → retrieve → evaluate ─┬─ (SUFFICIENT) → generate → END
                                                      │                                   │
                                                      │                                   └─ (INSUFFICIENT) → retry_rewrite → retrieve (loop up to 3x)
                                                      │
                                                      └─ (MULTI) → multi_retrieve → generate → END
    """
    workflow = StateGraph(RAGState)

    # Nodes
    workflow.add_node("rewrite_query", lambda state: rewrite_query_node(state, azure_clients))
    workflow.add_node("discovery_retrieve", lambda state: discovery_retrieve_node(state, azure_clients))
    workflow.add_node("discovery_generate", lambda state: discovery_generate_node(state, azure_clients))
    workflow.add_node("decompose_query", lambda state: decompose_query_node(state, azure_clients))
    workflow.add_node("retrieve", lambda state: retrieve_node(state, azure_clients))
    workflow.add_node("multi_retrieve", lambda state: multi_retrieve_node(state, azure_clients))
    workflow.add_node("evaluate", lambda state: evaluate_node(state, azure_clients))
    workflow.add_node("retry_rewrite", lambda state: retry_rewrite_node(state, azure_clients))
    workflow.add_node("generate", lambda state: generate_node(state, azure_clients))

    # Edges — intent routing after rewrite
    workflow.add_conditional_edges("rewrite_query", should_route_intent, {
        "discovery_retrieve": "discovery_retrieve",
        "decompose_query": "decompose_query",
    })

    # Discovery path
    workflow.add_edge("discovery_retrieve", "discovery_generate")
    workflow.add_edge("discovery_generate", END)

    # Answer path (existing)
    workflow.add_conditional_edges("decompose_query", should_decompose, {
        "retrieve": "retrieve",
        "multi_retrieve": "multi_retrieve",
    })

    workflow.add_edge("retrieve", "evaluate")
    workflow.add_edge("multi_retrieve", "generate")

    workflow.add_conditional_edges("evaluate", should_retry_retrieval, {
        "generate": "generate",
        "retry_rewrite": "retry_rewrite",
    })

    workflow.add_edge("retry_rewrite", "retrieve")
    workflow.add_edge("generate", END)

    workflow.set_entry_point("rewrite_query")

    return workflow.compile()


# ============================================================================
# FASTAPI APP
# ============================================================================

azure_clients: Optional[AzureClients] = None
rag_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global azure_clients, rag_graph

    logger.info("Initializing Azure clients...")
    config = Config.from_env()
    azure_clients = AzureClients(config)
    rag_graph = build_rag_graph(azure_clients)
    logger.info("RAG Backend ready!")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Legal AI RAG Backend",
    description="AI-powered legal document search and Q&A — Improved",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "legal-ai-rag", "version": "2.0.0"}


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search legal documents using isolated dual-index search"""
    try:
        query_embedding = generate_embedding(azure_clients, request.query)
        results, _ = hybrid_search_isolated(
            azure_clients, request.query, query_embedding, top_k=request.top_k
        )
        return [SearchResult(**r) for r in results]
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat with the legal AI assistant.

    FLOW:
    1. Detect query language → respond in same language
    2. Rewrite query to French keywords for retrieval
    3. Decompose if multi-entity question
    4. Hybrid search with cross-encoder re-ranking
    5. Evaluate retrieval quality (retry up to 3x if insufficient)
    6. Generate structured answer
    7. If email requested → send answer via ACS in background
    """
    try:
        detected_lang = detect_language(request.query, azure_clients.llm)
        logger.info(f"Detected language: {detected_lang}")

        target_lang = request.target_language
        if target_lang == "auto":
            target_lang = detected_lang

        history_dicts = [msg.model_dump() for msg in (request.history or [])]

        state: RAGState = {
            "query": request.query,
            "query_lang": detected_lang,
            "conversation_history": history_dicts,
            "source_mode": request.source_mode or "all",
            "allowed_files": request.allowed_files or []
        }
        final_state = rag_graph.invoke(state)

        answer = final_state.get("answer", "")
        sources = final_state.get("sources", [])

        # Translate if needed
        if target_lang == "fr" and detected_lang == "en":
            answer = translate_text(azure_clients.llm, answer, "en", "fr")
        elif target_lang == "en" and detected_lang == "fr":
            answer = translate_text(azure_clients.llm, answer, "fr", "en")

        # Check for email intent
        recipient_email, wants_email = extract_email_intent(request.query)
        if wants_email:
            background_tasks.add_task(
                send_email_acs,
                azure_clients.config,
                recipient_email,
                "Legal Assistant — Document Information",
                answer,
            )
            answer += f"\n\n📧 Answer is being sent to **{recipient_email}**."

        return ChatResponse(
            answer=answer,
            sources=[SearchResult(**s) for s in sources],
            conversation_id=request.conversation_id or "new",
            detected_query_language=detected_lang,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(payload: Dict[Any, Any], background_tasks: BackgroundTasks):
    """Webhook endpoint for WhatsApp (GreenAPI)"""
    logger.info(f"Received WhatsApp webhook: {payload}")
    return {"status": "received"}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)