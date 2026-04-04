import prompts
"""
RAG Backend API - Improved Version
FastAPI server with LangGraph workflow for legal document search and Q&A

Key Improvements:
- Professional, structured system prompt
- Bilingual search (queries both EN + FR indexes simultaneously)
- Larger context window per document chunk (800 → 1500 chars)
- Lower temperature (0.3 → 0.1) for factual legal accuracy
- Conversation history support
- WhatsApp-friendly response formatting
- Query rewriting node for better retrieval
- Confidence scoring and "no answer" handling
"""

from cgitb import lookup
import os
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
_file_handler = TimedRotatingFileHandler("logs/rag_backend.log", when="midnight", backupCount=30, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

load_dotenv()


                                                                              
               
                                                                              

@dataclass
class Config:
    search_endpoint: str
    search_key: str
    search_index: str
                                                                                 
    search_index_external: str
    search_index_internal: str
                                                                                             
    min_search_score: float
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
            min_search_score=float(os.environ.get("MIN_SEARCH_SCORE", "0.02")),
            openai_endpoint=os.environ["OPENAI_ENDPOINT"],
            openai_key=os.environ["OPENAI_KEY"],
            openai_chat_deployment=os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat"),
            openai_embedding_deployment=os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT", "Embeddings"),
            container_sas_url=os.environ["CONTAINER_SAS_URL"],
            internal_container_sas_url=os.environ.get("INTERNAL_CONTAINER_SAS_URL", ""),
            acs_connection_string=os.environ.get("ACS_CONNECTION_STRING", ""),
            acs_sender_email=os.environ.get("ACS_SENDER_EMAIL", ""),
        )


                                                                              
                 
                                                                              

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
    role: str                          
    content: str


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    target_language: Optional[str] = "auto"
    history: Optional[List[ChatMessage]] = []                              
    source_mode: Optional[str] = "all"                                                    
    allowed_files: Optional[List[str]] = []            

class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    conversation_id: str
    detected_query_language: Optional[str] = None


                                                                              
                                  
                                                                              

def detect_language(text: str, llm: AzureChatOpenAI) -> str:
    """
    Detect the language of the user's query.
    Returns a language code: 'en', 'fr', 'ur', 'ar', 'es', etc.
    Used to respond in the same language the user wrote in.
    """
                                                                         
    urdu_arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    if urdu_arabic_chars > 2:
                                                                         
        urdu_specific = sum(1 for c in text if c in 'ے،ہھگڈڑ')
        return "ur" if urdu_specific > 0 else "ar"

                                                       
    french_indicators = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'dans',
                         'pour', 'avec', 'qui', 'que', 'du', 'au', 'en', 'il', 'elle']
    text_lower = text.lower()
    french_word_count = sum(1 for word in french_indicators if f" {word} " in f" {text_lower} ")
    if french_word_count >= 2:
        return "fr"

                                      
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


                                                                              
               
                                                                              

class AzureClients:
    def __init__(self, config: Config):
        self.config = config

                                                         
                                                                      
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

        blob_file_path = unicodedata.normalize("NFC", blob_file_path)
        encoded_path = quote(blob_file_path, safe="/")
        url = f"{base_url}/{container_name}/{encoded_path}?{sas_token}"
        logger.info(f"SAS URL generated | container={source_container} | path={blob_file_path}")
        return url

    except Exception as e:
        logger.warning(f"Failed to generate URL for {blob_path}: {e}")
        return None

def _search_one_index(
    search_client: SearchClient,
    query: str,
    query_embedding: List[float],
    top_k: int,
    allowed_files: Optional[List[str]] = None            
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
    query_type="semantic",
    semantic_configuration_name="semantic-config",
    select=["content", "file_name", "folder_path", "blob_path", "chunk_index", "source_container"],
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
    allowed_files: Optional[List[str]] = None            
) -> tuple:
    """
    FIX: Search BOTH indices separately, pick the winner by highest score.

    CRITICAL BEHAVIOUR:
    - External index searched independently
    - Internal index searched independently
    - GPT-4 receives context from ONE winning source only — never a mix
    - If both sources score below min_search_score → return ([], "none")
      → no GPT-4 call → no hallucination possible

    Returns: (results: List[Dict], winning_source: str)
    """
    MIN_SCORE = azure_clients.config.min_search_score
    if allowed_files:             
        MIN_SCORE = 0.0  
                                           
                                                                             
       
                                           
                                                                             
                                     



                                                           
    allowed_files = set(allowed_files or [])

    external_results = []
    internal_results = []

    if source_mode in ("all", "external_only"):
        external_results = _search_one_index(
            azure_clients.search_client_external, 
            query, 
            query_embedding, 
            top_k,
            allowed_files=list(allowed_files) if allowed_files else None            
        )

    if source_mode in ("all", "internal_only"):
        internal_results = _search_one_index(
            azure_clients.search_client_internal, 
            query, 
            query_embedding, 
            top_k,
            allowed_files=list(allowed_files) if allowed_files else None            
        )

    # --- Cross-encoder re-ranking ---
    # Re-rank each index's results independently using the cross-encoder.
    # This jointly scores each (query, chunk) pair for much better precision
    # than the independent BM25/vector scores from Azure.
    RERANK_TOP_N = 5
    if external_results:
        logger.info(f"Re-ranking {len(external_results)} external results with cross-encoder...")
        external_results = rerank_results(azure_clients.cross_encoder, query, external_results, top_n=RERANK_TOP_N)
    if internal_results:
        logger.info(f"Re-ranking {len(internal_results)} internal results with cross-encoder...")
        internal_results = rerank_results(azure_clients.cross_encoder, query, internal_results, top_n=RERANK_TOP_N)

    # Use cross-encoder scores for comparison (fall back to original score)
    def _best_score(results):
        return max((r.get("cross_encoder_score", r["score"]) for r in results), default=0.0)

    ext_best = _best_score(external_results)
    int_best = _best_score(internal_results)

    logger.info(f"Re-ranked scores — External best: {ext_best:.4f} | Internal best: {int_best:.4f}")

    # Use original Azure scores for the MIN_SCORE threshold check
    # (cross-encoder scores are on a different scale)
    ext_azure_best = max((r["score"] for r in external_results), default=0.0)
    int_azure_best = max((r["score"] for r in internal_results), default=0.0)

    logger.info(f"Azure scores — External: {ext_azure_best:.4f} | Internal: {int_azure_best:.4f} | Threshold: {MIN_SCORE}")

    if ext_azure_best < MIN_SCORE and int_azure_best < MIN_SCORE:
        logger.warning("Both sources below confidence threshold — returning no results")
        return [], "none"

    # Filter by MIN_SCORE (on original Azure scores), then sort by cross-encoder score
    filtered_internal = sorted(
        [r for r in internal_results if r["score"] >= MIN_SCORE],
        key=lambda r: r.get("cross_encoder_score", r["score"]), reverse=True
    )
    filtered_external = sorted(
        [r for r in external_results if r["score"] >= MIN_SCORE],
        key=lambda r: r.get("cross_encoder_score", r["score"]), reverse=True
    )

    # Pick winning source by best cross-encoder score
    if int_best >= ext_best and filtered_internal:
        candidate_results = filtered_internal
        winning_source = "legal-documents-internal"
    elif filtered_external:
        candidate_results = filtered_external
        winning_source = "legal-documents"
    else:
        return [], "none"

    winning = candidate_results[:RERANK_TOP_N]

    for r in winning:
        r["source_url"] = generate_sas_url(
            azure_clients,
            r["blob_path"],
            r.get("source_container", ""),
            r.get("folder_path", ""),
            r.get("file_name", "")
        )


    logger.info(
    f"Multi-doc mode: {len(winning)} chunks | "
    f"Source: {winning_source} | "
    f"Best score: {winning[0]['score']:.4f}"
    )

    return winning, winning_source

                                                                              
                 
                                                                              

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
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in last_turns])
        history_text = f"\nRecent conversation:\n{history_text}\n"

    ql = query.lower()

    query_type = "general legal"
    if any(word in ql for word in ["income", "revenue", "profit", "financial", "creditors", "bilan", "benefit", "net income"]):
        query_type = "financial"
    elif any(word in ql for word in ["email", "courriel", "sender", "recipient", "subject", "message", "sent"]):
        query_type = "email"
    elif any(word in ql for word in ["invoice", "register", "supplier", "mutation", "project", "facture"]):
        query_type = "invoice/register"

    lookup_mode = "document" if any(
        term in ql for term in [
            "pdf", "pdfs", "pièce", "piece", "document original", "documents originaux",
            "fichier", "fichiers", "source", "sources", "lien", "liens",
            "trouve les pdf", "retourne les fichiers", "original file", "original files",
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

    return {
        "query": query,
        "query_lang": query_lang,
        "rewritten_query": variants[0],  # Primary variant for backward compat
        "query_variants": variants,
        "conversation_history": history,
        "lookup_mode": lookup_mode,
        "source_mode": state.get("source_mode", "all"),
        "allowed_files": state.get("allowed_files", [])
    }


def retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Multi-query retrieval: run all query variants against the index,
    merge + deduplicate results, then let the cross-encoder re-ranker
    (inside hybrid_search_isolated) pick the best chunks.
    """
    query_variants = state.get("query_variants", [])
    fallback_query = state.get("rewritten_query") or state.get("query", "")

    # If no variants, use the single rewritten query
    if not query_variants:
        query_variants = [fallback_query]

    source_mode = state.get("source_mode", "all")
    allowed_files = state.get("allowed_files", [])

    all_results = []
    winning_sources = []

    for i, variant in enumerate(query_variants):
        logger.info(f"Multi-query retrieval variant {i+1}/{len(query_variants)}: {variant[:60]}...")
        variant_embedding = generate_embedding(azure_clients, variant)
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
            # Keep the one with the higher cross-encoder score (or original score)
            r_score = r.get("cross_encoder_score", r.get("score", 0))
            e_score = existing.get("cross_encoder_score", existing.get("score", 0))
            if r_score > e_score:
                best_by_key[key] = r
    deduped = list(best_by_key.values())

    # Sort by cross-encoder score and take top results
    deduped.sort(key=lambda r: r.get("cross_encoder_score", r.get("score", 0)), reverse=True)
    search_results = deduped[:10]

    # Determine winning source from the best results
    source_counts = Counter(s for s in winning_sources if s != "none")
    winning_source = source_counts.most_common(1)[0][0] if source_counts else "none"

    logger.info(
        f"Multi-query retrieval: {len(query_variants)} variants → "
        f"{len(all_results)} raw → {len(deduped)} deduped → {len(search_results)} final | "
        f"source: {winning_source}"
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
        "lookup_mode": state.get("lookup_mode", "answer")
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
        "retrieval_sufficient": True,  # Skip evaluation for decomposed queries
    }


def evaluate_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Self-evaluation node: ask the LLM whether the retrieved chunks
    are sufficient to answer the query. If not, trigger a re-retrieval
    with a different query rewrite.
    """
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    attempts = state.get("retrieval_attempts", 0) + 1

    # If no results at all, mark as insufficient (unless max attempts reached)
    if not search_results:
        logger.info(f"Evaluate: no results found (attempt {attempts})")
        return {
            **state,
            "retrieval_attempts": attempts,
            "retrieval_sufficient": attempts >= MAX_RETRIEVAL_ATTEMPTS,
        }

    # Build a summary of retrieved chunks for the LLM to evaluate
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
        logger.warning(f"Max retrieval attempts ({MAX_RETRIEVAL_ATTEMPTS}) reached — proceeding with best results")
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

    logger.info(f"Retry rewrite (attempt {attempts}): {len(new_variants)} new variants: {[v[:50] for v in new_variants]}")

    return {
        **state,
        "rewritten_query": new_variants[0],
        "query_variants": new_variants,
    }


def generate_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    Generate a structured, professional legal answer from a single trusted source.
    FIX: zero-results guard prevents GPT-4 from being called when nothing was found.
    FIX: source label now matches actual stored value "legal-documents-internal".
    """
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    history = state.get("conversation_history", [])
    winning_source = state.get("winning_source", "unknown")
    lookup_mode = state.get("lookup_mode", "answer")         
    

                                                                            
                                                                                     
    if not search_results:
        logger.info("No results above confidence threshold — returning not-found response")
        return {
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

    
    if lookup_mode == "document":
        seen = set()
        unique_sources = []

        for s in search_results:
            fname = s.get("file_name", "")
            lower_name = fname.lower()

                                                                                          
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
                    "The available results appear to be transcript/reference documents rather than the original source files."
                ),
                "sources": [],
                "context": ""
            }

        return {
            "answer": "I found the most relevant source documents matching your request.",
            "sources": unique_sources[:10],
            "context": ""
        }
    
                                                                                                                                                      
                        
                                                   
                                                                                        
    
                                                                        
                                                                                         
    context_parts = []
    for i, result in enumerate(search_results, 1):
        source_type = "🔴 Internal (Confidential)" if result.get("source_container") == "legal-documents-internal" else "📗 External"
        context_parts.append(
            f"[Source {i}] {source_type} | File: {result['file_name']}\n"
            f"{result['content'][:3000]}\n"
        )
    context = "\n---\n".join(context_parts)

                                            
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

    confidence = "CONFIDENT"  # default
    for tag in ["[NOT_FOUND]", "[PARTIAL]", "[CONFIDENT]"]:
        if tag in answer:
            confidence = tag.strip("[]")
            answer = answer.replace(tag, "").strip()
            break

    logger.info(f"LLM confidence signal: {confidence}")

    logger.info(f"Generated answer ({len(answer)} chars)")

    seen = set()
    unique_sources = []

    for s in search_results:
        key = (
            s.get("source_container", ""),
            s.get("blob_path", ""),
            s.get("file_name", "")
        )
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "answer": answer,
        "sources": unique_sources,
        "context": context
    }


                                                                              
                          
                                                                              

def build_rag_graph(azure_clients: AzureClients):
    """
    Build the agentic RAG workflow graph.

    Flow:
      rewrite_query → decompose_query ──┬── (SINGLE) ──→ retrieve ──→ evaluate ──┬── (SUFFICIENT) → generate → END
                                         │                                        │
                                         │                                        └── (INSUFFICIENT) → retry_rewrite → retrieve (loop up to 3x)
                                         │
                                         └── (MULTI) ───→ multi_retrieve ────────────→ generate → END
    """

    workflow = StateGraph(RAGState)

    # Nodes
    workflow.add_node("rewrite_query", lambda state: rewrite_query_node(state, azure_clients))
    workflow.add_node("decompose_query", lambda state: decompose_query_node(state, azure_clients))
    workflow.add_node("retrieve", lambda state: retrieve_node(state, azure_clients))
    workflow.add_node("multi_retrieve", lambda state: multi_retrieve_node(state, azure_clients))
    workflow.add_node("evaluate", lambda state: evaluate_node(state, azure_clients))
    workflow.add_node("retry_rewrite", lambda state: retry_rewrite_node(state, azure_clients))
    workflow.add_node("generate", lambda state: generate_node(state, azure_clients))

    # Edges
    workflow.add_edge("rewrite_query", "decompose_query")

    # Conditional: decompose decides single vs multi retrieval
    workflow.add_conditional_edges("decompose_query", should_decompose, {
        "retrieve": "retrieve",
        "multi_retrieve": "multi_retrieve",
    })

    # Single retrieval → evaluate
    workflow.add_edge("retrieve", "evaluate")

    # Multi retrieval → straight to generate (already combined results)
    workflow.add_edge("multi_retrieve", "generate")

    # Conditional: evaluate decides retry or generate
    workflow.add_conditional_edges("evaluate", should_retry_retrieval, {
        "generate": "generate",
        "retry_rewrite": "retry_rewrite",
    })

    # Retry rewrite loops back to retrieve
    workflow.add_edge("retry_rewrite", "retrieve")

    workflow.add_edge("generate", END)

    workflow.set_entry_point("rewrite_query")

    return workflow.compile()


                                                                              
             
                                                                              

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
                                                   
        results, _ = hybrid_search_isolated(azure_clients, request.query, query_embedding, top_k=request.top_k)
        return [SearchResult(**r) for r in results]
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat with the legal AI assistant.
    Supports multilingual queries and optional email delivery via ACS.

    FLOW:
    1. Check email intent — if user wants to email the answer somewhere
    2. Detect query language → respond in same language
    3. Rewrite query to French keywords for retrieval
    4. Hybrid search (internal-first)
    5. Generate structured answer
    6. If email requested → send answer via ACS in background
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

                                      
        if target_lang == "fr" and detected_lang == "en":
            answer = translate_text(azure_clients.llm, answer, "en", "fr")
        elif target_lang == "en" and detected_lang == "fr":
            answer = translate_text(azure_clients.llm, answer, "fr", "en")

                                                                       
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


                                                                              
      
                                                                              

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
  
  