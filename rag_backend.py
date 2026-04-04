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
from dataclasses import dataclass, field
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

       
                                                           
    ext_best = max((r["score"] for r in external_results), default=0.0)
    int_best = max((r["score"] for r in internal_results), default=0.0)

    logger.info(f"Search scores — External: {ext_best:.4f} | Internal: {int_best:.4f} | Threshold: {MIN_SCORE}")

                                                            
    if ext_best < MIN_SCORE and int_best < MIN_SCORE:
        logger.warning("Both sources below confidence threshold — returning no results")
        return [], "none"


                                                                                                           
                                           
    filtered_internal = sorted(
        [r for r in internal_results if r["score"] >= MIN_SCORE],
        key=lambda r: r["score"], reverse=True
    )
    filtered_external = sorted(
        [r for r in external_results if r["score"] >= MIN_SCORE],
        key=lambda r: r["score"], reverse=True
    )

                                    
    if int_best >= ext_best and filtered_internal:
        candidate_results = filtered_internal
        winning_source = "legal-documents-internal"
    elif filtered_external:
        candidate_results = filtered_external
        winning_source = "legal-documents"
    else:
        return [], "none"
                                                                              

    winning = sorted(candidate_results, key=lambda x: x["score"], reverse=True)[:10]

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

                                                                              
                 
                                                                              

def rewrite_query_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """
    NEW NODE: Rewrite user query into a better retrieval query.
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
                                                 

    prompt = prompts.get_rewrite_query_prompt(history_text, query_type, query)

    response = azure_clients.llm.invoke([("human", prompt)])
    rewritten = response.content.strip()
    logger.info(f"Query rewritten: '{query[:40]}' → '{rewritten[:60]}'")

    return {
        "query": query,
        "query_lang": query_lang,
        "rewritten_query": rewritten,
        "conversation_history": history,
        "lookup_mode": lookup_mode,                                    
        "source_mode": state.get("source_mode", "all"),                
        "allowed_files": state.get("allowed_files", [])                
    }


def retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """Retrieve relevant documents using isolated dual-index search"""
    search_query = state.get("rewritten_query") or state.get("query", "")
    logger.info(f"Retrieving for: {search_query[:60]}...")

    query_embedding = generate_embedding(azure_clients, search_query)

    source_mode = state.get("source_mode", "all")             
    allowed_files = state.get("allowed_files", [])            


                                                                              
    search_results, winning_source = hybrid_search_isolated(
        azure_clients,
        search_query, 
        query_embedding, 
        top_k=15,
        source_mode=source_mode,            
        allowed_files=allowed_files            
    )

    logger.info(f"Found {len(search_results)} chunks from: {winning_source}")
    return {
        "query": state.get("query", ""),
        "rewritten_query": search_query,
        "query_embedding": query_embedding,
        "search_results": search_results,
        "winning_source": winning_source,
        "conversation_history": state.get("conversation_history", []),
        "source_mode": source_mode,            
        "allowed_files": allowed_files,            
        "lookup_mode": state.get("lookup_mode", "answer")
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
    """Build the RAG workflow graph with improved nodes"""

    workflow = StateGraph(RAGState)

           
    workflow.add_node("rewrite_query", lambda state: rewrite_query_node(state, azure_clients))
    workflow.add_node("retrieve", lambda state: retrieve_node(state, azure_clients))
    workflow.add_node("generate", lambda state: generate_node(state, azure_clients))

                                               
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
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
