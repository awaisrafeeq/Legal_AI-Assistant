"""
Sprint 3: RAG Backend API
FastAPI server with LangGraph workflow for legal document search and Q&A
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Azure imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from openai import AzureOpenAI
from datetime import datetime, timedelta

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    search_endpoint: str
    search_key: str
    search_index: str
    openai_endpoint: str
    openai_key: str
    openai_chat_deployment: str
    openai_embedding_deployment: str
    container_sas_url: str
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            search_endpoint=os.environ["SEARCH_ENDPOINT"],
            search_key=os.environ["SEARCH_KEY"],
            search_index=os.environ.get("SEARCH_INDEX", "legal-docs-index"),
            openai_endpoint=os.environ["OPENAI_ENDPOINT"],
            openai_key=os.environ["OPENAI_KEY"],
            openai_chat_deployment=os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat"),
            openai_embedding_deployment=os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT", "Embeddings"),
            container_sas_url=os.environ["CONTAINER_SAS_URL"],
        )


# ============================================================================
# Pydantic Models
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

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    target_language: Optional[str] = "auto"  # "en", "fr", or "auto" for auto-detect


class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    conversation_id: str
    detected_query_language: Optional[str] = None
    translated_answer: Optional[str] = None  # Original French answer if translated


# ============================================================================
# Translation Functions
# ============================================================================

def detect_language(text: str, llm: AzureChatOpenAI) -> str:
    """Detect if text is English or French"""
    # Simple heuristic first
    french_indicators = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'dans', 'pour', 'avec', 'qui', 'que']
    text_lower = text.lower()
    french_word_count = sum(1 for word in french_indicators if f" {word} " in f" {text_lower} ")
    
    # If many French words detected, likely French
    if french_word_count >= 2:
        return "fr"
    
    # Use LLM for ambiguous cases
    prompt = f"""Detect the language of this text. Respond with only 'en' for English or 'fr' for French.

Text: {text}

Language code:"""
    
    response = llm.invoke([("human", prompt)])
    result = response.content.strip().lower()
    return "fr" if "fr" in result else "en"


def translate_text(llm: AzureChatOpenAI, text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using LLM"""
    if source_lang == target_lang:
        return text
    
    lang_names = {"en": "English", "fr": "French"}
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    
    prompt = f"""Translate the following text from {source_name} to {target_name}.
Maintain legal terminology accuracy. Only return the translation, no explanations.

Text: {text}

Translation:"""
    
    response = llm.invoke([("human", prompt)])
    return response.content.strip()

class AzureClients:
    def __init__(self, config: Config):
        self.config = config
        
        # Search client
        self.search_client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.search_index,
            credential=AzureKeyCredential(config.search_key)
        )
        
        # Blob service client for SAS generation
        from urllib.parse import urlparse
        parsed = urlparse(config.container_sas_url)
        account_url = f"{parsed.scheme}://{parsed.netloc}"
        sas_token = parsed.query
        self.container_name = parsed.path.strip('/').split('/')[-1]
        self.blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)
        
        # OpenAI clients
        self.openai_client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.openai_key,
            api_version="2024-02-01",
        )
        
        # LangChain LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.openai_key,
            azure_deployment=config.openai_chat_deployment,
            api_version="2024-02-01",
            temperature=0.3,
        )


from typing_extensions import TypedDict

# ============================================================================
# RAG State for LangGraph
# ============================================================================

class RAGState(TypedDict, total=False):
    query: str
    query_embedding: List[float]
    search_results: List[Dict]
    context: str
    answer: str
    sources: List[Dict]


# ============================================================================
# Search Functions
# ============================================================================

def generate_embedding(azure_clients: AzureClients, text: str) -> List[float]:
    """Generate embedding for query text"""
    response = azure_clients.openai_client.embeddings.create(
        model=azure_clients.config.openai_embedding_deployment,
        input=text
    )
    return response.data[0].embedding


def generate_sas_url(azure_clients: AzureClients, blob_path: str) -> str:
    """Generate a blob URL using the existing container SAS token"""
    try:
        # The blob_path is like "extracted-text/folder/file.pdf.txt"
        # We need to find the original PDF path
        original_blob = blob_path.replace("extracted-text/", "").replace(".txt", "")
        
        # Parse the original SAS URL to get components
        from urllib.parse import urlparse, quote
        parsed = urlparse(azure_clients.config.container_sas_url)
        
        # Build blob URL: account/container/blob?sas_token
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        container_name = parsed.path.strip('/').split('/')[-1]
        sas_token = parsed.query
        
        # URL encode the blob path (spaces and special chars)
        encoded_blob = quote(original_blob, safe='/')
        
        # Full URL with blob path
        blob_url = f"{base_url}/{container_name}/{encoded_blob}?{sas_token}"
        return blob_url
    except Exception as e:
        logger.warning(f"Failed to generate URL for {blob_path}: {e}")
        return None


def hybrid_search(azure_clients: AzureClients, query: str, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Perform hybrid search (BM25 + Vector)"""
    
    # Vector search
    vector_query = {
        "kind": "vector",
        "vector": query_embedding,
        "fields": "content_vector",
        "k": top_k * 2,  # Get more for reranking
        "exhaustive": True
    }
    
    # Execute search
    results = azure_clients.search_client.search(
        search_text=query,  # BM25 text search
        vector_queries=[vector_query],
        select=["content", "file_name", "folder_path", "blob_path", "chunk_index", "source_container"],
        top=top_k * 2
    )
    
    # Format results
    search_results = []
    for result in results:
        blob_path = result["blob_path"]
        # Generate temporary SAS URL for the source document
        source_url = generate_sas_url(azure_clients, blob_path)
        
        search_results.append({
            "content": result["content"],
            "file_name": result["file_name"],
            "folder_path": result["folder_path"],
            "blob_path": blob_path,
            "chunk_index": result["chunk_index"],
            "source_container": result.get("source_container", "unknown"),
            "score": result.get("@search.score", 0),
            "source_url": source_url
        })
    
    # Return top_k after reranking (simple score-based)
    return search_results[:top_k]


# ============================================================================
# LangGraph Nodes
# ============================================================================

def retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """Retrieve relevant documents"""
    query = state.get("query", "")
    logger.info(f"Retrieving for query: {query[:50]}...")
    
    # Generate embedding for query
    query_embedding = generate_embedding(azure_clients, query)
    
    # Search
    search_results = hybrid_search(
        azure_clients, 
        query, 
        query_embedding, 
        top_k=5
    )
    
    logger.info(f"Found {len(search_results)} relevant chunks")
    return {
        "query": query,
        "query_embedding": query_embedding,
        "search_results": search_results
    }


def generate_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    """Generate answer using LLM"""
    
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    
    # Build context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"[Document {i}] {result['file_name']} (Path: {result['folder_path']})\n"
            f"Content: {result['content'][:800]}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Create prompt
    system_prompt = """You are a legal assistant AI. Answer the user's question based ONLY on the provided legal documents.
If the answer cannot be found in the documents, clearly state that.
Always cite the source document names in your answer.

Context from legal documents:
{context}
"""

    messages = [
        ("system", system_prompt.format(context=context)),
        ("human", query)
    ]
    
    # Generate response
    response = azure_clients.llm.invoke(messages)
    answer = response.content
    
    logger.info(f"Generated answer: {answer[:100]}...")
    return {
        "answer": answer,
        "sources": search_results,
        "context": context
    }


# ============================================================================
# Build LangGraph Workflow
# ============================================================================

def build_rag_graph(azure_clients: AzureClients):
    """Build the RAG workflow graph"""
    
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", lambda state: retrieve_node(state, azure_clients))
    workflow.add_node("generate", lambda state: generate_node(state, azure_clients))
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()


# ============================================================================
# FastAPI App
# ============================================================================

# Global clients
azure_clients: Optional[AzureClients] = None
rag_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global azure_clients, rag_graph
    
    # Startup
    logger.info("Initializing Azure clients...")
    config = Config.from_env()
    azure_clients = AzureClients(config)
    rag_graph = build_rag_graph(azure_clients)
    logger.info("RAG Backend ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Legal AI RAG Backend",
    description="AI-powered legal document search and Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "legal-ai-rag"}


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search legal documents"""
    try:
        # Generate embedding
        query_embedding = generate_embedding(azure_clients, request.query)
        
        # Search
        results = hybrid_search(
            azure_clients,
            request.query,
            query_embedding,
            top_k=request.top_k
        )
        
        return [SearchResult(**r) for r in results]
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the legal AI assistant - supports English/French bilingual"""
    try:
        # Step 1: Detect query language
        detected_lang = detect_language(request.query, azure_clients.llm)
        logger.info(f"Detected query language: {detected_lang}")
        
        # Step 2: Determine target language for response
        target_lang = request.target_language
        if target_lang == "auto":
            # If query is English, user wants English response
            # If query is French, user wants French response
            target_lang = detected_lang
        
        # Step 3: Translate query to French if needed (for search)
        search_query = request.query
        if detected_lang == "en":
            search_query = translate_text(azure_clients.llm, request.query, "en", "fr")
            logger.info(f"Translated query: {search_query[:50]}...")
        
        # Step 4: Run RAG workflow with (possibly translated) French query
        state: RAGState = {"query": search_query}
        final_state = rag_graph.invoke(state)
        
        french_answer = final_state.get("answer", "")
        sources = final_state.get("sources", [])
        
        # Step 5: Translate answer back if needed
        final_answer = french_answer
        translated_answer = None
        
        if detected_lang == "en" and target_lang == "en":
            # User asked in English, wants English answer
            final_answer = translate_text(azure_clients.llm, french_answer, "fr", "en")
            translated_answer = french_answer  # Keep original French
            logger.info(f"Translated answer back to English")
        
        return ChatResponse(
            answer=final_answer,
            sources=[SearchResult(**s) for s in sources],
            conversation_id=request.conversation_id or "new",
            detected_query_language=detected_lang,
            translated_answer=translated_answer
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(payload: Dict[Any, Any], background_tasks: BackgroundTasks):
    """Webhook endpoint for WhatsApp (GreenAPI)"""
    # This will be implemented in Sprint 4
    logger.info(f"Received WhatsApp webhook: {payload}")
    return {"status": "received"}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
