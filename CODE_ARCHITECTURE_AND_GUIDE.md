# Comprehensive System Guide: Legal AI Assistant Codebase

This document provides a detailed, file-by-file and component-by-component breakdown of the entire Legal AI Assistant codebase. It explains how data flows through the system, how the individual scripts function, and how each component interacts to provide a seamless, bilingual, legal document Q&A experience via WhatsApp.

---

## 1. High-Level Architecture & Workflow

The architecture is built on a robust Azure backbone, leveraging Azure Storage, Azure Document Intelligence (OCR), Azure AI Search (Vector + BM25), and Azure OpenAI. Fast communication to users is handled via a WhatsApp bot using GreenAPI. 

The system relies on a **strict separation of internal (confidential) and external (general) documents** to prevent data cross-contamination and hallucination in AI responses.

### **The Data Flow:**
1. **Document Upload:** Raw documents (PDFs, Word, Excel, images) are dropped into Azure Blob Storage containers (`legal-documents` for external, `legal-documents-internal` for internal).
2. **OCR Processing:** Python scripts parse these documents via Azure Document Intelligence or format-specific parsers, extracting text and placing it into `extracted-text/` blob paths.
3. **Indexing:** Indexing scripts chunk the extracted text, generate embeddings (using `text-embedding-ada-002`), and insert them into Azure AI Search indices. There are two distinct indices: one for external and one for internal.
4. **WhatsApp Webhook Payload:** A user texts or sends a voice message to the WhatsApp bot. The bot verifies group permissions, transcodes audio if necessary (using Whisper), and forwards the cleaned query to the RAG backend.
5. **RAG Backend Generation:**
   - **Language Detection:** Detect the user's language.
   - **Query Rewriting:** Formulate a French keyword search.
   - **Isolated Retrieval:** Search both internal and external indices independently. Pick highest confidence results *from one single source container* to prevent hallucination.
   - **Answer Generation:** GPT-4 generates the response based purely on the retrieved context, properly attributing the specific source document and SAS download URL.
6. **Reply:** The bot sends the formatted text back to the WhatsApp user.

---

## 2. Document Parsing & OCR Pipeline

Before documents can be queried, they must be formatted into clean text. This is handled by a set of scripts targeting specific Azure Blob containers.

### `ocr_to_blob.py`
- **Purpose:** Extracts text from external PDFs and images stored in the `legal-documents` container.
- **Workflow:** 
  It iterates through blobs, calling Azure Document Intelligence (`prebuilt-read`) to perform OCR. Resulting text is uploaded to the same container under the `extracted-text/` folder. This is heavily parallelized/batched and tracks progress to ensure fault tolerance.
- **Key Implementation:** Handles chunked upload generation and respects Azure API rate limits.

### `ocr_internal.py`
- **Purpose:** Essentially the same as `ocr_to_blob.py` but securely targets the `legal-documents-internal` container.
- **Workflow:** Extracted texts are saved under `extracted-text/internal/` path.

### `process_all_formats.py`
- **Purpose:** Expands the system's ingestion capabilities to standard office formats alongside PDFs.
- **Capabilities:**
  - **PDFs:** Handled via `PyPDF2` / `pdfplumber`.
  - **Docx/Xlsx/CSV:** Extracted using `python-docx` and `pandas`.
  - **Images/Media:** Basic metadata extraction for audio/video (using libraries like Pillow).
- **Format Header:** Notably injects a `Source: ...` / `Type: ...` header to the top of outlaid `.txt` files so that indexing scripts can accurately maintain file paths and document types.

### `retry_failed.py`
- **Purpose:** Error recovery script.
- **Workflow:** Scans the `retry_failed.log` for items that timed out or failed OCR API calls, applying an exponential backoff metric to safely reprocess them without hitting rate limit blockades.

---

## 3. Indexing & Embeddings

Once OCR text files exist in the Blob Storage, they need to be chunked into coherent overlapping text blocks, embedded into vectors, and stored in a search database.

### `recreate_index.py`
- **Purpose:** Sets up the Azure AI Search Index Schema.
- **Implementation:** Defines a `SearchIndex` containing `content`, `content_vector` (1536-dimensional HNSW vector search), `file_name`, and crucially, the `source_container` string which is used to bifurcate queries. It also attaches the `SemanticConfiguration` enabling advanced AI Search semantic reranking.

### `index_to_search.py`
- **Purpose:** Uploads text from the external `extracted-text/` container blob.
- **Mechanism:**
  - **Chunking:** Uses `tiktoken` encoding (matched to ADA-002) to segment text into 1000-token chunks with 200 token overlaps. Drops chunks smaller than 10 tokens to save API calls.
  - **Embedding:** Calls Azure OpenAI `text-embedding-ada-002`.
  - **Batching:** Pushed to Azure AI Search in highly optimized `batch_size=512`.
  - **Tagging:** All chunks are strictly tagged with `source_container: "legal-documents"`.

### `index_internal.py`
- **Purpose:** Parallel to external, but indexes internal text files from `extracted-text/internal/`.
- **Mechanism:**
  - Has a crucial fix to replace arbitrary memory hash functions (`hash()`) with deterministic `hashlib.md5()` mapping chunk names to consistent IDs, avoiding DB duplication.
  - Generates the firm classification index tag: `source_container: "legal-documents-internal"`.

---

## 4. The RAG Backend (`rag_backend.py`)

The RAG Backend uses **FastAPI** to serve endpoints and orchestrate a sophisticated LangGraph-based workflow.

### Azure Setup & Isolated Search
The backend instantiates two separate `SearchClient` objects—one for internal data, one for external data, fulfilling a "dual-index" strategy mathematically isolated to prevent data contamination.
- **`hybrid_search_isolated()` function:** Instead of using one monolithic search pool, this function fires search queries to **both** indices simultaneously. It then compares the top returned `score`. 
- **Anti-Hallucination Barrier:** If neither index returns a result score higher than `MIN_SEARCH_SCORE` (usually `0.02`), the query immediately aborts, preventing GPT-4 from making up a story. If confident, only the singular winning index hands context to the LLM. 

### LangGraph Components & Tools
The conversational flow leverages the following isolated steps defined as graph nodes:
1. **Language Detection & Translation:** `detect_language()` deduces if the question is French, English, Urdu, or Arabic using unicode heuristics and lightweight LLM prompting. It then routes to translation.
2. **`rewrite_query_node`:** Generates optimized keyword representations of the user's intent. Recognizing that the document corpus is predominantly French (`mandats notaire`, `courriels investisseur`), all English queries are rewritten specifically into structural French query-tags optimized for Azure AI search. Includes semantic classification (e.g., checking if the user asks for "invoice" versus "email").
3. **Retrieval (`retrieve_node`):** Conducts hybrid (BM25 + Semantic/Vector) searching using the isolated protocol described above. 
4. **Answer Generation (`generate_node`):** Constructs a strict contextual environment with `temperature=0.1` explicitly barring extraneous inferences. It builds SAS tokens so users can safely click documents linked heavily in citations.
5. **Auxiliary tools:** There is basic intent extraction to trigger tasks such as emails (`send_email_acs` integrating Azure Communication Services).

---

## 5. WhatsApp Integration (`whatsapp_bot.py`)

The front-line communication script relies on FastAPI to capture asynchronous callbacks from GreenAPI (a proxy service that interacts directly with WhatsApp Web).

### Key Architectural Traits
1. **Security / Group Guarding:** 
   - `_is_from_allowed_group()`: Hard-coded logic verifying the incoming `chatId` aligns precisely with known group constraints (`ALLOWED_GROUP_ID`). The bot ignores DMs natively unless whitelisted (`_is_allowed_dm`).
   - `_is_bot_mentioned()`: Validates if the Bot's exact display name (e.g. `@LegalBot`) was tagged within the raw WhatsApp string.
2. **Context Memory:**
   - Handled actively via an in-memory `ConversationMemory` dictionary caching last `MAX_HISTORY` messages. 
   - Crucially maintains memory tied to WhatsApp individual message ids (`whatsapp_message_id`) to process "replies" securely (letting users long-press previous bot messages to follow up precisely).
3. **Transcription Pipeline (Whisper Integration):**
   - When a payload tagged as `voiceMessage` or `audioMessage` triggers the webhook, the `green_api.download_media()` function grabs `.ogg` binaries.
   - Piped over to Azure OpenAI (or generic OpenAI) Whisper deployment (`self.transcriber.transcribe`), immediately feeding text to the RAG backend seamlessly simulating standard prompt behavior.
4. **Conversation Cleanup & Meta-Commands:**
   - Detects special instructions (like `/clear`, `/start`) independently of AI overhead.
5. **Robust Formatting (`_format_response`):**
   - Applies WhatsApp-specific Markdown syntax (e.g., `*bold*`, `_italic_`).
   - Dynamically renders source indicators (e.g., `🔴 Internal (Confidential)` vs `📗 External`) cleanly alongside dynamically generated, time-restricted SAS download URLs.

---

## Summary of Execution Flow
To run the fully functional legal bot:
1. `rag_backend.py` hosts Port `8000`.
2. `whatsapp_bot.py` hosts Port `8001` (to be connected with Ngrok/Webhook).
3. GreenAPI triggers payload onto `/8001` -> processes request -> passes API rest request to `/8000/chat` -> generates isolated RAG contextual response -> API responds to `/8001` -> pushes final response to WhatsApp.
