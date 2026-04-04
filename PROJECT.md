# Legal Documents AI Project - Complete Documentation

## Overview
Bilingual (English/French) AI-powered legal document assistant with WhatsApp integration. Uses Azure services for OCR, vector search, and OpenAI for RAG-based responses.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  WhatsApp User  │────▶│   WhatsApp Bot   │────▶│   RAG Backend   │
│   (GreenAPI)    │◄────│    (FastAPI)     │◄────│    (FastAPI)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                         │
                           ┌─────────────────────────────┼─────────────────────────────┐
                           │                             │                             │
                    ┌──────▼──────┐            ┌────────▼────────┐           ┌────────▼────────┐
                    │ Azure AI    │            │  Azure AI       │           │  Azure OpenAI   │
                    │  Search     │◄───────────│  Document       │           │  (Embeddings +  │
                    │ (Vector +   │            │  Intelligence   │           │   Chat)         │
                    │  BM25)      │            │    (OCR)        │           │                 │
                    └─────────────┘            └─────────────────┘           └─────────────────┘
                           ▲
                           │
                    ┌──────┴──────────────────────┐
                    │    Azure Blob Storage       │
                    │  ┌─────────────────────┐    │
                    │  │ legal-documents    │    │
                    │  │ (external files)   │    │
                    │  └─────────────────────┘    │
                    │  ┌─────────────────────┐    │
                    │  │ legal-documents-   │    │
                    │  │ internal           │    │
                    │  │ (confidential)     │    │
                    │  └─────────────────────┘    │
                    └───────────────────────────────┘
```

## Core Files

### 1. OCR Processing

#### `ocr_to_blob.py`
Main OCR processor for external legal documents.
- **Input**: `legal-documents` container (PDFs, images)
- **Output**: `extracted-text/` folder in same container
- **Azure Service**: Document Intelligence (prebuilt-read)
- **Features**:
  - Batch processing with progress tracking
  - SAS URL authentication
  - Chunked uploads for large files
  - Configurable max files via `MAX_FILES` env

#### `ocr_internal.py`
OCR processor for internal/confidential documents.
- **Input**: `legal-documents-internal` container
- **Output**: `extracted-text/internal/` folder (in main container)
- **Same Azure Service**: Document Intelligence
- **Note**: Uses separate input container, unified output location

#### `process_all_formats.py`
Multi-format document processor.
- **Supported Formats**:
  - PDF, DOCX, XLSX, CSV, TXT
  - Images: JPG, PNG, HEIC, TIFF, BMP
  - Audio/Video: MP3, WAV, MP4, AVI, MOV (metadata extraction)
- **Libraries**: 
  - PyPDF2 / pdfplumber for PDFs
  - python-docx for Word
  - pandas for Excel/CSV
  - Pillow for images

#### `retry_failed.py`
Retry mechanism for failed OCR operations.
- Reads `retry_failed.log` for failed files
- Re-attempts OCR with exponential backoff
- Updates log with new status

### 2. Indexing to Search

#### `index_to_search.py`
Indexer for external documents to Azure AI Search.
- **Source**: `extracted-text/` (OCR output from main container)
- **Target**: Azure AI Search index
- **Features**:
  - Text chunking (configurable size/overlap)
  - Azure OpenAI embeddings (text-embedding-ada-002)
  - Batch uploads (1000 docs/batch)
  - Metadata extraction: file_name, folder_path, blob_path
  - **Source tagging**: `source_container: "legal-documents"`

#### `index_internal.py`
Indexer for internal documents.
- **Source**: `extracted-text/internal/`
- **Same target index**: Unified search across both sources
- **Source tagging**: `source_container: "legal-documents-internal"`

#### `recreate_index.py`
Index management utility.
- Deletes and recreates search index
- **Schema includes**:
  - `id`, `content`, `content_vector` (1536 dims)
  - `blob_path`, `file_name`, `folder_path`
  - `chunk_index`, `total_chunks`
  - `source_container` (for filtering)

### 3. RAG Backend

#### `rag_backend.py`
FastAPI-based RAG service with bilingual support.
- **Port**: 8000
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /search` - Direct vector search
  - `POST /chat` - Full RAG with translation

**Bilingual Pipeline**:
```
User Query (FR/EN) → Detect Language → Translate to EN → 
RAG Search → GPT-4 Response → Translate to Original Lang → User
```

**LangGraph Workflow**:
1. `detect_language_node` - Detects query language
2. `translate_query_node` - Translates to English if needed
3. `retrieve_context_node` - Hybrid search (vector + BM25)
4. `generate_answer_node` - GPT-4 with context
5. `translate_response_node` - Translates back if needed

**Models**:
- Embeddings: `text-embedding-ada-002`
- Chat: `gpt-4`
- Translation: `gpt-4` with system prompt

### 4. WhatsApp Bot

#### `whatsapp_bot.py`
GreenAPI-based WhatsApp bot.
- **Port**: 8001
- **Webhook**: Receives messages from GreenAPI
- **Features**:
  - Group-only responses (configurable allowed group)
  - Bot mention detection (`@botname`)
  - Source container icons in responses:
    - 📗 Green = External documents
    - 🔴 Red = Internal documents
  - Download links with SAS tokens
  - Debug logging for group ID verification

**Response Flow**:
```
WhatsApp Message → Webhook → Check Group ID → 
Check Bot Mention → Clean Query → RAG Backend → 
Format Response with Sources → Send Reply
```

### 5. Utilities

#### `upload_drive_to_azure.py`
Uploads files from local/Google Drive to Azure Blob.
- Supports batch uploads
- Maintains folder structure

## Environment Variables

```bash
# Azure Storage
CONTAINER_SAS_URL=<sas-url-for-legal-documents>
INTERNAL_CONTAINER_SAS_URL=<sas-url-for-internal>

# Azure Document Intelligence
DI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
DI_KEY=<key>

# Azure AI Search
SEARCH_ENDPOINT=https://<resource>.search.windows.net
SEARCH_KEY=<key>
SEARCH_INDEX=legal-docs-index

# Azure OpenAI
OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
OPENAI_KEY=<key>
OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
OPENAI_CHAT_DEPLOYMENT=gpt-4

# WhatsApp (GreenAPI)
GREENAPI_URL=https://api.greenapi.com
GREENAPI_INSTANCE_ID=<id>
GREENAPI_TOKEN=<token>
ALLOWED_GROUP_ID=<group-id>@g.us
BOT_NAME=LegalBot

# RAG Backend
RAG_BACKEND_URL=http://localhost:8000

# OCR/Indexing Config
INPUT_PREFIX=uploads/
OUTPUT_PREFIX=extracted-text/
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILES=  # leave empty for unlimited
OVERWRITE=false
```

## Data Flow

### 1. Document Ingestion
```
Upload Files → Azure Blob (legal-documents or legal-documents-internal)
```

### 2. OCR Processing
```
PDFs/Images → Azure Document Intelligence → Text Files (extracted-text/)
```

### 3. Indexing
```
Text Files → Chunking → Embeddings → Azure AI Search (with source_container tag)
```

### 4. Query Flow
```
WhatsApp Query → RAG Backend → Search (Hybrid) → GPT-4 → 
Translated Response → WhatsApp Reply (with source indicators)
```

## Source Container Differentiation

| Source | Container | Output Path | Icon | Search Tag |
|--------|-----------|-------------|------|------------|
| External | legal-documents | extracted-text/ | 📗 | legal-documents |
| Internal | legal-documents-internal | extracted-text/internal/ | 🔴 | legal-documents-internal |

## Running the Project

### 1. OCR (One-time for new files)
```bash
# External documents
python ocr_to_blob.py

# Internal documents  
python ocr_internal.py

# Multi-format files (Excel, Word, etc.)
python process_all_formats.py

# Retry any failures
python retry_failed.py
```

### 2. Indexing (After OCR)
```bash
# Recreate index if schema changed
python recreate_index.py

# Index external documents
python index_to_search.py

# Index internal documents
python index_internal.py
```

### 3. Start Services
```bash
# Terminal 1: RAG Backend
python rag_backend.py

# Terminal 2: WhatsApp Bot
python whatsapp_bot.py
```

### 4. Configure Webhook
```bash
# Update GreenAPI webhook URL with ngrok
# ngrok http 8001
```

## Ports

| Service | Port |
|---------|------|
| RAG Backend | 8000 |
| WhatsApp Bot | 8001 |

## Key Features

1. **Bilingual Support**: Automatic detection and translation (FR/EN)
2. **Hybrid Search**: Vector similarity + BM25 keyword matching
3. **Source Attribution**: Clear indicators for internal vs external docs
4. **Secure Access**: SAS token-based download links
5. **Group Control**: Bot responds only in allowed WhatsApp groups
6. **Mention Detection**: Only responds when tagged (@botname)
7. **Multi-format**: PDF, Word, Excel, Images, Text files

## Dependencies

See `requirements.txt`:
- Azure SDK (Blob, Search, Document Intelligence)
- OpenAI
- LangGraph / LangChain
- FastAPI / Uvicorn
- Various file format libraries (PyPDF2, python-docx, pandas, Pillow)

---

## ⚠️ Known Issues & Problems

### 1. Embeddings & Indexing Issues

#### Issue #1: External Documents Override Internal/Test Files
**Status**: 🔴 **CRITICAL**  
**Description**: When searching, external documents (Oribus, St-Lin) always rank higher than internal test files, even when test files are more relevant.  
**Impact**: RAG returns wrong information for internal documents (e.g., Farrier evaluation shows wrong property data)  
**Example**: 
- Query: "What is Farrier property value?"
- Expected: $827,000 (from test file)
- Got: Wrong property (4,750,436 sq ft from external docs)
  
**Root Cause**: 
- Hybrid search (BM25 + Vector) gives higher scores to larger external documents
- `source_container` tag exists but no source filtering in search
- No document boosting for internal sources

**Workaround**: Use `index_test_files.py` to index local files with `source_container: test-local-files`

---

#### Issue #2: Document ID Key Errors with Special Characters
**Status**: 🟡 **FIXED** (in `index_test_files.py`)  
**Description**: French filenames with accents/spaces (e.g., `Pièces Caon.pdf.txt`, `Évaluation FARRIER`) cause Azure AI Search ID errors  
**Error**: `InvalidDocumentKey: Keys can only contain letters, digits, underscore (_), dash (-), or equal sign (=)`  
**Fix Applied**: Use MD5 hash of filename to generate safe IDs (hexadecimal only)

---

#### Issue #3: Test Files Not Retrieved Despite Indexing
**Status**: 🔴 **CRITICAL**  
**Description**: Even after indexing test files with `source_container: test-local-files`, RAG still prioritizes external documents  
**Test Results**: 
- 9 test files indexed (58 chunks total)
- Search returns external docs with higher scores (0.021 vs 0.019)
- Only 9/25 test queries passing

**Root Cause**: 
- No source filtering in `retrieve_context_node`
- Search returns top_k by relevance score, not by source priority
- Similar names (Tremblay, Brompton, Rawdon) exist in both internal and external docs

**Potential Solutions**:
1. Add source filter parameter to search endpoint
2. Boost `test-local-files` and `legal-documents-internal` scores
3. Use separate indices for internal vs external

---

#### Issue #4: Confusion Between Similar Project Names
**Status**: 🔴 **HIGH**  
**Description**: Multiple projects with same/similar names across internal and external docs cause mix-ups  
**Examples**:
| Query | Expected (Internal) | Got (External) |
|-------|---------------------|----------------|
| Brompton loan # | DP-0372, $700K | DP-0381, $1.1M |
| Tremblay investment | $250K (Joliette) | $1.1M (St-Lin) |
| Blache investment | $100K (Joliette) | $331K (Montreal) |

**Impact**: Wrong loan amounts, wrong dates, wrong people in responses

---

#### Issue #5: No Source Priority in RAG Pipeline
**Status**: 🔴 **HIGH**  
**Description**: LangGraph workflow does not prioritize internal over external sources  
**Current Flow**: `retrieve_context_node` → Returns top_k by hybrid score  
**Missing**: Source-based re-ranking or filtering  
**Files Affected**: `rag_backend.py` lines 245-270

---

### 2. Search & Retrieval Issues

#### Issue #6: Missing Internal Documents in Search Results
**Status**: 🔴 **CRITICAL**  
**Description**: `index_internal.py` looks for files in blob storage (`extracted-text/internal/`) but OCR outputs to same location  
**Test Result**: `test_source.py` shows only external documents (no `internal` or `test-local-files` in top results before test indexing)

---

#### Issue #7: Low Recall for Specific Numeric Values
**Status**: 🟡 **MEDIUM**  
**Description**: Exact numeric values (matricule numbers, specific dollar amounts) often not retrieved  
**Examples**:
- Matricule #8398-41-5065 not found
- $3,102,123 Brompton evaluation not found
- $100,000 Capital Transit payment not found

**Root Cause**: 
- Text chunking may split numbers across chunks
- BM25 weights may not prioritize exact numbers

---

### 3. RAG Backend Issues

#### Issue #8: Hallucinated Values When Document Missing
**Status**: 🔴 **HIGH**  
**Description**: When correct document not retrieved, RAG hallucinates plausible but wrong values  
**Examples**:
- Farrier property area: Expected 9,192,594 sq ft → Got 4,750,436 sq ft (different property)
- Brompton loan: Expected $700K → Got $1.1M (different loan)

**Problem**: No "document not found" response - always generates answer from available (wrong) context

---

#### Issue #9: Source Container Icons Not Consistent
**Status**: 🟡 **LOW**  
**Description**: WhatsApp bot shows 📗 for external but test files have no icon defined  
**Current Mapping**:
- `legal-documents` = 📗
- `internal` = 🔒
- `test-local-files` = 🧪 (added in test script only)

**Fix Needed**: Add `test-local-files` to icon mapping in `whatsapp_bot.py`

---

### 4. Indexing Workflow Issues

#### Issue #10: No Easy Way to Test with Clean Index
**Status**: 🟡 **MEDIUM**  
**Description**: Cannot easily test RAG with only specific documents (external docs always present)  
**Current Process**:
1. Index external docs (hundreds of files)
2. Index test files
3. Test results polluted by external docs

**Need**: Utility to temporarily disable/purge external docs for testing

---

#### Issue #11: Duplicate Documents Possible
**Status**: 🟡 **MEDIUM**  
**Description**: Running `index_to_search.py` multiple times may create duplicates  
**Current ID Generation**: `f"internal_{hash(blob_path) % 100000000}_{chunk_index}"`  
**Risk**: Same file re-indexed = different chunks get different IDs if hash collision

---

### 5. Testing Infrastructure Issues

#### Issue #12: No Automated Test Validation
**Status**: 🟡 **MEDIUM**  
**Description**: `run_rag_tests.py` exists but no automated comparison with expected answers  
**Current**: Manual keyword matching only  
**Need**: Semantic similarity check or exact value matching

---

## 🔧 Recommended Fix Priority

### Immediate (Do Now)
1. Add source filtering to `rag_backend.py` search endpoint
2. Boost `test-local-files` and `internal` source scores
3. Re-run tests after filtering fix

### Short Term (This Week)
4. Create utility to purge external docs for testing
5. Add document ID deduplication logic
6. Improve numeric value extraction in chunking

### Medium Term (Next Sprint)
7. Separate indices for internal vs external
8. Add re-ranking based on source priority
9. Implement "confidence threshold" - return "not found" if sources irrelevant

---

## 🧪 Current Test Status

**From `run_rag_tests.py` (25 test cases)**:
- ✅ PASS: 9/25 (36%)
- ❌ FAIL: 16/25 (64%)
- HIGH Priority Pass: 9/22 (41%)

**Key Passing Tests**:
- Rawdon→Brompton transfer recognition (after test indexing)
- November 2014 loan date
- Some cross-document connections

**Key Failing Tests**:
- Exact dollar amounts ($827K, $700K, $100K, $250K)
- Specific document numbers (DP-0372, P-14-365)
- Property specifications (9,192,594 sq ft, lot numbers)
- Notary contacts (Lucie Lafontaine)
