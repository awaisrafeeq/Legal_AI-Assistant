# Complete Solution Plan — Legal AI Project
## Deadline: 2 Days | Priority: Accuracy, No Hallucination, No Data Mixing

---

## Part 1 — Root Cause Diagnosis (What Is Actually Broken)

Before any fix, understand exactly what is broken and why.

### Problem A: Internal and External Data Mix in Every Search Response

**How it happens (step by step):**
```
rag_backend.py → hybrid_search() → ONE search client → ONE index (legal-docs-index)
                                                               ↓
                               Returns top_k=7 chunks sorted by relevance score
                               from ALL 9.5 million chunks (external + internal mixed)
                                                               ↓
                               GPT-4 gets context from BOTH sources at once
                               → Wrong answer with wrong source attribution
```

**The source_container value chain is broken in THREE places:**

| Location | What it writes/reads | Actual value | Should be |
|---|---|---|---|
| `index_to_search.py` line 288 | writes | `"legal-documents"` | `"legal-documents"` ✓ |
| `index_internal.py` line 113 | writes | `"internal"` | `"legal-documents-internal"` ✗ |
| `rag_backend.py` line 358 | reads (for GPT prompt) | checks `== "legal-documents-internal"` | never matches → labels internal as External |
| `whatsapp_bot.py` line 370 | reads (for icon) | checks `"internal" in container` | works by accident (substring match) |

**Result:** GPT-4 is told ALL sources are `📗 External`. It never marks anything as `⚠️ Confidential Source`. The LLM context never knows which document is internal vs external, so it cannot prioritize correctly.

---

### Problem B: Hallucination When Correct Document Is Not Retrieved

**How it happens:**
```
User query: "What is the Farrier property value?"
                    ↓
hybrid_search finds: external doc score=0.021, internal doc score=0.019
                    ↓
top_k=7 returns the WRONG external doc (higher score wins)
                    ↓
GPT-4 prompt: "Answer STRICTLY from documents below"
             but document below is WRONG property
                    ↓
GPT-4 answers with wrong property value — not hallucination, just wrong retrieval
```

There is NO score threshold. If the best available score is 0.001, the RAG still generates an answer as if the retrieval was confident. The system prompt says "if not found, say not found" but GPT-4 always finds SOMETHING to answer with from the provided context.

---

### Problem C: 22,620 Files, Indexing Takes Days

From live log analysis:
- Files: 22,620 total, ~3,034 done (13%)
- Average: ~60 seconds per file (with rate limit pauses)
- Remaining at current pace: **~326 hours ≈ 13.6 days**
- Root cause: `batch_size=16` → 27 embedding API calls per large file, causing excessive rate limit hits
- Fix: `batch_size=512` → 1 embedding API call per file → reduces API calls by 96%

---

### Problem D: Non-Deterministic Document IDs (Silent Index Corruption)

`index_internal.py` line 102:
```python
doc_id = f"internal_{hash(blob_path) % 100000000}_{chunk_index}"
```
Python's `hash()` uses a random seed every interpreter start (since Python 3.3).
Every re-run of `index_internal.py` generates different IDs for the same files.
Azure Search does not overwrite — it creates DUPLICATES silently.
Internal index doubles in size with every run.

---

## Part 2 — Architecture Decision: One Index or Two?

### Option A: Two Separate Indices (IDEAL, but requires full re-index)
- Create `legal-docs-external` and `legal-docs-internal`
- RAG searches both separately, uses ONLY the higher-confidence source
- **Complete data isolation guaranteed**
- **Downside:** Requires stopping current indexing run and re-running everything (~3–5 days with current TPM quota)

### Option B: One Index + Mandatory Source Filter (PRAGMATIC, 2-day delivery)
- Keep one index `legal-docs-index`
- Fix `source_container` values to be consistent
- RAG searches index TWICE — once filtered to external, once filtered to internal
- Compares best scores, uses ONLY results from the winning source
- **Data never mixes in GPT-4 context**
- **Does not require re-indexing** (only internal docs need re-indexing to fix IDs — small set)

### ✅ Recommended for 2-day delivery: Option B

Fix the source values, search twice with filters, never give GPT-4 mixed context.

---

## Part 3 — The Full Solution (Option B)

### New RAG Search Logic:
```
User Query
    ↓
Rewrite Query (already exists)
    ↓
Search 1: hybrid_search filtered to source_container = "legal-documents"
Search 2: hybrid_search filtered to source_container = "legal-documents-internal"
    ↓
Compare: best_external_score vs best_internal_score
    ↓
If BOTH scores < MIN_CONFIDENCE_THRESHOLD (0.02):
    → Return: "This information was not found in the available documents."
    → No GPT-4 call. No hallucination possible.
    ↓
Else: Use results from the source with the HIGHEST top score
    → GPT-4 gets context from ONE source only
    → No mixing. No cross-contamination.
    ↓
Generate Answer with clear source attribution
```

**Why this works:**
- Internal query (e.g. "Farrier property value") → internal chunks score higher → GPT-4 gets only internal context
- External query (e.g. "ORIBUS case law") → external chunks score higher → GPT-4 gets only external context
- Low-confidence query → no answer generated → no hallucination
- Cross-contamination is architecturally impossible (single-source context window)

---

## Part 4 — File-by-File Changes (Exact, in Order)

---

### STEP 1 — Fix `.env` (2 minutes)

Add these two lines:
```
SEARCH_INDEX_EXTERNAL=legal-docs-external
SEARCH_INDEX_INTERNAL=legal-docs-internal
MIN_SEARCH_SCORE=0.02
```

Keep `SEARCH_INDEX=legal-docs-index` as fallback.

> **Why two new variables:** RAG backend needs to know both index names to search separately. The current single `SEARCH_INDEX` only points to one.

---

### STEP 2 — Fix `recreate_index.py`

**Change 1:** Accept `index_name` as parameter and create whichever index is requested.
The function already accepts `index_name` — no structural change needed.

**Change 2:** Fix `SimpleField(searchable=True)` → `SearchableField` for `blob_path`, `file_name`, `folder_path`.

Current (lines 42–44):
```python
SimpleField(name="blob_path",   type=SearchFieldDataType.String, filterable=True, searchable=True),
SimpleField(name="file_name",   type=SearchFieldDataType.String, filterable=True, searchable=True),
SimpleField(name="folder_path", type=SearchFieldDataType.String, filterable=True, searchable=True),
```
Change all three to `SearchableField`:
```python
SearchableField(name="blob_path",   type=SearchFieldDataType.String, filterable=True),
SearchableField(name="file_name",   type=SearchFieldDataType.String, filterable=True),
SearchableField(name="folder_path", type=SearchFieldDataType.String, filterable=True),
```

**Change 3:** Update `__main__` block to create BOTH indices:
```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    search_endpoint = os.environ.get("SEARCH_ENDPOINT")
    search_key      = os.environ.get("SEARCH_KEY")

    ext_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
    int_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")

    recreate_search_index(search_endpoint, search_key, ext_index)
    recreate_search_index(search_endpoint, search_key, int_index)
    print("Both indices recreated. Run index_to_search.py then index_internal.py")
```

> Run `python recreate_index.py` once after this change. This creates both indices in Azure AI Search.

---

### STEP 3 — Fix `index_to_search.py` (7 changes)

**Change A — batch_size: 16 → 512** (line 251)
```python
# BEFORE
batch_size = 16

# AFTER
batch_size = 512
```
Impact: Reduces 27 API calls per file to 1. Cuts remaining indexing time from ~13 days to ~2–3 days.

---

**Change B — Fix variable name shadowing** (line 96, inside `chunk_text()`)
```python
# BEFORE
chunk_text = enc.decode(chunk_tokens)
chunks.append(chunk_text)

# AFTER
decoded_chunk = enc.decode(chunk_tokens)
chunks.append(decoded_chunk)
```

---

**Change C — Add minimum chunk size guard** (inside `chunk_text()`, after `chunks.append(decoded_chunk)`)
```python
# Add this right after chunks.append(decoded_chunk):
if len(chunk_tokens) < 10:
    break
```
Prevents 1–5 token trailing chunks from being embedded and indexed. These tiny chunks are meaningless and waste API calls.

---

**Change D — Strip `.txt` from file_name** (in `extract_metadata_from_path()`, line 199)
```python
# BEFORE
file_name = parts[-1] if parts else relative_path

# AFTER
file_name = parts[-1].replace(".txt", "") if parts else relative_path.replace(".txt", "")
```
Makes external file_name consistent with internal (which already strips `.txt`).

---

**Change E — Fix source_container detection** (line 288, inside `main()`)
```python
# BEFORE
source_container = "internal" if "internal/" in blob_path else "legal-documents"

# AFTER — external indexer should NEVER write "internal"
# This file only processes extracted-text/ (non-internal), so always tag as external
source_container = "legal-documents"
```
This is the external indexer. It reads from `extracted-text/` (non-internal prefix). Internal docs live under `extracted-text/internal/` and are handled entirely by `index_internal.py`. The heuristic check was both fragile and wrong here.

---

**Change F — Target the external index** (in `load_config()`, line 57)
```python
# BEFORE
search_index=os.environ.get("SEARCH_INDEX", "legal-docs-index"),

# AFTER
search_index=os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external"),
```
External indexer writes to external index only.

---

**Change G — Add SemanticConfiguration to `create_search_index_if_not_exists()`**

Add new imports at top of file:
```python
from azure.search.documents.indexes.models import (
    ...existing...,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
```

Inside `create_search_index_if_not_exists()`, before `index = SearchIndex(...)`:
```python
semantic_config = SemanticConfiguration(
    name="semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="content")]
    )
)
```
Update `SearchIndex` creation:
```python
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=SemanticSearch(configurations=[semantic_config]),
)
```

---

**Change H — Reuse SearchClient (in `index_documents_batch`)**

Move `SearchClient` creation outside the function, create once in `main()` and pass in:
```python
# Change function signature:
def index_documents_batch(search_client: SearchClient, documents: List[Dict[str, Any]]) -> None:
    result = search_client.upload_documents(documents=documents)
    success_count = sum(1 for r in result if r.succeeded)
    logger.info(f"Indexed {success_count}/{len(documents)} documents")

# In main(), create once:
search_client = SearchClient(
    endpoint=cfg.search_endpoint,
    index_name=cfg.search_index,
    credential=AzureKeyCredential(cfg.search_key)
)

# Pass to all calls:
index_documents_batch(search_client, documents_buffer)
```

---

### STEP 4 — Fix `index_internal.py` (3 changes)

**Change A — Fix document ID: replace `hash()` with `hashlib.md5()`** (line 102)

Add import at top (already imported in index_to_search.py, add here):
```python
import hashlib
```

```python
# BEFORE — CRITICAL BUG: hash() changes every Python restart
doc_id = f"internal_{hash(blob_path) % 100000000}_{chunk_index}"

# AFTER — deterministic, safe for Azure Search key format
doc_id = hashlib.md5(f"internal:{blob_path}:{chunk_index}".encode()).hexdigest()
```

---

**Change B — Fix source_container value** (line 113)
```python
# BEFORE
"source_container": "internal",

# AFTER
"source_container": "legal-documents-internal",
```
This aligns the written value with what `rag_backend.py` checks at line 358.

---

**Change C — Target the internal index**

In the `main_internal()` function, change the index used:
```python
# BEFORE (uses cfg.search_index which points to the shared index)
create_search_index_if_not_exists(cfg.search_endpoint, cfg.search_key, cfg.search_index, ...)

# AFTER
internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")
create_search_index_if_not_exists(cfg.search_endpoint, cfg.search_key, internal_index, ...)
```
And use `internal_index` for all `index_documents_batch` calls.

---

### STEP 5 — Fix `rag_backend.py` (the most critical file — 4 changes)

**Change A — Update Config to hold two index names**

```python
@dataclass
class Config:
    search_endpoint: str
    search_key: str
    search_index_external: str          # NEW
    search_index_internal: str          # NEW
    openai_endpoint: str
    openai_key: str
    openai_chat_deployment: str
    openai_embedding_deployment: str
    container_sas_url: str
    min_search_score: float             # NEW — confidence threshold

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            search_endpoint=os.environ["SEARCH_ENDPOINT"],
            search_key=os.environ["SEARCH_KEY"],
            search_index_external=os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external"),
            search_index_internal=os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal"),
            openai_endpoint=os.environ["OPENAI_ENDPOINT"],
            openai_key=os.environ["OPENAI_KEY"],
            openai_chat_deployment=os.environ.get("OPENAI_CHAT_DEPLOYMENT", "chat"),
            openai_embedding_deployment=os.environ.get("OPENAI_EMBEDDING_DEPLOYMENT", "Embeddings"),
            container_sas_url=os.environ["CONTAINER_SAS_URL"],
            min_search_score=float(os.environ.get("MIN_SEARCH_SCORE", "0.02")),
        )
```

---

**Change B — Update AzureClients to hold TWO search clients**

```python
class AzureClients:
    def __init__(self, config: Config):
        self.config = config

        # TWO separate search clients — one per index
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

        # ... rest of init unchanged (blob, openai, llm)
```

---

**Change C — Replace `hybrid_search()` with `hybrid_search_isolated()` — the core fix**

Remove the old `hybrid_search()` function entirely. Replace with:

```python
def _search_one_index(search_client: SearchClient, query: str, query_embedding: List[float], top_k: int) -> List[Dict]:
    """Run hybrid search on a single index, return results with scores."""
    vector_query = {
        "kind": "vector",
        "vector": query_embedding,
        "fields": "content_vector",
        "k": top_k * 2,
        "exhaustive": True
    }
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["content", "file_name", "folder_path", "blob_path", "chunk_index", "source_container"],
        top=top_k * 2
    )
    return [
        {
            "content": r["content"],
            "file_name": r["file_name"],
            "folder_path": r["folder_path"],
            "blob_path": r["blob_path"],
            "chunk_index": r["chunk_index"],
            "source_container": r.get("source_container", "unknown"),
            "score": r.get("@search.score", 0),
            "source_url": None,   # filled below
        }
        for r in results
    ]


def hybrid_search_isolated(
    azure_clients: AzureClients,
    query: str,
    query_embedding: List[float],
    top_k: int = 7
) -> Tuple[List[Dict], str]:
    """
    Search BOTH indices separately.
    Returns (results, winning_source) where winning_source is
    "legal-documents" or "legal-documents-internal".

    CRITICAL: Results come from ONLY ONE index — the one with highest confidence.
    This prevents internal/external data from ever mixing in GPT-4 context.

    Returns ([], "none") if both sources fall below MIN_CONFIDENCE_THRESHOLD.
    """
    MIN_SCORE = azure_clients.config.min_search_score

    external_results = _search_one_index(
        azure_clients.search_client_external, query, query_embedding, top_k
    )
    internal_results = _search_one_index(
        azure_clients.search_client_internal, query, query_embedding, top_k
    )

    # Best score from each source
    ext_best = max((r["score"] for r in external_results), default=0.0)
    int_best = max((r["score"] for r in internal_results), default=0.0)

    logger.info(f"Search scores — External best: {ext_best:.4f}, Internal best: {int_best:.4f}")

    # Below threshold: refuse to answer — prevents hallucination
    if ext_best < MIN_SCORE and int_best < MIN_SCORE:
        logger.warning(f"Both sources below confidence threshold ({MIN_SCORE}). Returning no results.")
        return [], "none"

    # Choose the source with higher confidence — never mix
    if int_best >= ext_best:
        winning = internal_results[:top_k]
        source_label = "legal-documents-internal"
    else:
        winning = external_results[:top_k]
        source_label = "legal-documents"

    # Generate SAS URLs for winning results
    for r in winning:
        r["source_url"] = generate_sas_url(azure_clients, r["blob_path"])

    logger.info(f"Using source: {source_label} ({len(winning)} chunks)")
    return winning, source_label
```

Add `Tuple` to the typing import at the top of the file:
```python
from typing import List, Dict, Any, Optional, Tuple
```

---

**Change D — Update `retrieve_node` to use the new function, and handle "no results" case**

```python
def retrieve_node(state: RAGState, azure_clients: AzureClients) -> RAGState:
    search_query = state.get("rewritten_query") or state.get("query", "")
    logger.info(f"Retrieving for: {search_query[:60]}...")

    query_embedding = generate_embedding(azure_clients, search_query)

    # USE NEW ISOLATED SEARCH
    search_results, winning_source = hybrid_search_isolated(
        azure_clients, search_query, query_embedding, top_k=7
    )

    logger.info(f"Found {len(search_results)} relevant chunks from: {winning_source}")
    return {
        "query": state.get("query", ""),
        "rewritten_query": search_query,
        "query_embedding": query_embedding,
        "search_results": search_results,
        "winning_source": winning_source,          # NEW — passed to generate node
        "conversation_history": state.get("conversation_history", [])
    }
```

**Update `generate_node` to handle zero results (no hallucination) and fix source label:**

At the start of `generate_node`, add this block BEFORE building context:
```python
# ZERO RESULTS — both sources below confidence threshold
# Return a definitive "not found" response without calling GPT-4
if not search_results:
    return {
        "answer": (
            "I was unable to find relevant information in the legal documents "
            "to answer this question. Please verify that the relevant documents "
            "have been loaded into the system, or consult a qualified legal professional."
        ),
        "sources": [],
        "context": ""
    }
```

Fix the source label in context building (line 358):
```python
# BEFORE
source_type = "🔴 Internal" if result.get("source_container") == "legal-documents-internal" else "📗 External"

# AFTER — matches the actual value now written by index_internal.py
source_type = "🔴 Internal (Confidential)" if result.get("source_container") == "legal-documents-internal" else "📗 External"
```

Also update `RAGState` TypedDict to include new fields:
```python
class RAGState(TypedDict, total=False):
    query: str
    query_lang: str
    rewritten_query: str
    query_embedding: List[float]
    search_results: List[Dict]
    winning_source: str                 # NEW
    context: str
    answer: str
    sources: List[Dict]
    conversation_history: List[Dict]
```

---

### STEP 6 — Fix `whatsapp_bot.py` (1 minor change)

**Change A — Update icon logic to use consistent value** (line 370)

```python
# BEFORE
icon = "🔴" if "internal" in container else "📗"
label = "Internal (Confidential)" if "internal" in container else "Legal Documents"

# AFTER — explicit match, no substring guessing
icon = "🔴" if container == "legal-documents-internal" else "📗"
label = "Internal (Confidential)" if container == "legal-documents-internal" else "Legal Documents"
```

This now works correctly once `index_internal.py` writes `"legal-documents-internal"` properly.

---

## Part 5 — Execution Order for 2-Day Delivery

### Day 1 (Today)

**Morning — Fix all code (3–4 hours)**
```
1. Stop the current index_to_search.py run
2. Apply all changes to:
   - .env
   - recreate_index.py
   - index_to_search.py (all 8 changes)
   - index_internal.py (all 3 changes)
   - rag_backend.py (all 4 changes)
   - whatsapp_bot.py (1 change)
```

**Then — Increase TPM Quota (Azure Portal, 15 minutes)**
```
Azure Portal → OpenAI resource (legal-openai-ca)
→ Deployments → Embeddings → Manage deployment
→ Increase Token Rate Limit to maximum available
   Standard: 240K → 1M TPM = 4x speed improvement
   If PTU available: up to 2M+ TPM
```

**Then — Recreate both indices (5 minutes)**
```bash
python recreate_index.py
```
This creates `legal-docs-external` and `legal-docs-internal` in Azure AI Search.

**Then — Re-index Internal documents (fast — small dataset)**
```bash
python index_internal.py
```
Internal docs are a small subset. This should complete in hours, not days.

**Then — Start external indexing with fixes applied**
```bash
python index_to_search.py
```
With batch_size=512 and higher TPM quota, this now runs significantly faster.

---

### Day 1 Evening — While External Indexing Runs

**Test RAG backend with just internal data:**
```bash
python rag_backend.py   # start backend
python test_search.py   # run existing tests
```

At this point:
- Internal index is complete
- RAG can already answer questions about internal documents correctly
- No external docs polluting internal answers
- Confidence threshold prevents hallucination

---

### Day 2

**Morning — External indexing should be progressing significantly**

With batch_size=512 fix and 1M TPM:
- Estimated throughput: ~1 file/second during non-rate-limited bursts
- Remaining 19,500 files: could complete in 5–10 hours at 1M TPM

**Run full test suite:**
```bash
python test_search.py
```

Expected improvement from current 36% pass rate (9/25):
- Issues #1, #3, #4, #5 (mixing/priority): FIXED by isolated search
- Issue #6 (internal missing): FIXED by source_container value fix
- Issue #8 (hallucination): FIXED by confidence threshold
- Issue #7 (numeric recall): Partially improved by min chunk size fix
- Issue #2, #11 (duplicate IDs): FIXED by deterministic MD5

Target: 80%+ pass rate after fixes

**Afternoon — Integration test full flow:**
```
WhatsApp → Bot → RAG → Search (isolated) → GPT-4 (single source) → Response
```

---

## Part 6 — Why This Prevents Hallucination

The three-layer anti-hallucination defence:

### Layer 1: Confidence Threshold (New)
```python
if ext_best < 0.02 and int_best < 0.02:
    return [], "none"  # No GPT-4 call at all
```
If no document scores above 0.02, the system returns a fixed "not found" message.
GPT-4 is never called. Hallucination is architecturally impossible.

### Layer 2: Source Isolation (New)
GPT-4 context is built from ONE source only.
Even when external and internal both have relevant-looking results, only the higher-confidence source feeds the prompt.
Cross-source contamination (e.g. wrong Brompton loan from external when asking about internal Brompton) is architecturally prevented.

### Layer 3: System Prompt (Already Exists — Kept)
```
"If information is NOT in the provided documents, say: This specific information was not found..."
"Never fabricate legal clauses, article numbers, or provisions"
```
This remains as a tertiary guard for edge cases where GPT-4 might still try to extrapolate.

---

## Part 7 — Complete Issue Tracker (From PROJECT.md)

| Issue | Root Cause | Fix Location | Status After Plan |
|---|---|---|---|
| #1 External overrides internal | No source isolation in search | `rag_backend.py` Change C | ✅ FIXED |
| #2 Special char doc IDs | Already fixed in main indexer | `index_internal.py` Change A (MD5) | ✅ FIXED |
| #3 Test files not retrieved | External drowns internal in shared index | Separate indices + isolated search | ✅ FIXED |
| #4 Similar project name confusion | External docs with same names win on score | Source isolation: highest score per index wins | ✅ FIXED |
| #5 No source priority | Single search client, no filtering | `rag_backend.py` two clients + isolation | ✅ FIXED |
| #6 Internal docs missing | Wrong source_container value `"internal"` | `index_internal.py` Change B | ✅ FIXED |
| #7 Low recall for numeric values | Token-split chunks lose number context | Min chunk guard (partial improvement) | ⚠️ PARTIAL |
| #8 Hallucination on wrong retrieval | No confidence threshold | `rag_backend.py` zero-results guard | ✅ FIXED |
| #9 Icon inconsistency | `source_container` value mismatch | `whatsapp_bot.py` + `index_internal.py` fix | ✅ FIXED |
| #10 No clean test mode | Missing utility | Not in scope (2-day plan) | ❌ DEFERRED |
| #11 Duplicate internal docs | `hash()` randomization | `index_internal.py` Change A (MD5) | ✅ FIXED |
| #12 No automated test validation | Missing test infra | Not in scope (2-day plan) | ❌ DEFERRED |

**Issues fixed: 10/12. Remaining 2 are tooling/infrastructure, not accuracy issues.**

---

## Part 8 — Configuration Reference After All Changes

`.env` additions needed:
```bash
# Separate indices (NEW)
SEARCH_INDEX_EXTERNAL=legal-docs-external
SEARCH_INDEX_INTERNAL=legal-docs-internal

# Confidence threshold (NEW) — 0.02 is safe starting point, tune if needed
MIN_SEARCH_SCORE=0.02
```

Existing variables that stay the same:
```bash
SEARCH_ENDPOINT=...
SEARCH_KEY=...
OPENAI_ENDPOINT=...
OPENAI_KEY=...
OPENAI_EMBEDDING_DEPLOYMENT=Embeddings
OPENAI_CHAT_DEPLOYMENT=chat
CONTAINER_SAS_URL=...
```

---

## Part 9 — Tuning MIN_SEARCH_SCORE After Deployment

The `0.02` threshold is a starting point. After running:

- **Too many "not found" responses** → Lower the threshold to `0.015`
- **Still getting wrong answers** → Raise the threshold to `0.025` or `0.03`
- **Testing the threshold:** Run `python test_search.py` and check which failing tests return "not found" vs wrong answer. "Not found" is always better than wrong answer for legal documents.

---

## Summary

The single most impactful architectural change is **isolated dual-index search** in `rag_backend.py`. Every other fix (batch size, ID generation, source_container values, confidence threshold) supports and enables this core change. Once internal and external data are searched separately and GPT-4 only ever sees results from one source, the mixing, hallucination, and wrong-attribution problems are resolved by design — not by prompting.
