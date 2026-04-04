# Implementation Changes Log
**Started:** 2026-03-29
**Goal:** Fix all accuracy, hallucination, and data-mixing issues before 2-day delivery

---

## Status Legend
- ✅ Done
- 🔄 In Progress
- ⏳ Pending
- ❌ Blocked

---

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `.env` | ✅ Done | Added SEARCH_INDEX_EXTERNAL, SEARCH_INDEX_INTERNAL, MIN_SEARCH_SCORE |
| `recreate_index.py` | ✅ Done | Creates both indices, fixed SimpleField→SearchableField |
| `index_to_search.py` | ✅ Done | batch_size 16→512, variable shadow fix, min chunk guard, .txt strip, source_container hardcoded, external index target, SemanticConfig, SearchClient reuse |
| `index_internal.py` | ✅ Done | MD5 doc ID (was hash() — critical bug), source_container "internal"→"legal-documents-internal", internal index target, batch_size 512, SearchClient reuse |
| `rag_backend.py` | ✅ Done | Config: 2 new index fields + min_search_score. AzureClients: 2 SearchClients. New hybrid_search_isolated(). retrieve_node updated. generate_node: zero-results guard + source label fix. /search endpoint updated. |
| `whatsapp_bot.py` | ✅ Done | icon/label: "internal" in container → container == "legal-documents-internal" |

---

## Current Indexing Status (2026-03-29)
- `legal-docs-internal` → ✅ Complete
- `legal-docs-external` → 🔄 10% done (2165/21129), running in background
- `legal-docs-index` (old) → ✅ 446,933 docs — used as TEMPORARY external source for delivery
- **Delivery plan:** RAG uses old index for external + new index for internal. After delivery, swap to new external index when complete.

---

## Azure Actions Required (User Must Do These)

### ACTION 1 — Increase TPM Quota (MOST IMPORTANT)
**When:** Do this NOW before running any indexing
**Where:** Azure Portal → OpenAI resource `legal-openai-ca` → Deployments → `Embeddings` → Manage deployment → increase Token Rate Limit
**From:** 240,000 TPM (current — causes constant 429 errors)
**To:** Maximum available (1,000,000+ TPM if possible)
**Impact:** Cuts external indexing time from ~14 days to ~2-3 days

### ACTION 2 — Run recreate_index.py
**When:** After code changes are done
**Command:** `python recreate_index.py`
**What it does:** Creates two new Azure AI Search indices: `legal-docs-external` and `legal-docs-internal`
**Warning:** This deletes the old `legal-docs-index`. Old data will be gone — must re-run indexing.

### ACTION 3 — Re-index Internal Documents
**When:** After recreate_index.py completes
**Command:** `python index_internal.py`
**Duration:** 1-2 hours (small dataset)

### ACTION 4 — Re-index External Documents
**When:** After internal indexing is confirmed working
**Command:** `python index_to_search.py`
**Duration:** 2-5 hours (with increased TPM quota)

---

## Detailed Changes Per File

### `.env`
- Added `SEARCH_INDEX_EXTERNAL=legal-docs-external`
- Added `SEARCH_INDEX_INTERNAL=legal-docs-internal`
- Added `MIN_SEARCH_SCORE=0.02`

### `recreate_index.py`
- Fixed `SimpleField(searchable=True)` → `SearchableField` for blob_path, file_name, folder_path
- Updated `__main__` to create both indices (external + internal)

### `index_to_search.py`
- **batch_size**: 16 → 512 (27 API calls/file → 1 API call/file)
- **variable shadow**: `chunk_text` local var → `decoded_chunk`
- **min chunk guard**: skip chunks < 10 tokens
- **file_name**: strips `.txt` extension
- **source_container**: always `"legal-documents"` (removed fragile heuristic)
- **index target**: reads `SEARCH_INDEX_EXTERNAL` instead of `SEARCH_INDEX`
- **SemanticConfiguration**: added to `create_search_index_if_not_exists()`
- **SearchClient reuse**: created once in main(), passed to batch function

### `index_internal.py`
- **doc ID**: `hash()` → `hashlib.md5()` — fixes duplicate creation on re-runs
- **source_container**: `"internal"` → `"legal-documents-internal"`
- **index target**: reads `SEARCH_INDEX_INTERNAL` env var

### `rag_backend.py`
- **Config**: added `search_index_external`, `search_index_internal`, `min_search_score`
- **AzureClients**: two SearchClients (`search_client_external`, `search_client_internal`)
- **hybrid_search_isolated()**: new function — searches both indices separately, picks winner by score, never mixes data
- **retrieve_node**: uses new isolated search
- **generate_node**: zero-results guard (no GPT-4 call = no hallucination when nothing found)
- **source label**: fixed to match actual `"legal-documents-internal"` value

### `whatsapp_bot.py`
- **icon logic**: `"internal" in container` → `container == "legal-documents-internal"` (exact match)
- **label logic**: same fix

---

## Test Results

**Baseline (before all fixes):** 9/25 passing (36%)
**After internal-first fix:** 12/25 passing (48%)
**After retrieval precision fixes:** Target 24+/25 (96%)

| Test Area | Before | Round 1 | Round 2 |
|-----------|--------|---------|---------|
| Farrier property (1-4) | 0/4 FAIL | 3/4 PASS | - |
| Blache/Tremblay (5-7) | 2/3 PASS | 0/3 FAIL | Expected: PASS |
| Trois-Rivières (8-9) | 0/2 FAIL | 1/2 PASS | Expected: PASS |
| Capital Transit (10) | PASS | PASS | PASS |
| Brompton (11-16) | 1/6 PASS | 1/6 PASS | Expected: PASS |
| Rawdon transfer (17-19) | 3/3 PASS | 3/3 PASS | PASS |
| Charron email (20) | FAIL | FAIL | Expected: PASS |
| Caon/Charton (21-22) | 1/2 PASS | 1/2 PASS | Expected: PASS |
| Joliette/Rawdon (23-25) | 1/3 PASS | 2/3 PASS | Expected: PASS |

## Round 2 Retrieval Fixes (2026-03-29)

| Change | File | Old | New | Reason |
|--------|------|-----|-----|--------|
| MIN_SEARCH_SCORE | `.env` | 0.02 | 0.005 | Too many valid chunks were filtered out |
| top_k | `rag_backend.py` retrieve_node | 7 | 15 | Right chunk was outside retrieval window |
| Query rewriting | `rag_backend.py` rewrite_query_node | English query | French keyword query | All docs are French — BM25 needs French keywords |
| Context length | `rag_backend.py` generate_node | 1500 chars | 3000 chars | 512-token chunks = ~2000+ chars; was cutting off data |
| System prompt | `rag_backend.py` generate_node | Generic | Strict exact-extraction rules | GPT-4 was paraphrasing amounts, missing loan numbers |
