# Embeddings Pipeline — Full Audit & Fix Plan

## Files That Need To Be Edited

| File | Severity | Issues |
|------|----------|--------|
| `index_to_search.py` | Critical + Medium | Batch size, variable shadowing, file_name inconsistency, fragile source detection, schema without SemanticConfig, no min chunk guard, SearchClient per batch |
| `index_internal.py` | Critical + Medium | Non-deterministic doc ID (hash randomization bug), file_name inconsistency |
| `recreate_index.py` | Low | SimpleField declared searchable — does not do full-text search |

`rag_backend.py` and `whatsapp_bot.py` do **not** need changes. They are internally consistent with the actual values the indexers write.

---

## Context: What The Live Logs Revealed

Before the fix plan, the live indexing logs (shared by user, timestamped 2026-03-28 23:19–23:28) revealed actual runtime behavior that cannot be seen from reading code alone:

```
Indexing documents:  13%|███  | 3032/22620 [22:23:21<346:33:07, 63.69s/it]
```

### Key findings from live logs

**1. Total dataset is 22,620 files — not ~3,000**
The `retry_failed.log` only showed 3,139 retry candidates. The actual index run is processing 22,620 OCR text files.

**2. OCR text files are ~780 KB each**
```
Content-Length: 797760   (file 3032)
Content-Length: 797768   (file 3033)
Content-Length: 797768   (file 3034)
```
These are large legal case compilations (ORIBUS case, banque de Montréal bank statement lots). At ~4 bytes per token this is approximately **195,000 tokens per file**.

**3. Actual chunks per file: ~422 — not ~11 as assumed**
With `MAX_CHUNK_SIZE=512` and `CHUNK_OVERLAP=50`:
- Step size = 512 − 50 = **462 tokens**
- Chunks per file = 195,000 ÷ 462 ≈ **422 chunks**

**4. Embedding API calls per file with batch_size=16: 27 calls**
```
ceil(422 / 16) = 27 API calls per file
```
This is the direct consequence of the small batch size combined with huge files.

**5. Rate limits hitting multiple times per file cycle**
```
23:19:45  → HTTP 429, retry in 35s
23:21:18  → HTTP 429, retry in 4s
23:21:23  → HTTP 429, retry in 1s   ← immediately again
23:21:28  → HTTP 429, retry in 31s
```
Four rate limit hits within a 2-minute window. Each pause wastes between 1 and 35 seconds.

**6. Azure Search batch upload is a hidden bottleneck**
```
Content-Length: 37,291,203 bytes  → 30 seconds to upload 1000 docs
Content-Length: 37,274,219 bytes  → 90 seconds to upload 1000 docs
```
Each 1,000-document batch sent to Azure AI Search is a **37 MB HTTP payload**. This comes from JSON-encoding 1,000 × 1,536 floats (~23 MB for vectors alone) plus text content. Upload time varies from 30 to 90 seconds per batch depending on Azure service load.

**7. Estimated completion at current pace: ~14 more days**
- Files remaining: 22,620 − 3,034 = **19,586**
- Average pace from tqdm: ~60 s/file
- Remaining time: 19,586 × 60 = **~326 hours ≈ 13.6 days**
- Already elapsed: **22+ hours**

---

## Fix Plan By File

---

### FILE 1: `index_to_search.py`

---

#### Fix A — Batch size: 16 → 512 (Performance, High Impact)

**Location:** Line 251
**Current:**
```python
batch_size = 16  # Embeddings API batch size
```
**Change to:**
```python
batch_size = 512
```

**Why this matters:**
With 422 chunks per file:
- At batch_size=16: `ceil(422/16)` = **27 API calls per file**
- At batch_size=512: `ceil(422/512)` = **1 API call per file**

That is 26 fewer HTTP round trips per file × 22,620 files = **588,120 fewer API calls total**.

At 400 ms per call, that recovers approximately **65 hours of pure HTTP overhead** from the remaining 19,586 files. It also reduces the RPM (requests per minute) load on your Azure OpenAI deployment, directly reducing the frequency of 429 rate limit hits.

Note: This does NOT reduce total tokens sent (TPM stays the same), so the TPM-based rate limit floor is unchanged. But eliminating unnecessary round trips is a real gain.

Do NOT set this to 2048. The Azure OpenAI embeddings API supports 2048 inputs per call, but large batches also consume more memory per request and can hit different server-side limits. 512 is safe and sufficient for your file sizes.

---

#### Fix B — Variable name shadows function name (Correctness, Low)

**Location:** Line 96 inside `chunk_text()` function
**Current:**
```python
chunk_text = enc.decode(chunk_tokens)
chunks.append(chunk_text)
```
**Change to:**
```python
decoded_chunk = enc.decode(chunk_tokens)
chunks.append(decoded_chunk)
```

**Why:** The local variable `chunk_text` has the same name as the function it is defined inside. Python does not error on this, but it is a silent naming collision. Any static analysis tool or future developer will be confused. Rename the local variable to `decoded_chunk`.

---

#### Fix C — No minimum chunk size guard (Quality, Medium)

**Location:** `chunk_text()` function, inside the while loop, after line 96
**Current:** No guard. The last chunk of a file can be 1 token.
**Add after the `chunks.append(...)` line:**
```python
# Skip chunks smaller than 10 tokens — not semantically useful
if len(chunk_tokens) < 10:
    break
```

**Why:** When a 195,000-token file ends, the final sliding window step may produce a chunk of only a few tokens (e.g., a stray punctuation mark or number at the end of a scanned document). Generating a full 1,536-dimensional embedding for 3 tokens wastes one embedding API call, one index document slot, and produces meaningless search results.

---

#### Fix D — `file_name` includes `.txt` extension (Correctness, Medium)

**Location:** `extract_metadata_from_path()`, line 199
**Current:**
```python
file_name = parts[-1] if parts else relative_path
```
**Change to:**
```python
file_name = parts[-1].replace(".txt", "") if parts else relative_path.replace(".txt", "")
```

**Why:** The OCR output files are named `original_document.pdf.txt`. This function currently stores `original_document.pdf.txt` as the `file_name` in the search index. `index_internal.py` already strips `.txt` from its file names. This inconsistency means the same physical document indexed by the two different scripts will have different `file_name` values in the same search index, breaking any filter or display that relies on file name matching.

---

#### Fix E — Source container detection is a fragile path heuristic (Correctness, Medium)

**Location:** `main()`, line 288
**Current:**
```python
source_container = "internal" if "internal/" in blob_path else "legal-documents"
```
**Change to:**
```python
source_container = "internal" if blob_path.startswith(cfg.input_prefix.strip("/") + "/internal/") else "legal-documents"
```

**Why:** The current check `"internal/" in blob_path` matches anywhere in the path. A file under a folder named `internal-affairs/` or `non-internal/` would be falsely classified as internal. The fixed version anchors the check to the known prefix and requires `internal/` to appear immediately after the root prefix, which matches your actual blob structure (`extracted-text/internal/...`).

---

#### Fix F — `create_search_index_if_not_exists` creates schema without SemanticConfiguration (Correctness, Low)

**Location:** `create_search_index_if_not_exists()`, lines 164–174
**Current:** The function creates a `SearchIndex` with only `vector_search`, no `SemanticConfiguration`.
**What `recreate_index.py` does (correctly):** It adds a `SemanticConfiguration` pointing `content` as the prioritized field.

**The problem:** If someone runs `index_to_search.py` on a fresh environment without having run `recreate_index.py` first, the index is created without semantic configuration. If `rag_backend.py` is later enhanced to use semantic re-ranking, it will silently fail.

**Fix:** Import and add the same `SemanticConfiguration` block that already exists in `recreate_index.py` into `create_search_index_if_not_exists()`.

Required new imports:
```python
from azure.search.documents.indexes.models import (
    ...existing imports...,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
)
```
Add before `index = SearchIndex(...)`:
```python
from azure.search.documents.indexes.models import SemanticSearch
semantic_config = SemanticConfiguration(
    name="semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="content")]
    )
)
```
And update:
```python
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=SemanticSearch(configurations=[semantic_config]),
)
```

---

#### Fix G — New `SearchClient` instantiated every batch upload (Performance, Low)

**Location:** `index_documents_batch()`, line 185
**Current:** A new `SearchClient` is constructed on every call to `index_documents_batch`.
**Fix:** Create the `SearchClient` once in `main()` and pass it as a parameter into `index_documents_batch()`.

**Impact:** Each `SearchClient` instantiation involves object allocation and HTTP connection setup. With 37 MB batches every 1,000 documents and 22,620 files × ~422 chunks = ~250 batch flushes, saving ~50 ms per instantiation recovers about 12 seconds. Low impact, but a clean fix.

---

### FILE 2: `index_internal.py`

---

#### Fix H — Non-deterministic doc ID using Python's `hash()` (CRITICAL BUG)

**Location:** Line 102
**Current:**
```python
doc_id = f"internal_{hash(blob_path) % 100000000}_{chunk_index}"
```
**Change to:**
```python
import hashlib
doc_id = hashlib.md5(f"internal:{blob_path}:{chunk_index}".encode()).hexdigest()
```

**Why this is critical:** Since Python 3.3, the built-in `hash()` function uses a random seed that changes every time the Python interpreter starts (PYTHONHASHSEED). This means:
- Run 1: `hash("extracted-text/internal/file.txt")` = `7234829104`
- Run 2: `hash("extracted-text/internal/file.txt")` = `1847362905`

Every time you re-run `index_internal.py`, every internal document gets a **different ID**. Azure AI Search will not overwrite the existing documents — it will **create duplicates**. Over time the index fills with multiple copies of every internal document. This does not crash — it silently pollutes the index.

The main `index_to_search.py` correctly uses `hashlib.md5` which is deterministic. `index_internal.py` must match this pattern.

---

#### Fix I — `file_name` strips `.txt` (inconsistency with `index_to_search.py`)

**Location:** Line 21
**Current:**
```python
file_name = parts[-1].replace(".txt", "")
```

This strips `.txt` — but Fix D above makes `index_to_search.py` also strip `.txt`. Once Fix D is applied to `index_to_search.py`, both files will behave identically and this line in `index_internal.py` is already correct. No change needed here **after Fix D is applied**.

If Fix D is applied to `index_to_search.py` before this script is reviewed: ✓ consistent.
If only `index_internal.py` is updated without Fix D: inconsistency remains.

**Action:** Apply Fix D to `index_to_search.py` first. No separate change needed in `index_internal.py` for this item.

---

### FILE 3: `recreate_index.py`

---

#### Fix J — `SimpleField(searchable=True)` does not perform full-text search (Correctness, Low)

**Location:** Lines 42–44
**Current:**
```python
SimpleField(name="blob_path",    type=SearchFieldDataType.String, filterable=True, searchable=True),
SimpleField(name="file_name",    type=SearchFieldDataType.String, filterable=True, searchable=True),
SimpleField(name="folder_path",  type=SearchFieldDataType.String, filterable=True, searchable=True),
```
**Change to:**
```python
SearchableField(name="blob_path",   type=SearchFieldDataType.String, filterable=True),
SearchableField(name="file_name",   type=SearchFieldDataType.String, filterable=True),
SearchableField(name="folder_path", type=SearchFieldDataType.String, filterable=True),
```

**Why:** In Azure AI Search, `SimpleField` stores the value as-is for exact matching and filtering. The `searchable=True` flag on a `SimpleField` has no effect — it does not enable tokenized full-text search. Only `SearchableField` uses a Lucene analyzer and participates in BM25 scoring. If a user searches for a partial file name like "Charron" and you want it to match `file_name = "11002.01F - 161129 - Documents Charron"`, the field must be a `SearchableField`.

**Important note:** This change requires re-creating the index (run `recreate_index.py` again) and re-running the full indexing pipeline because changing a field type on an existing index is not possible — you must delete and recreate.

The same `SimpleField(searchable=True)` bug exists in `index_to_search.py` lines 148–150 (the `create_search_index_if_not_exists` function). Once Fix F above is applied, update those three lines there as well to use `SearchableField`.

---

## Performance Summary: Before vs After Fixes

| Metric | Current (live logs) | After Fixes A–J |
|--------|---------------------|-----------------|
| Embedding API calls per file | 27 | 1–2 |
| Rate limit hits | Multiple per file cycle | Significantly reduced (fewer RPM calls) |
| Azure Search upload payload | 37 MB / 1000 docs, 30–90 s | Unchanged — this is the index vector size |
| Duplicate internal docs on re-run | YES (hash randomization) | No |
| file_name consistency | Broken (`.txt` in external, not in internal) | Consistent |
| Estimated remaining time (current pace) | ~326 hours (~13.6 days) | ~50–80 hours (~2–3 days) |

The estimated improvement from ~326 hours to ~50–80 hours comes primarily from Fix A (batch size) reducing 27 API calls to 1 per file, which drastically reduces the rate limit pressure and HTTP overhead.

---

## What Cannot Be Fixed By Code Changes Alone

**TPM rate limit floor:**
Your dataset has ~22,620 files × ~422 chunks × 512 tokens ≈ **4.9 billion tokens** to embed.
At the standard Azure OpenAI `text-embedding-ada-002` quota of **240,000 tokens/minute**, the absolute minimum time regardless of any code optimization is:
```
4,900,000,000 ÷ 240,000 = 20,417 minutes ≈ 14.2 days
```
You cannot go below this floor without increasing your TPM quota.

**Action required outside the code:**
Azure Portal → your OpenAI resource (`legal-openai-ca`) → Deployments → `Embeddings` → Manage deployment → increase token rate limit.
If you can reach **1,000,000 TPM**, the floor drops to ~3.4 days.
If you can reach **2,000,000 TPM** (requires PTU/provisioned throughput), the floor drops to ~1.7 days.

---

## Recommended Order of Changes

1. **Stop the current indexing run** (it is going to take 13+ more days as-is)
2. Apply **Fix H** to `index_internal.py` first — the hash bug is silently corrupting your internal index right now if it has been run before
3. Apply **Fix A** (batch size) and **Fix B, C, D, E** to `index_to_search.py`
4. Apply **Fix F** (SemanticConfig) and **Fix G** (SearchClient reuse) to `index_to_search.py`
5. Apply **Fix J** to `recreate_index.py` (and the matching lines in `create_search_index_if_not_exists` in `index_to_search.py`)
6. **Increase TPM quota** in Azure Portal (most impactful single action)
7. Run `recreate_index.py` to rebuild the index with the corrected schema
8. Re-run `index_to_search.py` and `index_internal.py` with all fixes applied
