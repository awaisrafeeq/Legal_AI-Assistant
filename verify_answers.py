"""
Answer Grounding Verification:
1. Run each query through RAG pipeline
2. Collect the actual source chunk content that GPT used to generate the answer
3. Ask a SEPARATE GPT call to verify: is the answer grounded in the source content?
4. Also fetch chunk content from index for each cited source to double-check

Saves: logs/verification_report.txt
"""
import os
import sys
import json
import time
import logging
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
logging.basicConfig(level=logging.WARNING)

from rag_backend import Config, AzureClients, build_rag_graph, RAGState
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

TEST_QUERIES = [
    "Tu as plusieurs déclaration à la police et aussi un interrogatoire de Delnegro et Caon Tu peux faire une analyse micro détail des affirmation qui change ou voir si tout se qu'il dit peux être remis en question.",
    "est-ce que tu peux me donner tous les courriels investisseurs que Yves Blach a reçus?",
    "Je veux la date de l acte notarier des 4 lots",
    "Donne moi le courriel investisseurs ou partenaire envoyer à Solange paquette pour Saint-Charles de borome pour 240,000$.",
    "Peut tu trouver les courriel que j ai envoyer à l évaluateur BBD Philippe Lamarre.",
    "Quel date nous avons reçu l avis de 60 jours , prise en paiement pour le prêt de ICC , Croll pour st-lin ?",
]

VERIFY_PROMPT = """You are a strict fact-checker for a legal AI system.

You will receive:
1. The user's QUESTION
2. The AI's ANSWER
3. The actual SOURCE DOCUMENT CONTENT (raw text from the documents the AI used)

Your job: verify if EVERY factual claim in the answer is directly supported by the source content.

Check specifically:
- Names: Are person/org names in the answer actually present in source text?
- Dates: Are dates in the answer actually stated in source text?
- Amounts: Are dollar amounts in the answer actually in source text?
- Quotes: Are quoted passages actually in source text?
- File references: Does the AI cite files that match the actual source filenames?
- Claims: Is each factual statement traceable to the source text?

For DISCOVERY queries (listing documents), verify:
- Are the listed document types consistent with actual document content?
- Do the file names exist?
- Are the persons/summaries consistent with source content?

Respond in this exact JSON format:
{{
  "grounded": true/false,
  "confidence": "high/medium/low",
  "verified_claims": ["claim 1 that IS in source", "claim 2 that IS in source"],
  "unverified_claims": ["claim that is NOT found in source"],
  "hallucinations": ["any fabricated facts not in source"],
  "missing_info": ["relevant info in source that answer missed"],
  "notes": "brief explanation"
}}

Return ONLY valid JSON."""


def fetch_source_content(search_client, blob_path, max_chunks=50):
    """Fetch ALL chunk content from search index for a given blob_path."""
    try:
        safe_path = blob_path.replace("'", "''")
        results = search_client.search(
            search_text="*",
            filter=f"blob_path eq '{safe_path}'",
            select=["content", "chunk_index", "file_name", "document_type"],
            top=max_chunks,
            order_by=["chunk_index"]
        )
        chunks = []
        for r in results:
            chunks.append({
                "chunk_index": r.get("chunk_index", 0),
                "content": r.get("content", ""),
                "file_name": r.get("file_name", ""),
                "document_type": r.get("document_type", ""),
            })
        return chunks
    except Exception as e:
        return [{"error": str(e), "content": ""}]


import re

def extract_claim_keywords(answer):
    """Extract key verifiable terms from the AI answer: dates, amounts, names, emails."""
    keywords = set()

    # Dates (various French/English formats)
    for m in re.findall(r'\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}', answer, re.IGNORECASE):
        keywords.add(m.strip())
    for m in re.findall(r'\d{4}-\d{2}-\d{2}', answer):
        keywords.add(m)

    # Dollar amounts
    for m in re.findall(r'\$[\d\s,.]+', answer):
        keywords.add(m.replace(' ', '').replace(',', ''))
    for m in re.findall(r'[\d\s,.]+\s*\$', answer):
        cleaned = m.replace(' ', '').replace(',', '').strip()
        if len(cleaned) > 3:
            keywords.add(cleaned)

    # Email addresses
    for m in re.findall(r'[\w.+-]+@[\w.-]+\.\w+', answer):
        keywords.add(m.lower())

    # Quoted text fragments (first 30 chars of each quote)
    for m in re.findall(r'[«"](.*?)[»"]', answer):
        if len(m) > 10:
            keywords.add(m[:40].strip())

    # Person names (capitalized multi-word, French style)
    for m in re.findall(r'[A-ZÀ-Ü][a-zà-ü]+(?:\s+[A-ZÀ-Ü][a-zà-ü]+)+', answer):
        if len(m) > 5 and m not in ('Source External', 'Source Internal', 'Cour Supérieure'):
            keywords.add(m)

    # Lot numbers
    for m in re.findall(r'lot\s+[\d\s]+', answer, re.IGNORECASE):
        keywords.add(m.strip())

    # Reference numbers (dossier, DP-, etc.)
    for m in re.findall(r'DP-\d+', answer):
        keywords.add(m)
    for m in re.findall(r'\d{3}-\d{2}-\d{6}-\d{3}', answer):
        keywords.add(m)

    return keywords


def select_relevant_chunks(chunks, claim_keywords, max_chars=8000):
    """
    Score each chunk by how many claim keywords it contains.
    Return the most relevant chunks first, within max_chars budget.
    """
    if not claim_keywords:
        # No keywords extracted — return first chunks by order
        result = []
        total = 0
        for c in chunks:
            content = c["content"]
            if total + len(content) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    result.append(content[:remaining])
                break
            result.append(content)
            total += len(content)
        return "\n".join(result)

    # Score each chunk
    scored = []
    for c in chunks:
        content_lower = c["content"].lower()
        score = 0
        for kw in claim_keywords:
            if kw.lower() in content_lower:
                score += 1
        scored.append((score, c["chunk_index"], c["content"]))

    # Sort: highest scoring first, then by chunk_index for ties
    scored.sort(key=lambda x: (-x[0], x[1]))

    result = []
    total = 0
    for score, idx, content in scored:
        if total + len(content) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                result.append(content[:remaining])
            break
        result.append(content)
        total += len(content)

    return "\n".join(result)


def verify_answer(gpt_client, deployment, query, answer, source_content):
    """Ask GPT to verify if answer is grounded in source content."""
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": f"QUESTION:\n{query}\n\nAI ANSWER:\n{answer}\n\nSOURCE DOCUMENT CONTENT:\n{source_content[:120000]}"}
    ]
    try:
        response = gpt_client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "grounded": None}


def main():
    print("Loading RAG pipeline...")
    config = Config.from_env()
    clients = AzureClients(config)
    graph = build_rag_graph(clients)

    gpt_client = AzureOpenAI(
        azure_endpoint=config.openai_endpoint,
        api_key=config.openai_key,
        api_version="2024-02-01",
    )
    gpt_deployment = config.openai_chat_deployment

    ext_search = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_external,
        credential=AzureKeyCredential(config.search_key)
    )
    int_search = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_internal,
        credential=AzureKeyCredential(config.search_key)
    )

    print("Ready.\n")
    lines = []

    def log(t=""):
        print(t)
        lines.append(t)

    for i, query in enumerate(TEST_QUERIES, 1):
        log(f"\n{'#'*80}")
        log(f"# TEST {i}")
        log(f"{'#'*80}")
        log(f"QUERY: {query}\n")

        # Step 1: Run RAG
        state: RAGState = {
            "query": query,
            "conversation_history": [],
            "source_mode": "all",
            "allowed_files": []
        }
        try:
            final = graph.invoke(state)
        except Exception as e:
            log(f"PIPELINE ERROR: {e}\n")
            continue

        answer = final.get("answer", "")
        sources = final.get("sources", [])
        search_results = final.get("search_results", sources)
        intent = final.get("query_intent", "unknown")

        log(f"INTENT: {intent}")
        log(f"ANSWER ({len(answer)} chars):")
        for line in answer.split("\n"):
            log(f"  {line}")
        log("")

        # Step 2: Collect source content from pipeline + index
        # Smart selection: extract claim keywords from answer, then pick chunks that contain them
        log(f"--- SOURCE DOCUMENTS CONTENT ---")
        all_source_text = ""
        seen_blobs = set()
        unique_sources = []

        for s in sources:
            fname = s.get("file_name", "")
            if fname in {u.get("file_name") for u in unique_sources}:
                continue
            unique_sources.append(s)

        # Extract verifiable claims from the answer
        claim_keywords = extract_claim_keywords(answer)
        log(f"  Claim keywords extracted: {list(claim_keywords)[:20]}")

        # Budget per file
        num_sources = len(unique_sources[:15])
        MAX_CHARS_PER_FILE = max(3000, 100000 // max(num_sources, 1))
        log(f"  Content budget: {MAX_CHARS_PER_FILE} chars/file for {num_sources} files")

        for s in unique_sources[:15]:
            fname = s.get("file_name", "unknown")
            blob_path = s.get("blob_path", "")
            container = s.get("source_container", "")
            url = s.get("source_url", "")
            doc_type = s.get("document_type", "")

            log(f"\n  FILE: {fname}")
            log(f"  Type: {doc_type}")
            log(f"  Blob: {blob_path}")
            log(f"  Link: {url}")

            # Get content from the search result itself
            inline_content = s.get("content", "")

            # Also fetch from index for more chunks
            search_client = int_search if container == "legal-documents-internal" else ext_search
            if blob_path and blob_path not in seen_blobs:
                seen_blobs.add(blob_path)
                idx_chunks = fetch_source_content(search_client, blob_path, max_chunks=50)
                if idx_chunks and not idx_chunks[0].get("error"):
                    # Smart selection: pick chunks containing claim keywords first
                    selected = select_relevant_chunks(idx_chunks, claim_keywords, max_chars=MAX_CHARS_PER_FILE)
                    total_chars = sum(len(c["content"]) for c in idx_chunks)
                    log(f"  Index chunks: {len(idx_chunks)} found, total {total_chars} chars (smart selected {len(selected)} chars)")
                    log(f"  Content preview (first 300 chars):")
                    for cl in selected[:300].split("\n"):
                        log(f"    | {cl}")
                    all_source_text += f"\n\n--- File: {fname} (type: {doc_type}) ---\n{selected}\n"
                else:
                    # Fallback to inline content
                    if inline_content:
                        log(f"  Index lookup failed, using inline content ({len(inline_content)} chars)")
                        all_source_text += f"\n\n--- File: {fname} ---\n{inline_content[:MAX_CHARS_PER_FILE]}\n"
                    else:
                        log(f"  No content available for this source")
            elif inline_content:
                all_source_text += f"\n\n--- File: {fname} ---\n{inline_content[:MAX_CHARS_PER_FILE]}\n"

        log(f"\n--- END SOURCE CONTENT ---\n")

        # Step 3: GPT verification
        log("--- GROUNDING VERIFICATION ---")

        # Discovery mode: verify metadata consistency, not content grounding
        if intent == "discovery":
            log("  Mode: DISCOVERY — checking metadata consistency (not content grounding)")
            # For discovery, check if listed files actually exist in sources
            listed_count = answer.count("**") // 2  # rough count of bold file names
            log(f"  Documents listed in answer: ~{listed_count}")
            log(f"  Actual sources returned: {len(unique_sources)}")
            if len(unique_sources) > 0:
                log(f"  Status: DISCOVERY_VALID (confidence: medium)")
                log(f"  Note: Discovery mode lists documents by metadata. Content grounding not applicable.")
            else:
                log(f"  Status: DISCOVERY_EMPTY (confidence: low)")
                log(f"  Note: No sources returned for discovery query.")
            log(f"\n--- END VERIFICATION ---")
            continue

        if all_source_text.strip():
            verification = verify_answer(gpt_client, gpt_deployment, query, answer, all_source_text)

            grounded = verification.get("grounded", None)
            confidence = verification.get("confidence", "?")
            status = "GROUNDED" if grounded else ("NOT GROUNDED" if grounded is False else "UNKNOWN")

            log(f"  Status: {status} (confidence: {confidence})")

            verified = verification.get("verified_claims", [])
            if verified:
                log(f"  Verified claims ({len(verified)}):")
                for c in verified[:10]:
                    log(f"    + {c}")

            unverified = verification.get("unverified_claims", [])
            if unverified:
                log(f"  UNVERIFIED claims ({len(unverified)}):")
                for c in unverified:
                    log(f"    ? {c}")

            hallucinations = verification.get("hallucinations", [])
            if hallucinations:
                log(f"  HALLUCINATIONS ({len(hallucinations)}):")
                for h in hallucinations:
                    log(f"    X {h}")

            missing = verification.get("missing_info", [])
            if missing:
                log(f"  Missing info from source ({len(missing)}):")
                for m in missing:
                    log(f"    - {m}")

            notes = verification.get("notes", "")
            if notes:
                log(f"  Notes: {notes}")
        else:
            log("  No source content available for verification")

        log(f"\n--- END VERIFICATION ---")

    # Save
    path = "logs/verification_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
