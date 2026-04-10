"""
End-to-end test: run client queries through RAG pipeline, verify answers + source links.
"""
import os
import sys
import json
import time
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

logging.basicConfig(level=logging.WARNING)

# Import RAG components directly
from rag_backend import Config, AzureClients, build_rag_graph, RAGState

TEST_QUERIES = [
    {
        "id": 1,
        "query": "Tu as plusieurs déclaration à la police et aussi un interrogatoire de Delnegro et Caon Tu peux faire une analyse micro détail des affirmation qui change ou voir si tout se qu'il dit peux être remis en question.",
        "expect_type": "answer",
        "expect_keywords": ["Delnegro", "Caon", "déclaration", "interrogatoire"],
        "expect_doc_types": ["déclaration", "interrogatoire"],
    },
    {
        "id": 2,
        "query": "est-ce que tu peux me donner tous les courriels investisseurs que Yves Blach a reçus?",
        "expect_type": "discovery",
        "expect_keywords": ["Blach", "courriel", "investisseur"],
        "expect_doc_types": ["courriel"],
    },
    {
        "id": 3,
        "query": "Je veux la date de l acte notarier des 4 lots",
        "expect_type": "answer",
        "expect_keywords": ["acte notarié", "lot", "date"],
        "expect_doc_types": ["acte_notarié"],
    },
    {
        "id": 4,
        "query": "Donne moi le courriel investisseurs ou partenaire envoyer à Solange paquette pour Saint-Charles de borome pour 240,000$.",
        "expect_type": "answer",
        "expect_keywords": ["Solange", "Paquette", "Saint-Charles", "240"],
        "expect_doc_types": ["courriel"],
    },
    {
        "id": 5,
        "query": "Peut tu trouver les courriel que j ai envoyer à l évaluateur BBD Philippe Lamarre.",
        "expect_type": "discovery",
        "expect_keywords": ["BBD", "Lamarre", "courriel"],
        "expect_doc_types": ["courriel"],
    },
    {
        "id": 6,
        "query": "Quel date nous avons reçu l avis de 60 jours , prise en paiement pour le prêt de ICC , Croll pour st-lin ?",
        "expect_type": "answer",
        "expect_keywords": ["60 jours", "ICC", "Croll", "st-lin", "prise en paiement"],
        "expect_doc_types": [],
    },
]


def verify_source_link(url: str, timeout: int = 15) -> dict:
    """Check if a source link is accessible (HEAD request)."""
    if not url:
        return {"status": "no_url", "ok": False}
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return {
            "status": resp.status_code,
            "ok": resp.status_code in (200, 206),
            "content_type": resp.headers.get("Content-Type", ""),
            "size": resp.headers.get("Content-Length", "unknown"),
        }
    except requests.exceptions.Timeout:
        return {"status": "timeout", "ok": False}
    except Exception as e:
        return {"status": f"error: {str(e)[:80]}", "ok": False}


def check_answer_quality(answer: str, expect_keywords: list) -> dict:
    """Check if answer contains expected keywords."""
    answer_lower = answer.lower()
    found = []
    missing = []
    for kw in expect_keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)
    return {
        "found": found,
        "missing": missing,
        "keyword_hit_rate": len(found) / len(expect_keywords) if expect_keywords else 1.0,
    }


def check_source_doc_types(sources: list, expect_doc_types: list) -> dict:
    """Check if source document types match expected types."""
    source_types = set()
    for s in sources:
        dt = s.get("document_type", "")
        if dt:
            source_types.add(dt.lower())

    matched = [t for t in expect_doc_types if t.lower() in source_types]
    unmatched = [t for t in expect_doc_types if t.lower() not in source_types]

    return {
        "source_types_found": list(source_types),
        "expected_matched": matched,
        "expected_missing": unmatched,
    }


def run_test(rag_graph, azure_clients, test_case: dict) -> dict:
    """Run a single test query through the RAG pipeline."""
    query = test_case["query"]
    print(f"\n{'='*70}")
    print(f"TEST #{test_case['id']}: {query[:80]}...")
    print(f"{'='*70}")

    start = time.time()

    state: RAGState = {
        "query": query,
        "conversation_history": [],
        "source_mode": "all",
        "allowed_files": []
    }

    try:
        final_state = rag_graph.invoke(state)
    except Exception as e:
        print(f"  ERROR: Pipeline failed: {e}")
        return {
            "id": test_case["id"],
            "status": "PIPELINE_ERROR",
            "error": str(e),
            "duration_s": time.time() - start,
        }

    duration = time.time() - start
    answer = final_state.get("answer", "")
    sources = final_state.get("sources", [])
    query_intent = final_state.get("query_intent", "unknown")

    print(f"\n  Intent detected: {query_intent} (expected: {test_case['expect_type']})")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Answer length: {len(answer)} chars")
    print(f"  Sources: {len(sources)} files")

    # 1. Check intent
    intent_correct = query_intent == test_case["expect_type"]
    print(f"  Intent correct: {'YES' if intent_correct else 'NO'}")

    # 2. Check answer keywords
    kw_result = check_answer_quality(answer, test_case["expect_keywords"])
    print(f"  Keywords: {kw_result['keyword_hit_rate']*100:.0f}% ({len(kw_result['found'])}/{len(test_case['expect_keywords'])})")
    if kw_result["missing"]:
        print(f"    Missing: {kw_result['missing']}")

    # 3. Check source document types
    dt_result = check_source_doc_types(sources, test_case["expect_doc_types"])
    print(f"  Source doc types found: {dt_result['source_types_found']}")
    if dt_result["expected_missing"]:
        print(f"    Missing expected types: {dt_result['expected_missing']}")

    # 4. Print answer preview
    print(f"\n  --- ANSWER PREVIEW ---")
    preview_lines = answer[:1000].split("\n")
    for line in preview_lines[:15]:
        print(f"  | {line}")
    if len(answer) > 1000:
        print(f"  | ... ({len(answer) - 1000} more chars)")

    # 5. Check source links
    print(f"\n  --- SOURCE LINKS ---")
    link_results = []
    unique_sources = []
    seen_files = set()
    for s in sources:
        fname = s.get("file_name", "")
        if fname in seen_files:
            continue
        seen_files.add(fname)
        unique_sources.append(s)

    for s in unique_sources[:10]:  # Check first 10 unique
        fname = s.get("file_name", "unknown")
        url = s.get("source_url", "")
        doc_type = s.get("document_type", "")
        link_check = verify_source_link(url)
        link_results.append(link_check)
        status_icon = "OK" if link_check["ok"] else "FAIL"
        print(f"  [{status_icon}] {fname[:60]} | type={doc_type} | HTTP {link_check['status']}")

    links_ok = sum(1 for r in link_results if r["ok"])
    links_total = len(link_results)
    print(f"\n  Links: {links_ok}/{links_total} accessible")

    # Overall verdict
    not_found = "unable to find" in answer.lower() or "could not find" in answer.lower()

    if not_found and test_case["expect_keywords"]:
        verdict = "FAIL_NO_RESULTS"
    elif kw_result["keyword_hit_rate"] >= 0.5 and links_ok > 0:
        verdict = "PASS"
    elif kw_result["keyword_hit_rate"] >= 0.25:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print(f"\n  VERDICT: {verdict}")

    return {
        "id": test_case["id"],
        "query": query[:100],
        "status": verdict,
        "intent_detected": query_intent,
        "intent_expected": test_case["expect_type"],
        "intent_correct": intent_correct,
        "duration_s": round(duration, 1),
        "answer_length": len(answer),
        "answer_preview": answer[:500],
        "sources_count": len(unique_sources),
        "keyword_hit_rate": kw_result["keyword_hit_rate"],
        "keywords_found": kw_result["found"],
        "keywords_missing": kw_result["missing"],
        "source_types": dt_result["source_types_found"],
        "links_ok": links_ok,
        "links_total": links_total,
        "not_found": not_found,
    }


def main():
    print("Initializing RAG pipeline for testing...")
    print("(This will load the cross-encoder model — may take a moment)\n")

    config = Config.from_env()
    clients = AzureClients(config)
    graph = build_rag_graph(clients)

    print("Pipeline ready. Running 6 client queries...\n")

    results = []
    for tc in TEST_QUERIES:
        result = run_test(graph, clients, tc)
        results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    for r in results:
        intent_mark = "o" if r.get("intent_correct") else "X"
        print(
            f"  #{r['id']} [{r['status']:18s}] "
            f"intent={r.get('intent_detected','?'):10s}[{intent_mark}] "
            f"kw={r.get('keyword_hit_rate',0)*100:3.0f}% "
            f"links={r.get('links_ok',0)}/{r.get('links_total',0)} "
            f"sources={r.get('sources_count',0)} "
            f"time={r.get('duration_s',0):.0f}s"
        )

    passed = sum(1 for r in results if r["status"] == "PASS")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    failed = sum(1 for r in results if "FAIL" in r["status"])

    print(f"\n  PASS: {passed} | PARTIAL: {partial} | FAIL: {failed} / {len(results)} total")

    # Save full results
    report_path = "logs/test_queries_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved to {report_path}")


if __name__ == "__main__":
    main()
