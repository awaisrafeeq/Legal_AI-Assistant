"""
Detailed test: run queries, show full answers + all source links + verify document content.
Saves a readable report to logs/test_detailed_report.txt
"""
import os
import sys
import json
import time
import requests
import logging
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
logging.basicConfig(level=logging.WARNING)

from rag_backend import Config, AzureClients, build_rag_graph, RAGState

TEST_QUERIES = [
    "Tu as plusieurs déclaration à la police et aussi un interrogatoire de Delnegro et Caon Tu peux faire une analyse micro détail des affirmation qui change ou voir si tout se qu'il dit peux être remis en question.",
    "est-ce que tu peux me donner tous les courriels investisseurs que Yves Blach a reçus?",
    "Je veux la date de l acte notarier des 4 lots",
    "Donne moi le courriel investisseurs ou partenaire envoyer à Solange paquette pour Saint-Charles de borome pour 240,000$.",
    "Peut tu trouver les courriel que j ai envoyer à l évaluateur BBD Philippe Lamarre.",
    "Quel date nous avons reçu l avis de 60 jours , prise en paiement pour le prêt de ICC , Croll pour st-lin ?",
]


def verify_link(url, timeout=15):
    if not url:
        return "NO_URL", 0
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        size = resp.headers.get("Content-Length", "?")
        return resp.status_code, size
    except:
        return "ERROR", 0


def main():
    print("Loading RAG pipeline...")
    config = Config.from_env()
    clients = AzureClients(config)
    graph = build_rag_graph(clients)
    print("Ready.\n")

    report_lines = []

    def log(text=""):
        print(text)
        report_lines.append(text)

    for i, query in enumerate(TEST_QUERIES, 1):
        log(f"\n{'#'*80}")
        log(f"# TEST {i}")
        log(f"{'#'*80}")
        log(f"QUERY: {query}")
        log("")

        start = time.time()
        state: RAGState = {
            "query": query,
            "conversation_history": [],
            "source_mode": "all",
            "allowed_files": []
        }

        try:
            final = graph.invoke(state)
        except Exception as e:
            log(f"ERROR: {e}")
            continue

        duration = time.time() - start
        answer = final.get("answer", "")
        sources = final.get("sources", [])
        intent = final.get("query_intent", "unknown")
        filters = final.get("discovery_filters", {})

        log(f"INTENT: {intent}")
        if filters:
            log(f"FILTERS: {json.dumps(filters, ensure_ascii=False)}")
        log(f"DURATION: {duration:.1f}s")
        log(f"SOURCES COUNT: {len(sources)}")
        log("")

        # Full answer
        log("--- FULL ANSWER ---")
        for line in answer.split("\n"):
            log(f"  {line}")
        log("--- END ANSWER ---")
        log("")

        # All sources with links and verification
        log("--- SOURCE FILES & LINKS ---")
        seen = set()
        source_num = 0
        for s in sources:
            fname = s.get("file_name", "unknown")
            if fname in seen:
                continue
            seen.add(fname)
            source_num += 1

            url = s.get("source_url", "")
            doc_type = s.get("document_type", "")
            doc_subtype = s.get("document_subtype", "")
            folder = s.get("folder_path", "")
            container = s.get("source_container", "")
            persons = s.get("persons", [])
            summary = s.get("summary", "")
            blob_path = s.get("blob_path", "")

            # Verify link
            status, size = verify_link(url)
            link_ok = "OK" if status in (200, 206) else f"FAIL({status})"

            icon = "INTERNAL" if container == "legal-documents-internal" else "EXTERNAL"

            log(f"  {source_num}. [{icon}] {fname}")
            log(f"     Type: {doc_type} / {doc_subtype}")
            if folder:
                log(f"     Folder: {folder}")
            if persons:
                log(f"     Persons: {', '.join(persons[:8])}")
            if summary:
                log(f"     Summary: {summary[:200]}")
            log(f"     Link [{link_ok}]: {url}")
            log(f"     Blob: {blob_path}")
            log("")

        log(f"--- END SOURCES ({source_num} unique files) ---")
        log("")

    # Save report
    report_path = "logs/test_detailed_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
