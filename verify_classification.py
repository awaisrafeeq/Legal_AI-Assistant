"""
Verify classification coverage across both Azure AI Search indices.
Uses facets + filtered counts to avoid Azure's 100K skip limit.
"""
import os
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["SEARCH_KEY"]
EXTERNAL_INDEX = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
INTERNAL_INDEX = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")


def get_total_count(client: SearchClient) -> int:
    """Get total document count using search with count=True."""
    results = client.search(
        search_text="*",
        include_total_count=True,
        top=0
    )
    return results.get_count()


def get_classified_count(client: SearchClient) -> int:
    """Count chunks that have a non-empty document_type."""
    # Search for chunks where document_type exists and is not empty
    results = client.search(
        search_text="*",
        filter="document_type ne '' and document_type ne null",
        include_total_count=True,
        top=0
    )
    return results.get_count()


def get_type_distribution(client: SearchClient) -> dict:
    """Get document_type distribution using facets."""
    results = client.search(
        search_text="*",
        facets=["document_type,count:50"],
        top=0
    )
    facets = results.get_facets()
    if not facets or "document_type" not in facets:
        return {}
    return {f["value"]: f["count"] for f in facets["document_type"]}


def get_unclassified_sample(client: SearchClient, count: int = 20) -> list:
    """Get sample of unclassified files."""
    results = client.search(
        search_text="*",
        filter="document_type eq '' or document_type eq null",
        select=["blob_path", "file_name", "chunk_index"],
        top=count
    )
    seen = set()
    files = []
    for doc in results:
        bp = doc.get("blob_path", "")
        if bp not in seen:
            seen.add(bp)
            files.append({
                "blob_path": bp,
                "file_name": doc.get("file_name", ""),
            })
    return files


def check_index(index_name: str) -> dict:
    """Check classification coverage for an index."""
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

    print(f"\nChecking {index_name}...")

    total = get_total_count(client)
    classified = get_classified_count(client)
    unclassified = total - classified
    pct = (classified / total * 100) if total > 0 else 0

    type_dist = get_type_distribution(client)
    unclassified_sample = get_unclassified_sample(client) if unclassified > 0 else []

    print(f"\n{'='*60}")
    print(f"INDEX: {index_name}")
    print(f"{'='*60}")
    print(f"Total chunks:        {total:,}")
    print(f"Classified chunks:   {classified:,} ({pct:.1f}%)")
    print(f"Unclassified chunks: {unclassified:,} ({100-pct:.1f}%)")

    if type_dist:
        print(f"\nDocument type distribution:")
        for doc_type, count in sorted(type_dist.items(), key=lambda x: -x[1]):
            print(f"  {doc_type:30s} ->{count:,} chunks")

    if unclassified_sample:
        print(f"\nSample unclassified files:")
        for f in unclassified_sample:
            print(f"  {f['blob_path']}")

    return {
        "index": index_name,
        "total_chunks": total,
        "classified": classified,
        "unclassified": unclassified,
        "pct_classified": round(pct, 1),
        "type_distribution": type_dist,
        "unclassified_sample": unclassified_sample,
    }


def main():
    print("Verifying classification coverage...\n")

    ext_result = check_index(EXTERNAL_INDEX)
    int_result = check_index(INTERNAL_INDEX)

    # Save results
    report = {"external": ext_result, "internal": int_result}
    with open("logs/classification_verification.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"External: {ext_result['classified']:,}/{ext_result['total_chunks']:,} classified ({ext_result['pct_classified']}%)")
    print(f"Internal: {int_result['classified']:,}/{int_result['total_chunks']:,} classified ({int_result['pct_classified']}%)")
    total = ext_result['total_chunks'] + int_result['total_chunks']
    total_classified = ext_result['classified'] + int_result['classified']
    total_pct = (total_classified / total * 100) if total > 0 else 0
    print(f"Overall:  {total_classified:,}/{total:,} classified ({total_pct:.1f}%)")
    print(f"\nFull report saved to logs/classification_verification.json")


if __name__ == "__main__":
    main()
