"""Find unclassified documents from search index for retry."""
import os
import json
from dotenv import load_dotenv
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

search_endpoint = os.environ["SEARCH_ENDPOINT"]
search_key = os.environ["SEARCH_KEY"]
external_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
internal_index = os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")

def find_unclassified(index_name):
    """Find documents without document_type field."""
    client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )
    
    unclassified = []
    # Search for docs where document_type is empty or null
    results = client.search(
        search_text="*",
        filter="document_type eq '' or document_type eq null",
        select=["id", "blob_path"],
        top=10000
    )
    
    for doc in results:
        if doc.get("blob_path"):
            unclassified.append(doc["blob_path"])
    
    return unclassified

print("Finding unclassified documents in external index...")
external_unclassified = find_unclassified(external_index)
print(f"Found {len(external_unclassified)} unclassified external documents")

print("\nFinding unclassified documents in internal index...")
internal_unclassified = find_unclassified(internal_index)
print(f"Found {len(internal_unclassified)} unclassified internal documents")

# Save to failed_documents.json for retry
all_unclassified = external_unclassified + internal_unclassified
with open('logs/failed_documents.json', 'w', encoding='utf-8') as f:
    json.dump(all_unclassified, f, ensure_ascii=False, indent=2)

print(f"\nTotal: {len(all_unclassified)} documents to retry")
print(f"Saved to logs/failed_documents.json")
print("\nRun: python classify_documents.py --retry")
