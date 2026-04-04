"""
Check indexed internal documents in Azure AI Search
"""
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = os.environ.get("SEARCH_ENDPOINT")
key = os.environ.get("SEARCH_KEY")
index_name = os.environ.get("SEARCH_INDEX", "legal-docs-index")

client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))

# Count by source container
print("=" * 60)
print("DOCUMENTS BY SOURCE CONTAINER")
print("=" * 60)

results = client.search("*", select="source_container")
containers = {}
for r in results:
    sc = r.get("source_container", "unknown")
    containers[sc] = containers.get(sc, 0) + 1

for container, count in containers.items():
    print(f"  {container}: {count} documents")

# Show internal samples
print("\n" + "=" * 60)
print("INTERNAL DOCUMENT SAMPLES")
print("=" * 60)

# First check what source_container values exist
all_results = list(client.search("*", select="source_container", top=1000))
print(f"\nTotal documents scanned: {len(all_results)}")

# Get unique container values
unique = set(r.get("source_container", "unknown") for r in all_results)
print(f"Unique containers: {unique}")

# Try without filter to see what's there
print("\n" + "-" * 60)
print("Sample documents (any source):")
print("-" * 60)

samples = list(client.search("*", select="id, file_name, source_container", top=5))
for doc in samples:
    print(f"  📄 {doc.get('file_name', 'N/A')}")
    print(f"     Source: {doc.get('source_container', 'unknown')}")
    print(f"     ID: {doc.get('id', 'N/A')[:50]}...")
    print()
