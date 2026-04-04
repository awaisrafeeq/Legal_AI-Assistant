"""Test RAG backend source display"""
import requests
import json

# Test search with source display
response = requests.post(
    'http://localhost:8000/search',
    json={'query': '827000 Sophie Farrier Rawdon', 'top_k': 10}
)

results = response.json()
print('Search Results:')
print(f"Total: {len(results)} results\n")

for i, r in enumerate(results, 1):
    source = r.get('source_container', 'unknown')
    if source == 'internal':
        icon = '🔒'
    elif source == 'test-local-files':
        icon = '🧪'
    else:
        icon = '📄'
    print(f"{i}. {icon} {r['file_name'][:60]}...")
    print(f"   Source: {source}")
    print(f"   Score: {r['score']:.3f}")
    print()
