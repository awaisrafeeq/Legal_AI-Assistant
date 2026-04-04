import requests

result = requests.post('http://localhost:8000/search', 
    json={'query': 'confidentialité', 'top_k': 5})
data = result.json()

print("Search results for: confidentialité\n")
for r in data:  # data is a list, not a dict with 'results' key
    print(f"- {r['file_name']} ({r['folder_path']})")
    print(f"  Score: {r['score']:.3f}")
    print()
