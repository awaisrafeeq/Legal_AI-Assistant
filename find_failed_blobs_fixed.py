"""Find actual blob paths for failed files - fixed version."""
import os
import json
from dotenv import load_dotenv
load_dotenv()
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse

def split_sas_url(url: str):
    p = urlparse(url)
    account_url = f"{p.scheme}://{p.netloc}"
    container = p.path.strip("/").split("/")[-1]
    return account_url, container, p.query

# Load failed filenames
with open('logs/failed_documents.json', 'r', encoding='utf-8') as f:
    failed_names = json.load(f)

# Clean up failed names - remove extensions for better matching
clean_failed = {}
for name in failed_names:
    # Remove common extensions and clean
    clean = name.replace('.pdf', '').replace('.xlsx', '').replace('.docx', '').strip()
    if clean:
        clean_failed[clean] = name

print(f"Loaded {len(clean_failed)} unique failed filenames")

# Connect to blob storage
container_sas = os.environ["CONTAINER_SAS_URL"]
account_url, container, sas = split_sas_url(container_sas)
blob_service = BlobServiceClient(account_url=account_url, credential=sas)
container_client = blob_service.get_container_client(container)

# Find matching blobs - only .txt files from extracted-text
external_matches = []
internal_matches = []

print("Scanning blobs...")
for blob in container_client.list_blobs():
    # Only look at OCR text files
    if not blob.name.endswith('.txt'):
        continue
    if not (blob.name.startswith('extracted-text/') or '/extracted-text/' in blob.name):
        continue
    
    full_path = blob.name
    # Get base filename without .txt extension
    base_name = full_path.replace('.txt', '')
    
    # Check for exact or close match
    for clean_name, original_name in clean_failed.items():
        # Check if clean name appears in blob path
        if clean_name in base_name or original_name in full_path:
            if 'internal' in full_path.lower() or '/internal/' in full_path:
                internal_matches.append(full_path)
            else:
                external_matches.append(full_path)
            break

print(f"Found {len(external_matches)} external matches")
print(f"Found {len(internal_matches)} internal matches")

# Save combined list
all_matches = sorted(set(external_matches + internal_matches))
with open('logs/failed_documents.json', 'w', encoding='utf-8') as f:
    json.dump(all_matches, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(all_matches)} full blob paths to failed_documents.json")
print("First 3 external:", external_matches[:3] if external_matches else "None")
print("First 3 internal:", internal_matches[:3] if internal_matches else "None")
