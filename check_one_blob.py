"""Quick check: does a specific blob path exist in the container?"""
import os
import sys
import unicodedata
from urllib.parse import urlparse, unquote
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

sas_url = os.environ["CONTAINER_SAS_URL"]
p = urlparse(sas_url)
account_url = f"{p.scheme}://{p.netloc}"
container = p.path.strip("/").split("/")[-1]
sas = p.query

blob_service = BlobServiceClient(account_url=account_url, credential=sas)
container_client = blob_service.get_container_client(container)

# The path from the broken URL (decoded)
test_path = unquote("Ant%C3%A9rieur/231219%20-%20Cahier%20de%20preuve%20Oribus/E%20-%20T%C3%A9moins%20%3E%20Les%20projets%20immobiliers/IV.%20Les%20investisseurs/44.%20Patrick%20MORYOUSSEF/Pi%C3%A8ces/08021.01G%20-%20170123%20Courriel%20du%203%20septembre%202015%20de%20Lucie%20Lafontaine%20%C3%A0%20Patrick%20Moryoussef/08021.01G%20-%20170123%20-%20Remise%20volontaire%20-%20Documents%20%5Bp.468-469%5D.pdf")

print(f"Test path: {test_path}")
print(f"NFC path:  {unicodedata.normalize('NFC', test_path)}")

# Try as-is
for label, path in [("as-is", test_path), ("NFC", unicodedata.normalize("NFC", test_path)), ("NFD", unicodedata.normalize("NFD", test_path))]:
    try:
        container_client.get_blob_client(path).get_blob_properties()
        print(f"\n✓ FOUND with {label}")
    except:
        print(f"\n✗ NOT FOUND with {label}")

# Search for similar blobs
search_term = "08021.01G - 170123 - Remise volontaire"
print(f"\nSearching for blobs containing: '{search_term}'")
count = 0
for b in container_client.list_blobs():
    if search_term.lower() in b.name.lower():
        print(f"  ACTUAL: {repr(b.name[:200])}")
        count += 1
        if count >= 5:
            break

if count == 0:
    # Try broader search
    search_term2 = "Remise volontaire"
    print(f"\nBroader search: '{search_term2}'")
    for b in container_client.list_blobs():
        if search_term2.lower() in b.name.lower():
            print(f"  ACTUAL: {repr(b.name[:200])}")
            count += 1
            if count >= 5:
                break
