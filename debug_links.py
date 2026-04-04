"""
Debug broken source links — find actual vs indexed path differences.
"""
import os
import sys
import unicodedata
from urllib.parse import urlparse
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

def split_sas(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}", p.path.strip("/").split("/")[-1], p.query

def main():
    account_url, container, sas = split_sas(os.environ["CONTAINER_SAS_URL"])
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    search_client = SearchClient(
        endpoint=os.environ["SEARCH_ENDPOINT"],
        index_name=os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external"),
        credential=AzureKeyCredential(os.environ["SEARCH_KEY"])
    )

    results = search_client.search(search_text="*", select=["file_name", "folder_path", "blob_path"], top=100)

    checked = set()
    broken_nfc_fixed = 0
    broken_still = 0
    working = 0

    for doc in results:
        file_name = doc.get("file_name", "")
        folder_path = doc.get("folder_path", "")
        blob_path = doc.get("blob_path", "")

        if folder_path and file_name:
            reconstructed = f"{folder_path}/{file_name}"
        else:
            reconstructed = blob_path
            if reconstructed.startswith("extracted-text/"):
                reconstructed = reconstructed[len("extracted-text/"):]
            if reconstructed.endswith(".txt"):
                reconstructed = reconstructed[:-4]

        if reconstructed in checked:
            continue
        checked.add(reconstructed)

        # Try 1: as-is
        try:
            container_client.get_blob_client(reconstructed).get_blob_properties()
            working += 1
            continue
        except:
            pass

        # Try 2: NFC normalized
        nfc_path = unicodedata.normalize("NFC", reconstructed)
        try:
            container_client.get_blob_client(nfc_path).get_blob_properties()
            broken_nfc_fixed += 1
            print(f"NFC FIX WORKS: {repr(file_name[:60])}")
            continue
        except:
            pass

        # Try 3: NFD normalized
        nfd_path = unicodedata.normalize("NFD", reconstructed)
        try:
            container_client.get_blob_client(nfd_path).get_blob_properties()
            print(f"NFD FIX WORKS: {repr(file_name[:60])}")
            continue
        except:
            pass

        # Still broken — find closest match
        broken_still += 1
        search_part = file_name[-20:].replace(".pdf", "") if file_name else ""
        print(f"\nBROKEN: {repr(file_name[:80])}")
        print(f"  folder_path: {repr(folder_path[:80])}")
        print(f"  reconstructed: {repr(reconstructed[:120])}")

        if search_part and len(search_part) > 5:
            # List blobs with a rough prefix match
            prefix = folder_path[:25] if folder_path else reconstructed[:25]
            nfc_prefix = unicodedata.normalize("NFC", prefix)
            found = False
            for attempt_prefix in [prefix, nfc_prefix, ""]:
                if found:
                    break
                try:
                    for b in container_client.list_blobs(name_starts_with=attempt_prefix or None):
                        if search_part.lower() in b.name.lower():
                            print(f"  ACTUAL BLOB: {repr(b.name[:120])}")
                            # Find first difference
                            cmp = reconstructed
                            for i in range(min(len(cmp), len(b.name))):
                                if cmp[i] != b.name[i]:
                                    print(f"  DIFF at pos {i}: index=U+{ord(cmp[i]):04X} ({repr(cmp[i])}) vs blob=U+{ord(b.name[i]):04X} ({repr(b.name[i])})")
                                    print(f"  Context index: {repr(cmp[max(0,i-5):i+10])}")
                                    print(f"  Context blob:  {repr(b.name[max(0,i-5):i+10])}")
                                    break
                            found = True
                            break
                except:
                    pass

        if len(checked) >= 50:
            break

    print(f"\n{'='*60}")
    print(f"Working:          {working}")
    print(f"Fixed by NFC:     {broken_nfc_fixed}")
    print(f"Still broken:     {broken_still}")
    print(f"Total checked:    {len(checked)}")

if __name__ == "__main__":
    main()
