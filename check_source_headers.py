"""
Check how many OCR text files have Source: headers vs not.
Also check if the Source: path matches an actual blob.
"""
import os
import sys
import unicodedata
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

def split_sas(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}", p.path.strip("/").split("/")[-1], p.query

def main():
    account_url, container, sas = split_sas(os.environ["CONTAINER_SAS_URL"])
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    has_header = 0
    no_header = 0
    header_valid = 0
    header_invalid = 0
    sample_no_header = []
    sample_invalid_header = []

    txt_files = []
    for b in container_client.list_blobs(name_starts_with="extracted-text/"):
        if b.name.endswith(".txt") and not b.name.startswith("extracted-text/internal/"):
            txt_files.append(b.name)

    print(f"Total OCR text files: {len(txt_files)}")

    for blob_path in tqdm(txt_files[:500], desc="Checking headers"):
        try:
            blob_client = container_client.get_blob_client(blob_path)
            first_bytes = blob_client.download_blob(offset=0, length=500).readall().decode("utf-8", errors="ignore")
            first_line = first_bytes.split("\n", 1)[0].strip()

            if first_line.startswith("Source: "):
                has_header += 1
                source_path = first_line[len("Source: "):].strip()

                # Check if this source path actually exists as a blob
                nfc_path = unicodedata.normalize("NFC", source_path)
                try:
                    container_client.get_blob_client(nfc_path).get_blob_properties()
                    header_valid += 1
                except:
                    # Try original (non-NFC)
                    try:
                        container_client.get_blob_client(source_path).get_blob_properties()
                        header_valid += 1
                    except:
                        header_invalid += 1
                        if len(sample_invalid_header) < 5:
                            sample_invalid_header.append((blob_path, source_path))
            else:
                no_header += 1
                if len(sample_no_header) < 5:
                    sample_no_header.append(blob_path)
        except Exception as e:
            pass

    print(f"\n{'='*60}")
    print(f"With Source: header:    {has_header}")
    print(f"  -> valid (blob exists):  {header_valid}")
    print(f"  -> invalid (no blob):    {header_invalid}")
    print(f"Without Source: header: {no_header}")

    if sample_no_header:
        print(f"\nSample files WITHOUT header:")
        for s in sample_no_header:
            print(f"  {s}")

    if sample_invalid_header:
        print(f"\nSample files with INVALID Source: path:")
        for blob_path, source_path in sample_invalid_header:
            print(f"  OCR: {blob_path}")
            print(f"  Source: {source_path}")
            print()

if __name__ == "__main__":
    main()
