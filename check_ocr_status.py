"""Check OCR status for large dataset"""
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

sas_url = os.environ.get('CONTAINER_SAS_URL')
account_url = sas_url.split('?')[0].replace('legal-documents', '').rstrip('/')
sas_token = sas_url.split('?')[1]

blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)
container = blob_service.get_container_client('legal-documents')

print("Scanning legal-documents container...\n")

pdf_files = []
ocr_files = []

for blob in container.list_blobs():
    if blob.name.lower().endswith('.pdf') and not blob.name.startswith('extracted-text/'):
        pdf_files.append(blob.name)
    elif blob.name.endswith('.txt') and 'extracted-text/' in blob.name:
        ocr_files.append(blob.name)

print(f"Total PDFs in root: {len(pdf_files)}")
print(f"Already OCR'd: {len(ocr_files)}")
print(f"Remaining to OCR: {len(pdf_files) - len(ocr_files)}")

if pdf_files:
    print("\nSample PDFs:")
    for f in pdf_files[:5]:
        print(f"  - {f}")
