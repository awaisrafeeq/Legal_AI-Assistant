"""Scan all file types in container"""
import os
from azure.storage.blob import BlobServiceClient
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

sas_url = os.environ.get('CONTAINER_SAS_URL')
account_url = sas_url.split('?')[0].replace('legal-documents', '').rstrip('/')
sas_token = sas_url.split('?')[1]

blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)
container = blob_service.get_container_client('legal-documents')

print("Scanning all files in legal-documents container...\n")

extensions = Counter()
file_types = {
    'pdf': [],
    'audio': [],
    'video': [],
    'excel': [],
    'csv': [],
    'word': [],
    'images': [],
    'other': []
}

audio_exts = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma']
video_exts = ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm']
excel_exts = ['.xlsx', '.xls', '.xlsm']
word_exts = ['.docx', '.doc']
image_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']

for blob in container.list_blobs():
    if blob.name.startswith('extracted-text/'):
        continue
        
    name_lower = blob.name.lower()
    ext = os.path.splitext(name_lower)[1]
    extensions[ext] += 1
    
    if ext == '.pdf':
        file_types['pdf'].append(blob.name)
    elif ext in audio_exts:
        file_types['audio'].append(blob.name)
    elif ext in video_exts:
        file_types['video'].append(blob.name)
    elif ext in excel_exts:
        file_types['excel'].append(blob.name)
    elif ext == '.csv':
        file_types['csv'].append(blob.name)
    elif ext in word_exts:
        file_types['word'].append(blob.name)
    elif ext in image_exts:
        file_types['images'].append(blob.name)
    else:
        file_types['other'].append(blob.name)

print("=" * 60)
print("FILE TYPE SUMMARY")
print("=" * 60)
for ext, count in sorted(extensions.items(), key=lambda x: -x[1])[:15]:
    print(f"  {ext:10} : {count:6} files")

print("\n" + "=" * 60)
print("CATEGORY SUMMARY")
print("=" * 60)
for cat, files in file_types.items():
    if files:
        print(f"\n{cat.upper()}: {len(files)} files")
        for f in files[:3]:
            print(f"  - {f[:60]}...")
        if len(files) > 3:
            print(f"  ... and {len(files)-3} more")

print(f"\nTOTAL: {sum(len(v) for v in file_types.values())} files")
