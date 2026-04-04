"""
Upload 200 files directly from Google Drive to Azure Blob Storage
No local disk space required - streams directly
"""
import os
import io
from typing import List, Tuple
from urllib.parse import quote

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
GOOGLE_DRIVE_FOLDER_ID = "1Bej4fRP1igQLI0YFpZPI73GY3OffmOcZ"  # From your URL
MAX_FILES = 200
AZURE_CONTAINER_SAS_URL = os.environ["CONTAINER_SAS_URL"]


def get_drive_service():
    """Initialize Google Drive API (requires credentials.json)"""
    # You need to download service account credentials from Google Cloud Console
    # https://console.cloud.google.com/apis/credentials
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    # Place credentials.json in E:\Jeff folder
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    
    if not os.path.exists(credentials_path):
        print("ERROR: credentials.json not found!")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create Service Account → Download JSON key")
        print("3. Place as E:\\Jeff\\credentials.json")
        print("4. Enable Google Drive API: https://console.cloud.google.com/apis/library/drive.googleapis.com")
        return None
    
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES)
    
    return build('drive', 'v3', credentials=credentials)


def list_pdf_files(service, folder_id: str, max_files: int = 200) -> List[dict]:
    """List PDF files from Google Drive folder (recursive)"""
    files = []
    page_token = None
    
    print(f"Listing files from Google Drive folder...")
    
    while len(files) < max_files:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, size, parents)',
            pageToken=page_token,
            pageSize=min(100, max_files - len(files))
        ).execute()
        
        items = results.get('files', [])
        if not items:
            break
        
        files.extend(items)
        print(f"Found {len(files)} PDFs so far...")
        
        page_token = results.get('nextPageToken')
        if not page_token:
            break
    
    return files[:max_files]


def upload_to_azure_streaming(service, file_info: dict, blob_service_client: BlobServiceClient, container_name: str):
    """Stream file from Google Drive directly to Azure Blob"""
    file_id = file_info['id']
    file_name = file_info['name']
    
    try:
        # Get file from Google Drive (stream)
        request = service.files().get_media(fileId=file_id)
        
        # Create stream buffer
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                print(f"\rDownloading {file_name}: {int(status.progress() * 100)}%", end="")
        
        # Reset buffer position
        fh.seek(0)
        
        # Upload to Azure
        blob_name = f"legal-documents/{file_name}"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        blob_client.upload_blob(fh, overwrite=True)
        
        print(f"\n✓ Uploaded: {file_name}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed {file_name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Google Drive → Azure Direct Upload (200 files max)")
    print("No local storage required - streaming upload")
    print("=" * 60)
    
    # Initialize Google Drive
    drive_service = get_drive_service()
    if not drive_service:
        return
    
    # Initialize Azure
    try:
        blob_service = BlobServiceClient.from_blob_url(AZURE_CONTAINER_SAS_URL)
        # Extract container name from SAS URL
        container_name = AZURE_CONTAINER_SAS_URL.split('/')[3].split('?')[0]
        print(f"Connected to Azure container: {container_name}")
    except Exception as e:
        print(f"Azure connection failed: {e}")
        return
    
    # List files from Google Drive
    files = list_pdf_files(drive_service, GOOGLE_DRIVE_FOLDER_ID, MAX_FILES)
    
    if not files:
        print("No PDF files found in Google Drive folder!")
        return
    
    print(f"\nFound {len(files)} PDF files. Starting upload...\n")
    
    # Upload each file
    success_count = 0
    for i, file_info in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_info['name']}")
        
        if upload_to_azure_streaming(drive_service, file_info, blob_service, container_name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Upload complete: {success_count}/{len(files)} files uploaded")
    print("=" * 60)


if __name__ == "__main__":
    main()
