"""
Multi-format document processor for legal-documents container.
Handles: PDFs, Images, Audio, Video, Excel, CSV, Word, and more.
"""
import os
import sys
import io
import logging
import tempfile
from typing import Optional, Tuple, List
from urllib.parse import quote, urlparse
from pathlib import Path

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# File type categories
AUDIO_EXTS = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma']
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm']
EXCEL_EXTS = ['.xlsx', '.xls', '.xlsm']
WORD_EXTS = ['.docx', '.doc']
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp']
PDF_EXTS = ['.pdf']
CSV_EXTS = ['.csv']
TEXT_EXTS = ['.txt', '.htm', '.html', '.xml', '.json', '.md']

def split_container_sas_url(container_sas_url: str) -> Tuple[str, str, str]:
    p = urlparse(container_sas_url)
    container_name = p.path.strip("/").split("/")[-1]
    account_url = f"{p.scheme}://{p.netloc}"
    return account_url, container_name, p.query

def blob_url(account_url: str, container: str, sas: str, blob_name: str) -> str:
    safe = quote(blob_name, safe="/-_.~ ").replace(" ", "%20")
    return f"{account_url}/{container}/{safe}?{sas}"

def out_blob_name(output_prefix: str, in_blob_name: str) -> str:
    """Generate output blob name preserving path structure"""
    op = output_prefix.strip("/")
    if op:
        return f"{op}/{in_blob_name}.txt"
    return f"{in_blob_name}.txt"

def get_file_category(filename: str) -> str:
    """Determine file category based on extension"""
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTS:
        return 'pdf'
    elif ext in IMAGE_EXTS:
        return 'image'
    elif ext in AUDIO_EXTS:
        return 'audio'
    elif ext in VIDEO_EXTS:
        return 'video'
    elif ext in EXCEL_EXTS:
        return 'excel'
    elif ext in CSV_EXTS:
        return 'csv'
    elif ext in WORD_EXTS:
        return 'word'
    elif ext in TEXT_EXTS:
        return 'text'
    else:
        return 'other'

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((HttpResponseError, ResourceNotFoundError)),
    reraise=True
)
def analyze_with_di(di_client, document_url: str):
    """Analyze document using Azure Document Intelligence"""
    poller = di_client.begin_analyze_document_from_url(
        "prebuilt-read", document_url, polling_interval=5
    )
    return poller.result(timeout=300)

def process_pdf_image(di_client, blob_url: str, blob_name: str) -> Optional[str]:
    """Process PDF or Image using Azure DI"""
    try:
        result = analyze_with_di(di_client, blob_url)
        text = "\n\n".join(
            page_text for page in result.pages if (page_text := " ".join(
                line.content for line in page.lines
            ))
        )
        return text if text else f"[No text extracted from {blob_name}]"
    except Exception as e:
        logger.error(f"DI failed for {blob_name}: {e}")
        return None

def process_excel(blob_client) -> Optional[str]:
    """Extract text from Excel file"""
    try:
        import pandas as pd
        from io import BytesIO
        
        data = blob_client.download_blob().readall()
        excel_file = BytesIO(data)
        
        text_parts = []
        # Read all sheets
        xl = pd.ExcelFile(excel_file)
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            text_parts.append(f"=== Sheet: {sheet_name} ===")
            text_parts.append(df.to_string(index=False))
            text_parts.append("")
        
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Excel processing failed: {e}")
        return None

def process_csv(blob_client) -> Optional[str]:
    """Extract text from CSV file"""
    try:
        import pandas as pd
        from io import StringIO
        
        data = blob_client.download_blob().readall().decode('utf-8', errors='ignore')
        df = pd.read_csv(StringIO(data))
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"CSV processing failed: {e}")
        return None

def process_word(blob_client) -> Optional[str]:
    """Extract text from Word document"""
    try:
        from docx import Document
        from io import BytesIO
        
        data = blob_client.download_blob().readall()
        doc = Document(BytesIO(data))
        
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                text_parts.append(" | ".join(row_text))
        
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Word processing failed: {e}")
        return None

def process_text_file(blob_client) -> Optional[str]:
    """Process plain text files"""
    try:
        data = blob_client.download_blob().readall()
        # Try multiple encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return data.decode(encoding)
            except:
                continue
        return data.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Text file processing failed: {e}")
        return None

def process_audio_video(blob_client, blob_name: str) -> Optional[str]:
    """
    Placeholder for audio/video transcription.
    For now, returns metadata as transcription requires Azure Speech Services setup.
    """
    logger.info(f"Audio/Video file detected: {blob_name}")
    return f"[Audio/Video file: {blob_name}. Transcription requires Azure Speech Services. Size: {blob_client.get_blob_properties().size} bytes]"

def process_other(blob_client, blob_name: str) -> Optional[str]:
    """Handle other file types"""
    try:
        # For other files, try to extract as text or record metadata
        props = blob_client.get_blob_properties()
        return f"[File: {blob_name}, Size: {props.size} bytes, Content-Type: {props.content_settings.content_type}]"
    except Exception as e:
        return f"[File: {blob_name} - Metadata extraction failed: {e}]"

def process_file(blob_client, blob_name: str, di_client, category: str) -> Optional[str]:
    """Route file to appropriate processor based on category"""
    if category in ['pdf', 'image']:
        # Need SAS URL for DI
        return None  # Will be handled separately with URL
    elif category == 'excel':
        return process_excel(blob_client)
    elif category == 'csv':
        return process_csv(blob_client)
    elif category == 'word':
        return process_word(blob_client)
    elif category == 'text':
        return process_text_file(blob_client)
    elif category in ['audio', 'video']:
        return process_audio_video(blob_client, blob_name)
    else:
        return process_other(blob_client, blob_name)

def main():
    # Load configuration
    container_sas_url = os.environ.get("CONTAINER_SAS_URL")
    if not container_sas_url:
        logger.error("CONTAINER_SAS_URL not set")
        sys.exit(1)
    
    di_endpoint = os.environ.get("DI_ENDPOINT")
    di_key = os.environ.get("DI_KEY")
    input_prefix = os.environ.get("INPUT_PREFIX", "")
    output_prefix = os.environ.get("OUTPUT_PREFIX", "extracted-text")
    max_files_str = os.environ.get("MAX_FILES", "")
    max_files = int(max_files_str) if max_files_str.strip() else None
    overwrite = os.environ.get("OVERWRITE", "0") == "1"
    
    # Parse storage URL
    account_url, container, sas = split_container_sas_url(container_sas_url)
    logger.info(f"Processing container: {container}")
    
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)
    
    # DI client for PDFs and Images
    di_client = DocumentAnalysisClient(
        endpoint=di_endpoint,
        credential=AzureKeyCredential(di_key)
    )
    
    # List all files
    prefix = input_prefix.strip("/") + "/" if input_prefix else ""
    all_blobs = []
    
    logger.info(f"Listing all files with prefix: {prefix or '(root)'}")
    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        # Skip extracted-text folder and system files
        if blob.name.startswith('extracted-text/'):
            continue
        if blob.name.startswith('.') or '/.' in blob.name:
            continue
        all_blobs.append(blob.name)
        if max_files and len(all_blobs) >= max_files:
            break
    
    if not all_blobs:
        logger.warning("No files found")
        return
    
    logger.info(f"Found {len(all_blobs)} files to process")
    
    # Categorize files
    by_category = {}
    for blob_name in all_blobs:
        cat = get_file_category(blob_name)
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(blob_name)
    
    logger.info("File breakdown:")
    for cat, files in sorted(by_category.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {cat:10}: {len(files):5} files")
    
    # Process files
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for blob_name in tqdm(all_blobs, desc="Processing files"):
        try:
            category = get_file_category(blob_name)
            out_name = out_blob_name(output_prefix, blob_name)
            out_client = blob_service.get_blob_client(container, out_name)
            
            # Check if already exists
            if not overwrite and out_client.exists():
                skip_count += 1
                continue
            
            # Process based on category
            if category in ['pdf', 'image']:
                # Use Azure DI with URL
                doc_url = blob_url(account_url, container, sas, blob_name)
                text = process_pdf_image(di_client, doc_url, blob_name)
            else:
                # Download and process
                blob_client = container_client.get_blob_client(blob_name)
                text = process_file(blob_client, blob_name, di_client, category)
            
            if text:
                # Add metadata header
                full_text = f"Source: {blob_name}\nType: {category.upper()}\n{'='*60}\n\n{text}"
                out_client.upload_blob(full_text.encode("utf-8"), overwrite=True)
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {blob_name}: {e}")
            fail_count += 1
            continue
    
    logger.info(f"Processing complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Failed:  {fail_count}")

if __name__ == "__main__":
    main()
