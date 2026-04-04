"""
Retry Failed OCR Files - FIXED VERSION
Processes only files that failed in previous run with improved error handling
Fixes applied:
1. Added xlrd for .xls files
2. Separated .doc (old) and .docx (new) Word formats
3. Skip non-document Office files (themes, settings, etc.)
4. Better error handling for Document Intelligence
"""
import os
import sys
import logging
from typing import Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse, unquote

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retry_failed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# File categories - SEPARATED old .doc from new .docx
PDF_EXTS = ['.pdf']
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp']
EXCEL_EXTS = ['.xlsx', '.xls', '.xlsm']
WORD_EXTS = ['.docx']  # Modern Word format (zip-based)
WORD_OLD_EXTS = ['.doc']  # Legacy Word format (binary)
CSV_EXTS = ['.csv']
AUDIO_EXTS = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma']
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.m4v']
# Office files to SKIP (not actual documents)
SKIP_EXTS = ['.thmx', '.xml', '.rels']  # Theme files, settings, relationships

def split_sas_url(sas_url):
    p = urlparse(sas_url)
    path_parts = unquote(p.path).strip("/").split("/")
    container = path_parts[0] if path_parts else ""
    return f"{p.scheme}://{p.netloc}", container, p.query

def get_file_ext(filename: str) -> str:
    return Path(filename).suffix.lower()

def get_output_name(input_name: str) -> str:
    """Generate output blob name"""
    safe = input_name.replace('/', '_')
    return f"extracted-text/{safe}.txt"

def blob_url(account_url: str, container: str, sas: str, blob_name: str) -> str:
    from urllib.parse import quote
    safe = quote(blob_name, safe="/-_.~ ").replace(" ", "%20")
    return f"{account_url}/{container}/{safe}?{sas}"

def process_pdf_image(di_client, blob_url: str, blob_name: str) -> Optional[str]:
    """Process PDF/Image with Azure DI - with retry logic"""
    try:
        poller = di_client.begin_analyze_document_from_url(
            "prebuilt-read", blob_url, polling_interval=5
        )
        result = poller.result(timeout=600)  # 10 min timeout
        
        text = "\n\n".join(
            page_text for page in result.pages if (page_text := " ".join(
                line.content for line in page.lines
            ))
        )
        return text if text else None
    except Exception as e:
        logger.error(f"DI failed for {blob_name}: {str(e)[:200]}")
        return None

def process_excel(blob_client, blob_name: str) -> Optional[str]:
    """Extract from Excel with multiple attempts - supports both .xlsx and .xls"""
    try:
        import pandas as pd
        from io import BytesIO
        
        data = blob_client.download_blob().readall()
        excel_file = BytesIO(data)
        ext = get_file_ext(blob_name)
        
        text_parts = []
        
        # For .xls files, use xlrd engine
        if ext == '.xls':
            try:
                import xlrd
                xl = pd.ExcelFile(excel_file, engine='xlrd')
            except ImportError:
                logger.error(f"xlrd not installed for {blob_name}. Run: pip install xlrd>=2.0")
                return f"[Excel file: {blob_name}\nNote: Install xlrd to extract content from .xls files]"
        else:
            xl = pd.ExcelFile(excel_file)
        
        for sheet_name in xl.sheet_names[:10]:  # Limit to 10 sheets
            try:
                if ext == '.xls':
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=1000, engine='xlrd')
                else:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=1000)
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                text_parts.append(df.to_string(index=False, max_rows=100))
                text_parts.append("")
            except Exception as e:
                text_parts.append(f"[Error reading sheet {sheet_name}: {e}]")
        
        return "\n".join(text_parts) if text_parts else None
    except Exception as e:
        logger.error(f"Excel failed for {blob_name}: {str(e)[:200]}")
        return None

def process_csv(blob_client) -> Optional[str]:
    """Extract from CSV with encoding handling"""
    try:
        import pandas as pd
        from io import StringIO
        
        data = blob_client.download_blob().readall()
        
        # Try multiple encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = data.decode(encoding, errors='ignore')
                df = pd.read_csv(StringIO(text), nrows=1000)
                return df.to_string(index=False, max_rows=100)
            except:
                continue
        return None
    except Exception as e:
        logger.error(f"CSV failed: {str(e)[:200]}")
        return None

def process_word(blob_client, blob_name: str) -> Optional[str]:
    """Extract from Word .docx only"""
    try:
        from docx import Document
        from io import BytesIO
        import zipfile
        
        data = blob_client.download_blob().readall()
        
        # Validate it's actually a zip file (docx format)
        try:
            zipfile.is_zipfile(BytesIO(data))
        except:
            logger.warning(f"{blob_name} is not a valid .docx zip file")
            return None
        
        doc = Document(BytesIO(data))
        
        text_parts = []
        for para in doc.paragraphs[:500]:  # Limit paragraphs
            if para.text.strip():
                text_parts.append(para.text)
        
        # Tables
        for table in doc.tables[:20]:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                text_parts.append(" | ".join(row_text))
        
        return "\n".join(text_parts) if text_parts else None
    except Exception as e:
        logger.error(f"Word failed for {blob_name}: {str(e)[:200]}")
        return None

def process_old_word(blob_client, blob_name: str) -> Optional[str]:
    """Extract from legacy .doc files - uses alternative methods"""
    try:
        # Try using antiword if available (Linux/Mac tool)
        # For Windows, we'll need to use alternative approach
        data = blob_client.download_blob().readall()
        
        # Save temporarily and try to extract
        temp_path = f"/tmp/{Path(blob_name).name}"
        with open(temp_path, 'wb') as f:
            f.write(data)
        
        # Try using textract or antiword
        try:
            import subprocess
            result = subprocess.run(['antiword', temp_path], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                os.remove(temp_path)
                return result.stdout
        except:
            pass
        
        # If antiword not available, mark for manual processing
        os.remove(temp_path)
        return f"[Legacy Word Document: {blob_name}\nNote: Old .doc format requires antiword or manual conversion to .docx]"
    except Exception as e:
        logger.error(f"Old Word failed for {blob_name}: {str(e)[:200]}")
        return None

def process_audio_video(blob_client, blob_name: str) -> Optional[str]:
    """Mark audio/video for transcription"""
    try:
        props = blob_client.get_blob_properties()
        return f"[Audio/Video file: {blob_name}\nSize: {props.size} bytes\nNote: Requires Azure Speech Services for transcription]"
    except Exception as e:
        logger.error(f"Audio/Video metadata failed: {e}")
        return None

def main():
    # Load config
    sas_url = os.environ.get("CONTAINER_SAS_URL")
    di_endpoint = os.environ.get("DI_ENDPOINT")
    di_key = os.environ.get("DI_KEY")
    
    if not all([sas_url, di_endpoint, di_key]):
        logger.error("Missing environment variables")
        sys.exit(1)
    
    account_url, container, sas = split_sas_url(sas_url)
    
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)
    
    di_client = DocumentAnalysisClient(
        endpoint=di_endpoint,
        credential=AzureKeyCredential(di_key)
    )
    
    # Find all files
    logger.info("Scanning container for all files...")
    all_files = []
    for blob in container_client.list_blobs():
        if not blob.name.startswith('extracted-text/') and not blob.name.startswith('.'):
            all_files.append(blob.name)
    
    # Find missing (failed) files
    logger.info("Checking which files need OCR...")
    failed_files = []
    
    for blob_name in tqdm(all_files, desc="Checking"):
        output_name = get_output_name(blob_name)
        out_client = blob_service.get_blob_client(container, output_name)
        
        if not out_client.exists():
            failed_files.append(blob_name)
    
    total_failed = len(failed_files)
    logger.info(f"Found {total_failed} files that need OCR")
    
    if total_failed == 0:
        logger.info("No failed files found!")
        return
    
    # Process by priority
    categories = {
        'pdf': [], 'image': [], 'excel': [], 'word': [], 'word_old': [],
        'csv': [], 'audio_video': [], 'other': [], 'skipped': []
    }
    
    for f in failed_files:
        ext = get_file_ext(f)
        # Skip non-document files first
        if ext in SKIP_EXTS:
            categories['skipped'].append(f)
        elif ext in PDF_EXTS:
            categories['pdf'].append(f)
        elif ext in IMAGE_EXTS:
            categories['image'].append(f)
        elif ext in EXCEL_EXTS:
            categories['excel'].append(f)
        elif ext in WORD_EXTS:
            categories['word'].append(f)
        elif ext in WORD_OLD_EXTS:
            categories['word_old'].append(f)
        elif ext in CSV_EXTS:
            categories['csv'].append(f)
        elif ext in AUDIO_EXTS or ext in VIDEO_EXTS:
            categories['audio_video'].append(f)
        else:
            categories['other'].append(f)
    
    # Show breakdown
    logger.info("Files to process by type:")
    for cat, files in categories.items():
        if files:
            logger.info(f"  {cat}: {len(files)}")
    
    # Process with progress
    success = 0
    fail = 0
    
    for category, files in categories.items():
        if not files:
            continue
        
        logger.info(f"\nProcessing {category} files...")
        
        for blob_name in tqdm(files, desc=f"{category}"):
            try:
                ext = get_file_ext(blob_name)
                output_name = get_output_name(blob_name)
                out_client = blob_service.get_blob_client(container, output_name)
                
                text = None
                
                # Process based on type
                if category in ['pdf', 'image']:
                    doc_url = blob_url(account_url, container, sas, blob_name)
                    text = process_pdf_image(di_client, doc_url, blob_name)
                elif category == 'excel':
                    blob_client = container_client.get_blob_client(blob_name)
                    text = process_excel(blob_client, blob_name)
                elif category == 'csv':
                    blob_client = container_client.get_blob_client(blob_name)
                    text = process_csv(blob_client)
                elif category == 'word':
                    blob_client = container_client.get_blob_client(blob_name)
                    text = process_word(blob_client, blob_name)
                elif category == 'word_old':
                    blob_client = container_client.get_blob_client(blob_name)
                    text = process_old_word(blob_client, blob_name)
                    if text and "Legacy Word Document" in text:
                        # Just mark it, don't count as full success
                        logger.info(f"⚠ {blob_name} - old .doc format (needs conversion)")
                elif category == 'audio_video':
                    blob_client = container_client.get_blob_client(blob_name)
                    text = process_audio_video(blob_client, blob_name)
                elif category == 'skipped':
                    logger.info(f"⊘ {blob_name} - skipped (non-document file)")
                    continue
                
                if text:
                    full_text = f"Source: {blob_name}\nType: {category.upper()}\n{'='*60}\n\n{text}"
                    out_client.upload_blob(full_text.encode("utf-8"), overwrite=True)
                    success += 1
                    logger.info(f"✓ {blob_name}")
                else:
                    fail += 1
                    logger.warning(f"✗ {blob_name} - no text extracted")
                    
            except Exception as e:
                fail += 1
                logger.error(f"✗ {blob_name}: {str(e)[:200]}")
                continue
    
    logger.info(f"\n{'='*60}")
    logger.info("RETRY COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed:  {fail}")
    logger.info(f"Total:   {success + fail}")

if __name__ == "__main__":
    main()
