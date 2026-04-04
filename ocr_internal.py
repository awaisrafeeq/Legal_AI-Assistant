"""
OCR script for internal documents container.
Uses INTERNAL_CONTAINER_SAS_URL from .env
"""
import os
import sys
import logging
from typing import Optional, Tuple
from urllib.parse import quote, urlparse

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


def split_container_sas_url(container_sas_url: str) -> Tuple[str, str, str]:
    p = urlparse(container_sas_url)
    if not p.scheme or not p.netloc:
        raise ValueError("CONTAINER_SAS_URL must be a full https URL")
    if not p.query:
        raise ValueError("CONTAINER_SAS_URL must include SAS query string")
    container_name = p.path.strip("/").split("/")[-1]
    if not container_name:
        raise ValueError("CONTAINER_SAS_URL path must end with container name")
    account_url = f"{p.scheme}://{p.netloc}"
    return account_url, container_name, p.query


def blob_url(account_url: str, container: str, sas: str, blob_name: str) -> str:
    safe = quote(blob_name, safe="/-_.~ ").replace(" ", "%20")
    return f"{account_url}/{container}/{safe}?{sas}"


def out_blob_name(output_prefix: str, in_blob_name: str) -> str:
    op = output_prefix.strip("/")
    # For internal container, add "internal-" prefix to output
    if op:
        return f"{op}/internal/{in_blob_name}.txt"
    return f"internal/{in_blob_name}.txt"


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((HttpResponseError, ResourceNotFoundError)),
    reraise=True
)
def analyze_read(di_client, document_url):
    poller = di_client.begin_analyze_document_from_url(
        "prebuilt-read", document_url, polling_interval=5
    )
    return poller.result(timeout=300)


def main():
    # Load from INTERNAL_CONTAINER_SAS_URL
    container_sas_url = os.environ.get("INTERNAL_CONTAINER_SAS_URL")
    if not container_sas_url:
        logger.error("INTERNAL_CONTAINER_SAS_URL not set in .env")
        sys.exit(1)
    
    di_endpoint = os.environ.get("DI_ENDPOINT")
    di_key = os.environ.get("DI_KEY")
    input_prefix = os.environ.get("INPUT_PREFIX", "")
    output_prefix = os.environ.get("OUTPUT_PREFIX", "extracted-text")
    max_files_str = os.environ.get("MAX_FILES", "")
    max_files = int(max_files_str) if max_files_str.strip() else None
    overwrite = os.environ.get("OVERWRITE", "0") == "1"
    
    # Parse storage URLs
    account_url, container, sas = split_container_sas_url(container_sas_url)
    logger.info(f"Processing container: {container}")
    
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)
    
    # Create separate blob service for output (main container)
    output_sas_url = os.environ.get("CONTAINER_SAS_URL")
    out_account_url, out_container, out_sas = split_container_sas_url(output_sas_url)
    output_blob_service = BlobServiceClient(account_url=out_account_url, credential=out_sas)
    
    di_client = DocumentAnalysisClient(
        endpoint=di_endpoint,
        credential=AzureKeyCredential(di_key)
    )
    
    # List PDFs
    prefix = input_prefix.strip("/") + "/" if input_prefix else ""
    pdf_blobs = []
    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        if blob.name.lower().endswith(".pdf"):
            pdf_blobs.append(blob.name)
            if max_files and len(pdf_blobs) >= max_files:
                break
    
    if not pdf_blobs:
        logger.warning("No PDF files found")
        return
    
    logger.info(f"Found {len(pdf_blobs)} PDFs to process")
    
    # Process PDFs
    success_count = 0
    for blob_name in tqdm(pdf_blobs, desc="OCR Processing"):
        try:
            doc_url = blob_url(account_url, container, sas, blob_name)
            result = analyze_read(di_client, doc_url)
            
            # Extract text
            full_text = "\n\n".join(
                page_text for page in result.pages if (page_text := " ".join(
                    line.content for line in page.lines
                ))
            )
            
            # Upload to blob storage
            out_name = out_blob_name(output_prefix, blob_name)
            out_client = output_blob_service.get_blob_client(out_container, out_name)
            
            if not overwrite and out_client.exists():
                logger.info(f"Skipping {blob_name} (already exists)")
                continue
            
            out_client.upload_blob(full_text.encode("utf-8"), overwrite=True)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {blob_name}: {e}")
            continue
    
    logger.info(f"OCR complete! Processed {success_count}/{len(pdf_blobs)} files")


if __name__ == "__main__":
    main()
