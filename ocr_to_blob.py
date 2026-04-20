import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import quote, urlparse

from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(frozen=True)
class Config:
    container_sas_url: str
    di_endpoint: str
    di_key: str
    input_prefix: str
    output_prefix: str
    overwrite: bool
    max_files: Optional[int]


def split_container_sas_url(container_sas_url: str, container_override: str = "") -> Tuple[str, str, str]:
    p = urlparse(container_sas_url)
    if not p.scheme or not p.netloc:
        raise ValueError("CONTAINER_SAS_URL must be a full https URL")
    if not p.query:
        raise ValueError("CONTAINER_SAS_URL must include SAS query string")
    container_name = container_override or p.path.strip("/").split("/")[-1]
    if not container_name:
        raise ValueError(
            "Container name not found in SAS URL. "
            "Set CONTAINER_NAME env variable (e.g. CONTAINER_NAME=legal-documents)"
        )
    account_url = f"{p.scheme}://{p.netloc}"
    return account_url, container_name, p.query


def blob_url(account_url: str, container: str, sas: str, blob_name: str) -> str:
    safe = quote(blob_name, safe="/-_.~ ").replace(" ", "%20")
    return f"{account_url}/{container}/{safe}?{sas}"


def out_blob_name(output_prefix: str, in_blob_name: str) -> str:
    op = output_prefix.strip("/")
    return f"{op}/{in_blob_name}.txt" if op else f"{in_blob_name}.txt"


@retry(
    wait=wait_exponential(min=2, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((HttpResponseError, Exception))
)
def analyze_read(client: DocumentAnalysisClient, url_source: str) -> str:
    poller = client.begin_analyze_document_from_url(
        "prebuilt-read",
        url_source,
        polling_interval=5,
    )
    result: AnalyzeResult = poller.result(timeout=300)  # 5 minute timeout
    lines = []
    if result.content:
        return result.content
    for page in result.pages or []:
        for line in page.lines or []:
            if line.content:
                lines.append(line.content)
    return "\n".join(lines)


def load_config() -> Config:
    load_dotenv()
    return Config(
        container_sas_url=os.environ["CONTAINER_SAS_URL"],
        di_endpoint=os.environ["DI_ENDPOINT"],
        di_key=os.environ["DI_KEY"],
        input_prefix=os.environ.get("INPUT_PREFIX", "").lstrip("/"),
        output_prefix=os.environ.get("OUTPUT_PREFIX", "extracted-text").strip("/"),
        overwrite=os.environ.get("OVERWRITE", "0").lower() in {"1", "true", "yes"},
        max_files=int(os.environ["MAX_FILES"]) if os.environ.get("MAX_FILES") else None,
    )


def main() -> None:
    cfg = load_config()
    container_name = os.environ.get("CONTAINER_NAME", "")
    account_url, container, sas = split_container_sas_url(cfg.container_sas_url, container_name)

    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    di = DocumentAnalysisClient(
        cfg.di_endpoint,
        AzureKeyCredential(cfg.di_key),
        polling_interval=5,
    )

    pdfs = []
    for b in container_client.list_blobs(name_starts_with=cfg.input_prefix or None):
        name = b.name
        if name.lower().endswith(".pdf"):
            pdfs.append(name)
            if cfg.max_files and len(pdfs) >= cfg.max_files:
                break

    if not pdfs:
        print("No PDFs found. Check INPUT_PREFIX and container.")
        return

    for in_name in tqdm(pdfs, desc="OCR PDFs"):
        out_name = out_blob_name(cfg.output_prefix, in_name)
        out_client = container_client.get_blob_client(out_name)

        if not cfg.overwrite:
            try:
                out_client.get_blob_properties()
                continue
            except Exception:
                pass

        src_url = blob_url(account_url, container, sas, in_name)
        try:
            text = analyze_read(di, src_url)
        except ResourceNotFoundError as e:
            print(f"\n[ERROR] Document Intelligence returned 404 ResourceNotFound for {in_name}")
            print(f"DI_ENDPOINT={cfg.di_endpoint}")
            print("This usually indicates an endpoint/resource mismatch.")
            continue  # Skip this file and continue with next
        except HttpResponseError as e:
            if "timeout" in str(e.message).lower():
                logging.warning(f"Timeout on {in_name}, skipping after retries")
                continue  # Skip this file and continue with next
            print(f"\n[ERROR] Document Intelligence request failed for {in_name}")
            print(f"DI_ENDPOINT={cfg.di_endpoint}")
            print(f"Source blob={src_url}")
            print(f"Error: {e.message}")
            continue  # Skip this file and continue with next
        except Exception as e:
            logging.error(f"Unexpected error processing {in_name}: {e}")
            continue  # Skip this file and continue with next

        out_client.upload_blob(text.encode("utf-8"), overwrite=True)

    print("Done")


if __name__ == "__main__":
    main()
