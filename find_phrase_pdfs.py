import argparse
import csv
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import ContainerClient


DEFAULT_PHRASE = "Courriel investisseur"


@dataclass
class SearchConfig:
    search_endpoint: str
    search_key: str
    search_index_external: str
    search_index_internal: str
    container_sas_url: str
    internal_container_sas_url: str

    @classmethod
    def from_sources(cls, args: argparse.Namespace) -> "SearchConfig":
        search_endpoint = args.search_endpoint or os.environ.get("SEARCH_ENDPOINT", "")
        search_key = args.search_key or os.environ.get("SEARCH_KEY", "")
        search_index_external = args.search_index_external or os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")
        search_index_internal = args.search_index_internal or os.environ.get("SEARCH_INDEX_INTERNAL", "legal-docs-internal")
        container_sas_url = args.container_sas_url or os.environ.get("CONTAINER_SAS_URL", "")
        internal_container_sas_url = args.internal_container_sas_url or os.environ.get("INTERNAL_CONTAINER_SAS_URL", "")

        missing = []
        if not search_endpoint:
            missing.append("SEARCH_ENDPOINT")
        if not search_key:
            missing.append("SEARCH_KEY")

        if missing:
            joined = ", ".join(missing)
            raise SystemExit(
                "Missing Azure Search configuration: "
                f"{joined}\n"
                "Set env vars or pass CLI args, for example:\n"
                "  python find_phrase_pdfs.py --search-endpoint https://<name>.search.windows.net --search-key <key>"
            )

        return cls(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_external=search_index_external,
            search_index_internal=search_index_internal,
            container_sas_url=container_sas_url,
            internal_container_sas_url=internal_container_sas_url,
        )


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.lower().split())


def phrase_in_text(phrase: str, text: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def all_phrases_in_text(phrases: List[str], text: str) -> bool:
    normalized_haystack = normalize_text(text)
    return all(normalize_text(phrase) in normalized_haystack for phrase in phrases if phrase.strip())


def build_structured_haystack(result: Dict) -> str:
    persons = " ".join(result.get("persons", []) or [])
    organizations = " ".join(result.get("organizations", []) or [])
    projects = " ".join(result.get("projects", []) or [])
    return "\n".join(
        [
            result.get("file_name", "") or "",
            result.get("summary", "") or "",
            persons,
            organizations,
            projects,
            result.get("content", "") or "",
        ]
    )


def build_row_haystack(row: Dict[str, str], extra_text: str = "") -> str:
    return "\n".join(
        [
            row.get("file_name", "") or "",
            row.get("summary", "") or "",
            row.get("persons", "") or "",
            row.get("organizations", "") or "",
            row.get("projects", "") or "",
            extra_text or "",
        ]
    )


def resolve_container_sas_url(sas_url: str, source_container: str) -> str:
    if not sas_url:
        return ""

    parsed = urlparse(sas_url)
    parsed_path = parsed.path.strip("/")
    path_parts = [part for part in parsed_path.split("/") if part]

    if path_parts:
        return sas_url

    return f"{parsed.scheme}://{parsed.netloc}/{source_container}?{parsed.query}"


def build_blob_url(blob_path: str, source_container: str, config: SearchConfig) -> str:
    raw_sas_url = config.internal_container_sas_url if source_container == "legal-documents-internal" else config.container_sas_url
    sas_url = resolve_container_sas_url(raw_sas_url, source_container)
    if not sas_url:
        return ""

    parsed = urlparse(sas_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    parsed_path = parsed.path.strip("/")
    container_name = parsed_path.split("/")[-1] if parsed_path else source_container
    sas_token = parsed.query

    blob_file_path = blob_path or ""
    if source_container == "legal-documents-internal" and blob_file_path.startswith("extracted-text/internal/"):
        blob_file_path = blob_file_path[len("extracted-text/internal/"):]
    elif blob_file_path.startswith("extracted-text/"):
        blob_file_path = blob_file_path[len("extracted-text/"):]

    if blob_file_path.endswith(".txt"):
        blob_file_path = blob_file_path[:-4]

    if not blob_file_path:
        return ""

    return f"{base_url}/{container_name}/{quote(blob_file_path, safe='/')}?{sas_token}"


def get_container_client(source_container: str, config: SearchConfig) -> Optional[ContainerClient]:
    raw_sas_url = config.internal_container_sas_url if source_container == "legal-documents-internal" else config.container_sas_url
    sas_url = resolve_container_sas_url(raw_sas_url, source_container)
    if not sas_url:
        return None
    return ContainerClient.from_container_url(sas_url)


def read_blob_text(
    row: Dict[str, str],
    config: SearchConfig,
    container_cache: Dict[str, Optional[ContainerClient]],
    text_cache: Dict[Tuple[str, str], str],
) -> str:
    source_container = row.get("source_container", "") or ""
    blob_path = row.get("blob_path", "") or ""
    cache_key = (source_container, blob_path)
    if cache_key in text_cache:
        return text_cache[cache_key]

    if not blob_path:
        text_cache[cache_key] = ""
        return ""

    if source_container not in container_cache:
        container_cache[source_container] = get_container_client(source_container, config)

    container_client = container_cache[source_container]
    if container_client is None:
        text_cache[cache_key] = ""
        return ""

    try:
        blob_client = container_client.get_blob_client(blob_path)
        text = blob_client.download_blob().readall().decode("utf-8", errors="ignore")
    except Exception:
        text = ""

    text_cache[cache_key] = text
    return text


def extract_row(
    result: Dict,
    source_container: str,
    required_phrases: List[str],
    config: SearchConfig,
) -> Optional[Dict[str, str]]:
    file_name = result.get("file_name", "") or ""
    blob_path = result.get("blob_path", "") or ""
    folder_path = result.get("folder_path", "") or ""
    summary = result.get("summary", "") or ""

    haystack = build_structured_haystack(result)
    if not all_phrases_in_text(required_phrases, haystack):
        return None

    return {
        "source_container": result.get("source_container", source_container) or source_container,
        "file_name": file_name,
        "folder_path": folder_path,
        "blob_path": blob_path,
        "document_type": result.get("document_type", "") or "",
        "document_subtype": result.get("document_subtype", "") or "",
        "persons": ", ".join(result.get("persons", []) or []),
        "organizations": ", ".join(result.get("organizations", []) or []),
        "projects": ", ".join(result.get("projects", []) or []),
        "summary": summary.replace("\n", " ").strip(),
        "source_url": build_blob_url(blob_path, result.get("source_container", source_container) or source_container, config),
    }


def search_index(
    client: SearchClient,
    source_container: str,
    search_phrase: str,
    required_phrases: List[str],
    config: SearchConfig,
    top: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    search_text = f"\"{search_phrase}\""
    results = client.search(
        search_text=search_text,
        query_type="full",
        search_mode="all",
        select=[
            "content",
            "file_name",
            "folder_path",
            "blob_path",
            "source_container",
            "document_type",
            "document_subtype",
            "persons",
            "organizations",
            "projects",
            "summary",
        ],
        top=top,
    )

    for result in results:
        row = extract_row(dict(result), source_container, required_phrases, config)
        if row is not None:
            rows.append(row)
    return rows


def dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    best: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["source_container"], row["blob_path"], row["file_name"])
        if key not in best:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["source_container"], r["file_name"], r["blob_path"]))


def scan_local_texts(root: Path, required_phrases: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not all_phrases_in_text(required_phrases, f"{path.name}\n{text}"):
            continue

        original_name = path.name[:-4] if path.name.endswith(".txt") else path.name
        rows.append(
            {
                "source_container": "local-scan",
                "file_name": original_name,
                "folder_path": str(path.parent),
                "blob_path": str(path),
                "document_type": "",
                "document_subtype": "",
                "persons": "",
                "organizations": "",
                "projects": "",
                "summary": "",
                "source_url": "",
            }
        )
    return dedupe_rows(rows)


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "source_container",
        "file_name",
        "folder_path",
        "blob_path",
        "document_type",
        "document_subtype",
        "persons",
        "organizations",
        "projects",
        "summary",
        "source_url",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_rows(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No matching PDFs found.\n"

    lines: List[str] = [f"Found {len(rows)} matching documents.", ""]
    for idx, row in enumerate(rows, 1):
        lines.append(f"{idx}. {row['file_name']}")
        lines.append(f"   Container: {row['source_container']}")
        lines.append(f"   Blob path: {row['blob_path']}")
        if row["document_type"] or row["document_subtype"]:
            dtype = row["document_type"] or "document"
            subtype = f" ({row['document_subtype']})" if row["document_subtype"] else ""
            lines.append(f"   Type: {dtype}{subtype}")
        if row["persons"]:
            lines.append(f"   Persons: {row['persons'][:220]}")
        if row["organizations"]:
            lines.append(f"   Organizations: {row['organizations'][:220]}")
        if row["projects"]:
            lines.append(f"   Projects: {row['projects'][:220]}")
        if row["summary"]:
            lines.append(f"   Summary: {row['summary'][:220]}")
        if row["source_url"]:
            lines.append(f"   URL: {row['source_url']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_rows(rows: List[Dict[str, str]]) -> None:
    print(render_rows(rows), end="")


def write_text_report(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.write_text(render_rows(rows), encoding="utf-8")


def run_azure_search(
    search_phrase: str,
    required_phrases: List[str],
    top: int,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    config = SearchConfig.from_sources(args)
    credential = AzureKeyCredential(config.search_key)

    external_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_external,
        credential=credential,
    )
    internal_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_internal,
        credential=credential,
    )

    rows: List[Dict[str, str]] = []
    rows.extend(search_index(external_client, "legal-documents", search_phrase, required_phrases, config, top))
    rows.extend(search_index(internal_client, "legal-documents-internal", search_phrase, required_phrases, config, top))
    return dedupe_rows(rows)


def search_investor_email_index(
    client: SearchClient,
    source_container: str,
    required_phrases: List[str],
    config: SearchConfig,
    top: int,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    results = client.search(
        search_text="*",
        filter=(
            "(document_type eq 'courriel' or document_type eq 'Courriel' or document_type eq 'COURRIEL') "
            "and "
            "(document_subtype eq 'investisseur' or document_subtype eq 'Investisseur' or document_subtype eq 'INVESTISSEUR')"
        ),
        select=[
            "content",
            "file_name",
            "folder_path",
            "blob_path",
            "source_container",
            "document_type",
            "document_subtype",
            "persons",
            "organizations",
            "projects",
            "summary",
        ],
        top=top,
    )

    for result in results:
        row = extract_row(dict(result), source_container, required_phrases, config)
        if row is not None:
            rows.append(row)
    return rows


def run_investor_related_search(related_to: str, top: int, args: argparse.Namespace) -> List[Dict[str, str]]:
    config = SearchConfig.from_sources(args)
    credential = AzureKeyCredential(config.search_key)

    external_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_external,
        credential=credential,
    )
    internal_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_internal,
        credential=credential,
    )

    rows: List[Dict[str, str]] = []
    rows.extend(search_investor_email_index(external_client, "legal-documents", [related_to], config, top))
    rows.extend(search_investor_email_index(internal_client, "legal-documents-internal", [related_to], config, top))
    return dedupe_rows(rows)


def run_deep_investor_related_search(related_to: str, top: int, args: argparse.Namespace) -> List[Dict[str, str]]:
    config = SearchConfig.from_sources(args)
    credential = AzureKeyCredential(config.search_key)

    external_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_external,
        credential=credential,
    )
    internal_client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.search_index_internal,
        credential=credential,
    )

    rows: List[Dict[str, str]] = []
    rows.extend(search_investor_email_index(external_client, "legal-documents", [], config, top))
    rows.extend(search_investor_email_index(internal_client, "legal-documents-internal", [], config, top))

    deduped_rows = dedupe_rows(rows)
    target_phrases = [related_to]

    container_cache: Dict[str, Optional[ContainerClient]] = {}
    text_cache: Dict[Tuple[str, str], str] = {}
    filtered_rows: List[Dict[str, str]] = []

    for row in deduped_rows:
        full_ocr_text = read_blob_text(row, config, container_cache, text_cache)
        haystack = build_row_haystack(row, full_ocr_text)
        if all_phrases_in_text(target_phrases, haystack):
            filtered_rows.append(row)

    return filtered_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find PDFs whose indexed text contains a target phrase."
    )
    parser.add_argument(
        "--phrase",
        default=DEFAULT_PHRASE,
        help=f"Phrase to search for. Default: {DEFAULT_PHRASE!r}",
    )
    parser.add_argument(
        "--contains-all",
        nargs="+",
        default=[],
        help="Extra phrases that must also appear in the same document.",
    )
    parser.add_argument(
        "--investor-related-to",
        help="Fetch all Courriel (investisseur) documents first, then keep only those related to this person/entity.",
    )
    parser.add_argument(
        "--deep-read-investor-related-to",
        help="Fetch all Courriel (investisseur) documents, download their full OCR text blobs, then keep only those related to this person/entity.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=500,
        help="Max results to fetch from each Azure Search index.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        help="Optional local folder of OCR .txt files to scan instead of Azure Search.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--txt-out",
        type=Path,
        help="Optional readable text report output path.",
    )
    parser.add_argument(
        "--search-endpoint",
        help="Azure Search endpoint, e.g. https://<name>.search.windows.net",
    )
    parser.add_argument(
        "--search-key",
        help="Azure Search admin/query key.",
    )
    parser.add_argument(
        "--search-index-external",
        help="External Azure Search index name.",
    )
    parser.add_argument(
        "--search-index-internal",
        help="Internal Azure Search index name.",
    )
    parser.add_argument(
        "--container-sas-url",
        help="Optional SAS URL for external document container.",
    )
    parser.add_argument(
        "--internal-container-sas-url",
        help="Optional SAS URL for internal document container.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required_phrases = [args.phrase] + list(args.contains_all or [])

    if args.local_root:
        rows = scan_local_texts(args.local_root, required_phrases)
    elif args.deep_read_investor_related_to:
        rows = run_deep_investor_related_search(args.deep_read_investor_related_to, args.top, args)
    elif args.investor_related_to:
        rows = run_investor_related_search(args.investor_related_to, args.top, args)
    else:
        rows = run_azure_search(args.phrase, required_phrases, args.top, args)

    print_rows(rows)

    if args.csv_out:
        write_csv(rows, args.csv_out)
        print(f"CSV written to: {args.csv_out}")
    if args.txt_out:
        write_text_report(rows, args.txt_out)
        print(f"Text report written to: {args.txt_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
