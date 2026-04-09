"""
Data Audit Script — Scans both internal and external containers.
Classifies documents by:
  1. File extension (pdf, image, excel, etc.)
  2. Legal document type from filename patterns
  3. Legal document type from OCR content (first 500 chars)
  4. Folder structure / top-level categories

Generates a detailed report in logs/audit_report.txt
"""
import os
import sys
import re
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/audit_data.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── File extensions ───────────────────────────────────────────────────
EXT_CATEGORIES = {
    'pdf': {'.pdf'},
    'image': {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'},
    'excel': {'.xlsx', '.xls', '.xlsm'},
    'word': {'.docx', '.doc'},
    'csv': {'.csv'},
    'text': {'.txt', '.htm', '.html', '.xml', '.json', '.md'},
    'audio': {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'},
    'video': {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm'},
    'powerpoint': {'.pptx', '.ppt'},
    'archive': {'.zip', '.rar', '.7z', '.tar', '.gz'},
    'system': {'.ds_store', '.db', '.ini', '.log', '.tmp'},
}

def get_ext_category(filename: str) -> str:
    basename = Path(filename).name.lower()
    if basename in {'.ds_store', 'thumbs.db', 'desktop.ini'}:
        return 'system'
    ext = Path(filename).suffix.lower()
    for cat, exts in EXT_CATEGORIES.items():
        if ext in exts:
            return cat
    return f'other ({ext})' if ext else 'no_extension'

# ── Legal document type patterns (filename-based) ─────────────────────
# Each pattern: (regex, label)
# Ordered by specificity — first match wins
FILENAME_PATTERNS = [
    # Declarations
    (r'(?i)d[eé]claration', 'déclaration'),
    (r'(?i)affidavit', 'déclaration/affidavit'),
    (r'(?i)serment', 'déclaration/serment'),
    # Interrogation
    (r'(?i)interrogatoire', 'interrogatoire'),
    (r'(?i)contre[_\s-]?interrogatoire', 'contre-interrogatoire'),
    (r'(?i)examen\s*pr[eé]alable', 'examen préalable'),
    # Correspondence
    (r'(?i)courriel', 'courriel/email'),
    (r'(?i)e-?mail', 'courriel/email'),
    (r'(?i)correspondance', 'correspondance'),
    (r'(?i)lettre', 'lettre'),
    (r'(?i)message', 'message'),
    # Financial
    (r'(?i)facture', 'facture/invoice'),
    (r'(?i)invoice', 'facture/invoice'),
    (r'(?i)[eé]tat\s*de\s*compte', 'état de compte'),
    (r'(?i)relev[eé]', 'relevé'),
    (r'(?i)re[cç]u', 'reçu/receipt'),
    (r'(?i)ch[eè]que', 'chèque'),
    (r'(?i)bilan', 'bilan financier'),
    (r'(?i)[eé]tats?\s*financ', 'états financiers'),
    # Legal proceedings
    (r'(?i)jugement', 'jugement'),
    (r'(?i)ordonnance', 'ordonnance'),
    (r'(?i)requ[eê]te', 'requête'),
    (r'(?i)mise\s*en\s*demeure', 'mise en demeure'),
    (r'(?i)proc[eè]s[_\s-]?verbal', 'procès-verbal'),
    (r'(?i)plaidoirie', 'plaidoirie'),
    (r'(?i)d[eé]fense', 'défense'),
    (r'(?i)motion', 'motion'),
    (r'(?i)subpoena|assignation', 'assignation/subpoena'),
    # Notary
    (r'(?i)notai?re|notar', 'notaire/notarial'),
    (r'(?i)acte\s*(?:de|notari)', 'acte notarié'),
    (r'(?i)mandat', 'mandat'),
    (r'(?i)procuration', 'procuration'),
    # Property / Real estate
    (r'(?i)hypoth[eè]que', 'hypothèque/mortgage'),
    (r'(?i)mortgage', 'hypothèque/mortgage'),
    (r'(?i)[eé]valuation', 'évaluation'),
    (r'(?i)bail|lease', 'bail/lease'),
    (r'(?i)offre\s*(?:de\s*)?(?:service|achat)', 'offre'),
    (r'(?i)contrat|contract', 'contrat'),
    (r'(?i)convention', 'convention'),
    (r'(?i)cession', 'cession'),
    (r'(?i)pr[eê]t|loan|dossier\s*de\s*pr[eê]t', 'prêt/loan'),
    # Evidence / exhibits
    (r'(?i)pi[eè]ce', 'pièce/exhibit'),
    (r'(?i)exhibit', 'pièce/exhibit'),
    (r'(?i)preuve', 'preuve/evidence'),
    (r'(?i)cahier', 'cahier'),
    (r'(?i)remise\s*volontaire', 'remise volontaire'),
    (r'(?i)divulgation|disclosure', 'divulgation'),
    # Corporate
    (r'(?i)r[eé]solution', 'résolution'),
    (r'(?i)certificat', 'certificat'),
    (r'(?i)rapport', 'rapport/report'),
    (r'(?i)plan\s', 'plan'),
    (r'(?i)photo|image|scan', 'photo/scan'),
]

# Same patterns but for content-based detection (first 500 chars of OCR text)
CONTENT_PATTERNS = [
    (r'(?i)d[eé]claration\s+(?:sous\s+serment|solennelle)', 'déclaration'),
    (r'(?i)affidavit', 'déclaration/affidavit'),
    (r'(?i)interrogatoire', 'interrogatoire'),
    (r'(?i)contre[_\s-]?interrogatoire', 'contre-interrogatoire'),
    (r'(?i)courriel|e-?mail|courrier\s+[eé]lectronique', 'courriel/email'),
    (r'(?i)de\s*:\s*.*\n.*[àa]\s*:\s*', 'courriel/email'),  # De: ... À: pattern
    (r'(?i)objet\s*:\s*', 'courriel/email'),  # Objet: (subject line)
    (r'(?i)facture\s*(?:n[o°]|#|\d)', 'facture/invoice'),
    (r'(?i)[eé]tat\s*de\s*compte', 'état de compte'),
    (r'(?i)jugement|la\s+cour\s+(?:sup[eé]rieure|du\s+qu[eé]bec)', 'jugement'),
    (r'(?i)ordonnance', 'ordonnance'),
    (r'(?i)requ[eê]te', 'requête'),
    (r'(?i)acte\s+(?:de\s+)?(?:vente|hypoth[eè]que|notari)', 'acte notarié'),
    (r'(?i)notaire|par-devant\s+m[ae][iî]tre', 'notaire/notarial'),
    (r'(?i)contrat\s+(?:de\s+)?(?:pr[eê]t|vente|service|location)', 'contrat'),
    (r'(?i)bail|locataire|loyer', 'bail/lease'),
    (r'(?i)hypoth[eè]que|mortgage', 'hypothèque/mortgage'),
    (r'(?i)proc[eè]s[_\s-]?verbal', 'procès-verbal'),
    (r'(?i)mise\s+en\s+demeure', 'mise en demeure'),
    (r'(?i)[eé]valuation\s+(?:immobili[eè]re|fonci[eè]re|march[eé])', 'évaluation'),
    (r'(?i)rapport\s+(?:d\'|de\s+)', 'rapport/report'),
    (r'(?i)certificat', 'certificat'),
    (r'(?i)r[eé]solution', 'résolution'),
    (r'(?i)mandat|procuration', 'mandat/procuration'),
    (r'(?i)remise\s+volontaire', 'remise volontaire'),
    (r'(?i)divulgation', 'divulgation'),
]


def classify_by_filename(filename: str) -> list:
    """Return all matching legal doc types from filename. Can match multiple."""
    matches = []
    for pattern, label in FILENAME_PATTERNS:
        if re.search(pattern, filename):
            matches.append(label)
    return matches if matches else ['unclassified']


def classify_by_content(text: str) -> list:
    """Return all matching legal doc types from first ~500 chars of content."""
    snippet = text[:500]
    matches = []
    for pattern, label in CONTENT_PATTERNS:
        if re.search(pattern, snippet):
            matches.append(label)
    return matches if matches else ['unclassified']


def split_container_sas_url(url: str) -> Tuple[str, str, str]:
    p = urlparse(url)
    account_url = f"{p.scheme}://{p.netloc}"
    container = p.path.strip("/").split("/")[-1]
    return account_url, container, p.query


def get_top_level_folder(blob_name: str) -> str:
    """Get the first folder in the path for category grouping."""
    parts = blob_name.strip().split("/")
    if len(parts) > 1:
        return parts[0].strip()
    return "(root)"


def scan_container(sas_url: str, label: str, ocr_sas_url: Optional[str] = None,
                   ocr_prefix: str = "") -> dict:
    """
    Scan a container and return audit data.
    If ocr_sas_url is provided, reads OCR text from that container for content classification.
    """
    account_url, container, sas = split_container_sas_url(sas_url)
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)

    # OCR container (may be different from source container)
    ocr_container_client = None
    if ocr_sas_url:
        ocr_acc, ocr_cont, ocr_sas_token = split_container_sas_url(ocr_sas_url)
        ocr_service = BlobServiceClient(account_url=ocr_acc, credential=ocr_sas_token)
        ocr_container_client = ocr_service.get_container_client(ocr_cont)

    logger.info(f"Scanning {label} container: {container}")

    # Collect all blob names first
    all_blobs = []
    for blob in container_client.list_blobs():
        name = blob.name
        # Skip extracted-text output folder
        if name.startswith('extracted-text/'):
            continue
        if Path(name).name.lower() in {'.ds_store', 'thumbs.db', 'desktop.ini'}:
            continue
        all_blobs.append(name)

    logger.info(f"  Found {len(all_blobs)} files")

    # Stats
    ext_counts = Counter()
    filename_type_counts = Counter()
    content_type_counts = Counter()
    folder_counts = Counter()
    ext_by_folder = defaultdict(Counter)
    type_by_folder = defaultdict(Counter)

    # Samples: store up to 5 example filenames per type
    filename_type_samples = defaultdict(list)
    content_type_samples = defaultdict(list)

    # Track content-classified vs filename-classified mismatches
    mismatches = []

    # How many OCR texts to sample for content classification
    CONTENT_SAMPLE_LIMIT = 2000
    content_sampled = 0

    for blob_name in tqdm(all_blobs, desc=f"Auditing {label}"):
        ext_cat = get_ext_category(blob_name)
        ext_counts[ext_cat] += 1

        top_folder = get_top_level_folder(blob_name)
        folder_counts[top_folder] += 1
        ext_by_folder[top_folder][ext_cat] += 1

        # Filename-based classification
        fname = Path(blob_name).name
        fn_types = classify_by_filename(fname)
        for t in fn_types:
            filename_type_counts[t] += 1
            if len(filename_type_samples[t]) < 5:
                filename_type_samples[t].append(blob_name)
        type_by_folder[top_folder][fn_types[0]] += 1

        # Content-based classification (sample OCR texts)
        if ocr_container_client and content_sampled < CONTENT_SAMPLE_LIMIT:
            try:
                # Build OCR blob path
                ocr_blob_path = f"{ocr_prefix}/{blob_name}.txt" if ocr_prefix else f"{blob_name}.txt"
                ocr_client = ocr_container_client.get_blob_client(ocr_blob_path)
                # Read just first 600 bytes
                first_bytes = ocr_client.download_blob(offset=0, length=600).readall()
                text = first_bytes.decode("utf-8", errors="ignore")

                # Strip Source: header if present
                if text.startswith("Source: "):
                    lines = text.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("=" * 10):
                            text = "\n".join(lines[i+1:])
                            break

                ct_types = classify_by_content(text)
                for t in ct_types:
                    content_type_counts[t] += 1
                    if len(content_type_samples[t]) < 5:
                        content_type_samples[t].append(blob_name)

                # Check mismatch
                if fn_types != ['unclassified'] and ct_types != ['unclassified']:
                    if not set(fn_types) & set(ct_types):
                        if len(mismatches) < 50:
                            mismatches.append({
                                'file': blob_name,
                                'filename_types': fn_types,
                                'content_types': ct_types,
                            })

                content_sampled += 1
            except Exception:
                pass  # OCR file not found — skip silently

    return {
        'label': label,
        'container': container,
        'total_files': len(all_blobs),
        'ext_counts': ext_counts,
        'filename_type_counts': filename_type_counts,
        'content_type_counts': content_type_counts,
        'content_sampled': content_sampled,
        'folder_counts': folder_counts,
        'ext_by_folder': ext_by_folder,
        'type_by_folder': type_by_folder,
        'filename_type_samples': filename_type_samples,
        'content_type_samples': content_type_samples,
        'mismatches': mismatches,
    }


def write_report(results: list, output_path: str):
    """Write a human-readable audit report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  LEGAL DOCUMENT DATA AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")

        grand_total = sum(r['total_files'] for r in results)
        f.write(f"Total files across all containers: {grand_total:,}\n\n")

        for data in results:
            f.write("\n" + "─" * 80 + "\n")
            f.write(f"  CONTAINER: {data['label']} ({data['container']})\n")
            f.write(f"  Total files: {data['total_files']:,}\n")
            f.write("─" * 80 + "\n\n")

            # 1. File extension breakdown
            f.write("📁 FILE TYPE BREAKDOWN (by extension):\n")
            f.write("-" * 50 + "\n")
            for ext, count in sorted(data['ext_counts'].items(), key=lambda x: -x[1]):
                pct = count / data['total_files'] * 100
                bar = "█" * int(pct / 2)
                f.write(f"  {ext:20} {count:6,}  ({pct:5.1f}%) {bar}\n")
            f.write("\n")

            # 2. Legal document type (from filename)
            f.write("📋 LEGAL DOCUMENT TYPES (from filename patterns):\n")
            f.write("-" * 50 + "\n")
            for dtype, count in sorted(data['filename_type_counts'].items(), key=lambda x: -x[1]):
                pct = count / data['total_files'] * 100
                f.write(f"  {dtype:30} {count:6,}  ({pct:5.1f}%)\n")
                # Show samples
                samples = data['filename_type_samples'].get(dtype, [])
                for s in samples[:3]:
                    f.write(f"    └─ {Path(s).name}\n")
            f.write("\n")

            # 3. Legal document type (from content)
            if data['content_sampled'] > 0:
                f.write(f"📝 LEGAL DOCUMENT TYPES (from OCR content, {data['content_sampled']:,} sampled):\n")
                f.write("-" * 50 + "\n")
                for dtype, count in sorted(data['content_type_counts'].items(), key=lambda x: -x[1]):
                    pct = count / data['content_sampled'] * 100
                    f.write(f"  {dtype:30} {count:6,}  ({pct:5.1f}%)\n")
                    samples = data['content_type_samples'].get(dtype, [])
                    for s in samples[:3]:
                        f.write(f"    └─ {Path(s).name}\n")
                f.write("\n")

            # 4. Folder structure
            f.write("📂 TOP-LEVEL FOLDER BREAKDOWN:\n")
            f.write("-" * 50 + "\n")
            for folder, count in sorted(data['folder_counts'].items(), key=lambda x: -x[1]):
                pct = count / data['total_files'] * 100
                f.write(f"  {folder[:50]:50} {count:6,}  ({pct:5.1f}%)\n")
                # Show type distribution within folder
                types_in_folder = data['type_by_folder'].get(folder, {})
                top_types = sorted(types_in_folder.items(), key=lambda x: -x[1])[:5]
                for t, c in top_types:
                    f.write(f"    ├─ {t}: {c}\n")
            f.write("\n")

            # 5. Filename vs content mismatches
            if data['mismatches']:
                f.write(f"⚠️  FILENAME vs CONTENT TYPE MISMATCHES ({len(data['mismatches'])} found):\n")
                f.write("-" * 50 + "\n")
                for m in data['mismatches'][:20]:
                    f.write(f"  File: {Path(m['file']).name}\n")
                    f.write(f"    Filename says: {', '.join(m['filename_types'])}\n")
                    f.write(f"    Content says:  {', '.join(m['content_types'])}\n\n")

        # Summary / recommendations
        f.write("\n" + "=" * 80 + "\n")
        f.write("  SUMMARY & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Check how many are unclassified by filename
        for data in results:
            unclassified = data['filename_type_counts'].get('unclassified', 0)
            pct = unclassified / data['total_files'] * 100 if data['total_files'] > 0 else 0
            f.write(f"  {data['label']}:\n")
            f.write(f"    - Files classifiable by filename: {data['total_files'] - unclassified:,} / {data['total_files']:,} ({100-pct:.1f}%)\n")
            f.write(f"    - Unclassifiable by filename: {unclassified:,} ({pct:.1f}%)\n")

            if data['content_sampled'] > 0:
                content_unclass = data['content_type_counts'].get('unclassified', 0)
                content_pct = content_unclass / data['content_sampled'] * 100
                f.write(f"    - Classifiable by content (of {data['content_sampled']:,} sampled): {data['content_sampled'] - content_unclass:,} ({100-content_pct:.1f}%)\n")

            if data['mismatches']:
                f.write(f"    - Filename/content mismatches: {len(data['mismatches'])}\n")
            f.write("\n")

        f.write("  Recommendation notes:\n")
        f.write("    - If >70% classifiable by filename → rule-based classification is viable\n")
        f.write("    - If many unclassified → GPT-based content classification needed at indexing time\n")
        f.write("    - If many mismatches → filename alone is unreliable, content analysis required\n")
        f.write("    - High 'unclassified' in content → may need more patterns or GPT fallback\n")

    logger.info(f"Report written to {output_path}")


def main():
    container_sas = os.environ.get("CONTAINER_SAS_URL")
    internal_sas = os.environ.get("INTERNAL_CONTAINER_SAS_URL")

    results = []

    # External container
    if container_sas:
        logger.info("=== Scanning EXTERNAL container ===")
        ext_data = scan_container(
            sas_url=container_sas,
            label="External",
            ocr_sas_url=container_sas,      # OCR texts are in same container
            ocr_prefix="extracted-text",
        )
        results.append(ext_data)

    # Internal container
    if internal_sas:
        logger.info("=== Scanning INTERNAL container ===")
        int_data = scan_container(
            sas_url=internal_sas,
            label="Internal",
            ocr_sas_url=container_sas,       # OCR output goes to external container
            ocr_prefix="extracted-text/internal",
        )
        results.append(int_data)

    if not results:
        logger.error("No container SAS URLs found in .env")
        return

    # Write report
    report_path = "logs/audit_report.txt"
    write_report(results, report_path)

    # Also save raw data as JSON for further analysis
    json_path = "logs/audit_data.json"
    json_results = []
    for r in results:
        json_results.append({
            'label': r['label'],
            'container': r['container'],
            'total_files': r['total_files'],
            'content_sampled': r['content_sampled'],
            'ext_counts': dict(r['ext_counts']),
            'filename_type_counts': dict(r['filename_type_counts']),
            'content_type_counts': dict(r['content_type_counts']),
            'folder_counts': dict(r['folder_counts']),
            'mismatches': r['mismatches'],
        })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Raw data saved to {json_path}")

    # Print quick summary to console
    print("\n" + "=" * 60)
    print("  QUICK SUMMARY")
    print("=" * 60)
    for data in results:
        print(f"\n  {data['label']} ({data['total_files']:,} files):")
        top5_ext = sorted(data['ext_counts'].items(), key=lambda x: -x[1])[:5]
        print(f"    Top extensions: {', '.join(f'{e}={c}' for e,c in top5_ext)}")
        top5_types = sorted(data['filename_type_counts'].items(), key=lambda x: -x[1])[:5]
        print(f"    Top doc types:  {', '.join(f'{t}={c}' for t,c in top5_types)}")
        unclass = data['filename_type_counts'].get('unclassified', 0)
        print(f"    Unclassified:   {unclass:,} ({unclass/max(data['total_files'],1)*100:.1f}%)")
    print(f"\n  Full report: {report_path}")
    print(f"  Raw JSON:    {json_path}")


if __name__ == "__main__":
    main()
