"""
Analyze OCR results and failures from process_all_formats.py
"""
import os
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse, unquote
from collections import defaultdict

def split_container_sas_url(container_sas_url: str):
    p = urlparse(str(container_sas_url))
    path_parts = unquote(p.path).strip("/").split("/")
    container_name = path_parts[0] if path_parts else ""
    account_url = f"{p.scheme}://{p.netloc}"
    return account_url, container_name, p.query

def main():
    container_sas_url = os.environ.get("CONTAINER_SAS_URL")
    account_url, container, sas = split_container_sas_url(container_sas_url)
    
    blob_service = BlobServiceClient(account_url=account_url, credential=sas)
    container_client = blob_service.get_container_client(container)
    
    # Find all files in container
    all_files = []
    extracted_files = []
    
    print("Scanning container...")
    for blob in container_client.list_blobs():
        if blob.name.startswith('extracted-text/'):
            extracted_files.append(blob.name)
        elif not blob.name.startswith('.'):
            all_files.append(blob.name)
    
    # Analyze by extension
    all_exts = defaultdict(list)
    extracted_exts = defaultdict(list)
    
    for f in all_files:
        ext = f.split('.')[-1].lower() if '.' in f else 'no_ext'
        all_exts[ext].append(f)
    
    for f in extracted_files:
        # Remove 'extracted-text/' prefix and '.txt' suffix to get original
        orig = f.replace('extracted-text/', '').replace('.txt', '')
        # Try to determine original extension
        for ext in all_exts.keys():
            if orig.endswith(ext):
                extracted_exts[ext].append(f)
                break
        else:
            extracted_exts['unknown'].append(f)
    
    print("\n" + "="*80)
    print("OCR PROCESSING RESULTS")
    print("="*80)
    
    total_input = len(all_files)
    total_output = len(extracted_files)
    failed = total_input - total_output
    
    print(f"\nTotal files in container: {total_input}")
    print(f"Successfully processed: {total_output}")
    print(f"Failed: {failed}")
    print(f"Success rate: {total_output/total_input*100:.1f}%")
    
    print("\n" + "="*80)
    print("BY FILE TYPE")
    print("="*80)
    print(f"{'Type':<15} {'Input':<10} {'OCRd':<10} {'Failed':<10} {'Success %':<10}")
    print("-"*80)
    
    all_extensions = set(all_exts.keys()) | set(extracted_exts.keys())
    
    for ext in sorted(all_extensions):
        inp = len(all_exts.get(ext, []))
        out = len(extracted_exts.get(ext, []))
        fail = inp - out
        pct = (out/inp*100) if inp > 0 else 0
        print(f"{ext:<15} {inp:<10} {out:<10} {fail:<10} {pct:<10.1f}")
    
    # Show sample OCR outputs
    print("\n" + "="*80)
    print("SAMPLE OCR OUTPUTS (Non-PDF)")
    print("="*80)
    
    categories = [
        ('xlsx', 'Excel files'),
        ('csv', 'CSV files'),
        ('docx', 'Word files'),
        ('jpg', 'Images (JPG)'),
        ('png', 'Images (PNG)'),
        ('mp3', 'Audio files'),
        ('mp4', 'Video files'),
    ]
    
    for ext, label in categories:
        print(f"\n--- {label} ({ext}) ---")
        samples = extracted_exts.get(ext, [])[:2]  # First 2 samples
        
        if not samples:
            print(f"  No {ext} files processed")
            continue
        
        for sample in samples:
            try:
                blob_client = container_client.get_blob_client(sample)
                content = blob_client.download_blob().readall().decode('utf-8', errors='ignore')
                # Show first 500 chars
                preview = content[:500].replace('\n', ' | ')
                print(f"  📄 {sample.replace('extracted-text/', '')}")
                print(f"     {preview[:200]}...")
            except Exception as e:
                print(f"  ❌ Error reading {sample}: {e}")
    
    # Find failed files
    print("\n" + "="*80)
    print("FAILED FILES (not processed)")
    print("="*80)
    
    # Files that exist in input but not in extracted-text
    processed_names = set()
    for f in extracted_files:
        # Remove extracted-text/ prefix and .txt suffix
        orig = f.replace('extracted-text/', '').replace('.txt', '')
        processed_names.add(orig)
    
    failed_files = []
    for f in all_files:
        # Check if this file was processed
        processed_name = f.replace('/', '_')  # Same logic as out_blob_name
        if processed_name not in processed_names:
            failed_files.append(f)
    
    # Group by extension
    failed_by_ext = defaultdict(list)
    for f in failed_files:
        ext = f.split('.')[-1].lower() if '.' in f else 'no_ext'
        failed_by_ext[ext].append(f)
    
    print(f"\nTotal failed: {len(failed_files)}")
    print("\nBy type:")
    for ext in sorted(failed_by_ext.keys()):
        files = failed_by_ext[ext]
        print(f"  {ext}: {len(files)} files")
        # Show first 3
        for f in files[:3]:
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... and {len(files)-3} more")

if __name__ == "__main__":
    main()
