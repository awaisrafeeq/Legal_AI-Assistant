"""
OCR Results Analysis - Check what files were processed and view samples
"""
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse, unquote
from collections import defaultdict

# Load environment
load_dotenv()

def split_sas_url(sas_url):
    """Parse SAS URL to get account_url, container, sas_token"""
    p = urlparse(sas_url)
    account_url = f"{p.scheme}://{p.netloc}"
    path_parts = unquote(p.path).strip("/").split("/")
    container = path_parts[0] if path_parts else ""
    sas_token = p.query
    return account_url, container, sas_token

def main():
    sas_url = os.environ.get("CONTAINER_SAS_URL")
    if not sas_url:
        print("ERROR: CONTAINER_SAS_URL not found in .env")
        return
    
    print("Connecting to Azure Blob Storage...")
    account_url, container_name, sas_token = split_sas_url(sas_url)
    
    client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container = client.get_container_client(container_name)
    
    print(f"\nScanning container: {container_name}")
    print("="*80)
    
    # Categorize all files
    all_files = defaultdict(list)
    extracted_files = defaultdict(list)
    
    print("Listing all blobs...")
    for blob in container.list_blobs():
        name = blob.name
        
        if name.startswith('extracted-text/'):
            # This is an extracted OCR file
            original = name.replace('extracted-text/', '').replace('.txt', '')
            # Determine original extension
            if '.' in original:
                ext = original.rsplit('.', 1)[1].lower()
            else:
                ext = 'unknown'
            extracted_files[ext].append(name)
        elif not name.startswith('.'):
            # Original file
            if '.' in name:
                ext = name.rsplit('.', 1)[1].lower()
            else:
                ext = 'no_ext'
            all_files[ext].append(name)
    
    # Summary
    print("\n" + "="*80)
    print("OCR PROCESSING SUMMARY")
    print("="*80)
    
    total_input = sum(len(v) for v in all_files.values())
    total_output = sum(len(v) for v in extracted_files.values())
    
    print(f"\nTotal input files:  {total_input}")
    print(f"Successfully OCR'd: {total_output}")
    print(f"Failed:             {total_input - total_output}")
    print(f"Success rate:       {total_output/total_input*100:.1f}%")
    
    # By file type
    print("\n" + "="*80)
    print("BREAKDOWN BY FILE TYPE")
    print("="*80)
    print(f"{'Type':<12} {'Input':<8} {'OCRd':<8} {'Failed':<8} {'%':<6}")
    print("-"*50)
    
    all_types = set(all_files.keys()) | set(extracted_files.keys())
    
    for ext in sorted(all_types):
        inp = len(all_files.get(ext, []))
        out = len(extracted_files.get(ext, []))
        fail = inp - out
        pct = (out/inp*100) if inp > 0 else 0
        print(f"{ext:<12} {inp:<8} {out:<8} {fail:<8} {pct:<6.1f}")
    
    # Show non-PDF samples
    print("\n" + "="*80)
    print("SAMPLE OCR OUTPUTS - NON-PDF FILES")
    print("="*80)
    
    non_pdf_types = ['xlsx', 'xls', 'csv', 'docx', 'doc', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'mp3', 'mp4', 'wav', 'html', 'xml']
    
    for ext in non_pdf_types:
        samples = extracted_files.get(ext, [])
        if not samples:
            continue
        
        print(f"\n{'='*60}")
        print(f"FILE TYPE: {ext.upper()} ({len(samples)} files)")
        print(f"{'='*60}")
        
        # Show first 2 samples
        for i, sample in enumerate(samples[:2], 1):
            print(f"\n--- Sample {i}: {sample.replace('extracted-text/', '')} ---")
            try:
                blob_client = container.get_blob_client(sample)
                content = blob_client.download_blob().readall().decode('utf-8', errors='ignore')
                # Show metadata header + first 300 chars of content
                lines = content.split('\n')
                for line in lines[:5]:
                    print(f"  {line}")
                if len(content) > 500:
                    print(f"  ... [{len(content)} chars total]")
            except Exception as e:
                print(f"  ERROR reading: {e}")
    
    # Failed files
    print("\n" + "="*80)
    print("FAILED FILES (NOT OCR'D)")
    print("="*80)
    
    failed_by_type = defaultdict(list)
    
    for ext, files in all_files.items():
        processed = set(extracted_files.get(ext, []))
        for f in files:
            # Check if processed (extracted-text/original_name.txt)
            processed_name = f"extracted-text/{f.replace('/', '_')}.txt"
            if processed_name not in processed:
                failed_by_type[ext].append(f)
    
    total_failed = sum(len(v) for v in failed_by_type.values())
    print(f"\nTotal failed: {total_failed}")
    
    for ext in sorted(failed_by_type.keys()):
        files = failed_by_type[ext]
        if not files:
            continue
        print(f"\n{ext.upper()} ({len(files)} failed):")
        for f in files[:3]:
            short = f[:80] + "..." if len(f) > 80 else f
            print(f"  - {short}")
        if len(files) > 3:
            print(f"  ... and {len(files)-3} more")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
