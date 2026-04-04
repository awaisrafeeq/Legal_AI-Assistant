"""
Quick OCR Summary and Non-PDF Samples
"""
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse, unquote
from collections import defaultdict

load_dotenv()

def split_sas_url(sas_url):
    p = urlparse(sas_url)
    return f"{p.scheme}://{p.netloc}", unquote(p.path.strip("/").split("/")[0]), p.query

def main():
    sas_url = os.environ.get("CONTAINER_SAS_URL")
    account_url, container_name, sas_token = split_sas_url(sas_url)
    
    client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container = client.get_container_client(container_name)
    
    # Count files
    all_files = defaultdict(int)
    extracted = defaultdict(int)
    
    print("Counting files...")
    for blob in container.list_blobs():
        name = blob.name
        if name.startswith('extracted-text/'):
            orig = name.replace('extracted-text/', '').replace('.txt', '')
            if '.' in orig:
                ext = orig.rsplit('.', 1)[1].lower()
            else:
                ext = 'unknown'
            extracted[ext] += 1
        elif not name.startswith('.'):
            ext = name.rsplit('.', 1)[1].lower() if '.' in name else 'no_ext'
            all_files[ext] += 1
    
    # Summary
    total_in = sum(all_files.values())
    total_out = sum(extracted.values())
    
    print("\n" + "="*60)
    print("OCR SUMMARY")
    print("="*60)
    print(f"Total files:     {total_in}")
    print(f"Successfully:    {total_out}")
    print(f"Failed:          {total_in - total_out}")
    print(f"Success rate:    {total_out/total_in*100:.1f}%")
    
    print("\n" + "="*60)
    print("BY FILE TYPE (Input → OCR'd → Failed)")
    print("="*60)
    
    for ext in sorted(set(all_files.keys()) | set(extracted.keys())):
        inp = all_files.get(ext, 0)
        out = extracted.get(ext, 0)
        fail = inp - out
        if inp > 0:
            print(f"{ext:10}: {inp:5} → {out:5} → {fail:5} ({out/inp*100:5.1f}%)")
    
    # Show non-PDF samples
    print("\n" + "="*60)
    print("NON-PDF OCR SAMPLES")
    print("="*60)
    
    sample_types = ['xlsx', 'csv', 'docx', 'jpg', 'png', 'html']
    
    for ext in sample_types:
        count = extracted.get(ext, 0)
        if count == 0:
            continue
        
        print(f"\n{'='*50}")
        print(f"{ext.upper()}: {count} files OCR'd")
        print(f"{'='*50}")
        
        # Find samples
        samples = []
        for blob in container.list_blobs(name_starts_with='extracted-text/'):
            if blob.name.endswith(f'.{ext}.txt'):
                samples.append(blob.name)
            if len(samples) >= 2:
                break
        
        for sample in samples[:1]:  # Just 1 sample per type
            print(f"\nFile: {sample.replace('extracted-text/', '')}")
            try:
                content = container.get_blob_client(sample).download_blob().readall()
                text = content.decode('utf-8', errors='ignore')[:600]
                print("-" * 40)
                print(text)
                print("-" * 40)
            except Exception as e:
                print(f"Error: {e}")
    
    # Key failures to note
    print("\n" + "="*60)
    print("IMPORTANT FAILURES TO FIX")
    print("="*60)
    
    important = ['pdf', 'xlsx', 'docx', 'csv', 'jpg', 'png']
    for ext in important:
        inp = all_files.get(ext, 0)
        out = extracted.get(ext, 0)
        fail = inp - out
        if fail > 0:
            print(f"{ext:8}: {fail:4} failed out of {inp} total")

if __name__ == "__main__":
    main()
