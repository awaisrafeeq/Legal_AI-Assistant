"""
Check classification status of internal documents.
Queries Azure AI Search to find which internal documents have been classified.
"""
import os
import json
from dotenv import load_dotenv
load_dotenv()

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

search_endpoint = os.environ["SEARCH_ENDPOINT"]
search_key = os.environ["SEARCH_KEY"]
external_index = os.environ.get("SEARCH_INDEX_EXTERNAL", "legal-docs-external")

def check_external_classification():
    """Check classification status of external documents."""
    client = SearchClient(
        endpoint=search_endpoint,
        index_name=external_index,
        credential=AzureKeyCredential(search_key)
    )
    
    print(f"\n🔍 Checking external index: {external_index}")
    print("=" * 60)
    
    # Get total document count (Azure limit is 100,000 per query)
    # Use facets to get accurate count
    total_count = sum(1 for _ in client.search(search_text="*", select=["id"], top=100000))
    
    print(f"\n📊 Total documents in external index: {total_count}")
    
    # Count classified documents (have document_type field with value)
    classified_filter = "document_type ne '' and document_type ne null"
    classified_results = client.search(
        search_text="*",
        filter=classified_filter,
        select=["id", "blob_path", "document_type", "file_name"],
        top=100000
    )
    classified_docs = list(classified_results)
    classified_count = len(classified_docs)
    
    print(f"✅ Classified documents: {classified_count}")
    
    # Count unclassified documents
    unclassified_filter = "document_type eq '' or document_type eq null"
    unclassified_results = client.search(
        search_text="*",
        filter=unclassified_filter,
        select=["id", "blob_path", "file_name"],
        top=100000
    )
    unclassified_docs = list(unclassified_results)
    unclassified_count = len(unclassified_docs)
    
    print(f"❌ Unclassified documents: {unclassified_count}")
    
    # Calculate percentage
    if total_count > 0:
        percentage = (classified_count / total_count) * 100
        print(f"\n📈 Classification progress: {percentage:.1f}% ({classified_count}/{total_count})")
    
    # Show document type distribution
    if classified_count > 0:
        print(f"\n📋 Document Type Distribution:")
        print("-" * 40)
        
        type_counts = {}
        for doc in classified_docs:
            doc_type = doc.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # Sort by count descending
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        for doc_type, count in sorted_types[:15]:  # Top 15 types
            print(f"  {doc_type}: {count}")
    
    # Show some unclassified examples
    if unclassified_count > 0:
        print(f"\n🔍 Sample Unclassified Documents:")
        print("-" * 40)
        for doc in unclassified_docs[:5]:
            blob_path = doc.get("blob_path", "N/A")
            file_name = doc.get("file_name", "N/A")
            print(f"  - {file_name}")
            print(f"    Path: {blob_path[:80]}...")
    
    # Save unclassified list for retry
    if unclassified_count > 0:
        unclassified_paths = [doc.get("blob_path") for doc in unclassified_docs if doc.get("blob_path")]
        with open('logs/external_unclassified.json', 'w', encoding='utf-8') as f:
            json.dump(unclassified_paths, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Unclassified list saved to: logs/external_unclassified.json")
        print(f"   Run: python classify_documents.py --retry (after updating failed_documents.json)")
    
    print(f"\n" + "=" * 60)
    
    return {
        "total": total_count,
        "classified": classified_count,
        "unclassified": unclassified_count,
        "percentage": (classified_count / total_count * 100) if total_count > 0 else 0
    }

if __name__ == "__main__":
    try:
        stats = check_external_classification()
        
        # Summary
        print(f"\n📋 SUMMARY:")
        print(f"   Total External Documents: {stats['total']}")
        print(f"   Classified: {stats['classified']}")
        print(f"   Unclassified: {stats['unclassified']}")
        print(f"   Progress: {stats['percentage']:.1f}%")
        
        if stats['unclassified'] == 0:
            print(f"\n🎉 All external documents are classified!")
        else:
            print(f"\n⚠️  {stats['unclassified']} documents still need classification")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
