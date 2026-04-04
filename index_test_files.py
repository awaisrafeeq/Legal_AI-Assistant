"""
Index local test files from e:\Jeff\Test directly to Azure AI Search
Skips blob storage - reads local files and indexes them
"""
import os
import sys
import hashlib
sys.path.insert(0, r'e:\Jeff')

from index_to_search import load_config, create_search_index_if_not_exists, generate_embeddings, chunk_text, index_documents_batch
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import logging
from tqdm import tqdm
from pathlib import Path

# Local test folder
TEST_FOLDER = r"e:\Jeff\Test"

def safe_doc_id(filename, chunk_index):
    """Create URL-safe document ID using hash"""
    # Create hash of filename
    hash_obj = hashlib.md5(filename.encode('utf-8'))
    hash_str = hash_obj.hexdigest()[:16]  # Use first 16 chars of hash
    return f"test_{hash_str}_{chunk_index}"

def index_local_files():
    cfg = load_config()
    
    # Initialize OpenAI client
    openai_client = AzureOpenAI(
        azure_endpoint=cfg.openai_endpoint,
        api_key=cfg.openai_key,
        api_version="2024-02-01",
    )
    
    # Create index if needed
    create_search_index_if_not_exists(
        cfg.search_endpoint,
        cfg.search_key,
        cfg.search_index,
        vector_dimensions=1536
    )
    
    # Find all .txt files in test folder
    test_path = Path(TEST_FOLDER)
    txt_files = list(test_path.glob("*.txt"))
    
    if not txt_files:
        logging.error(f"No .txt files found in {TEST_FOLDER}")
        return
    
    logging.info(f"Found {len(txt_files)} test files to index")
    print(f"\n{'='*60}")
    print(f"INDEXING LOCAL TEST FILES")
    print(f"{'='*60}")
    print(f"Folder: {TEST_FOLDER}")
    print(f"Files found: {len(txt_files)}")
    print(f"{'='*60}\n")
    
    # Process files
    batch_size = 16
    documents_buffer = []
    total_chunks = 0
    
    for txt_file in tqdm(txt_files, desc="Indexing test files"):
        try:
            # Read local file
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logging.warning(f"Empty content in {txt_file.name}, skipping")
                continue
            
            # Chunk the content
            chunks = chunk_text(content, cfg.max_chunk_size, cfg.chunk_overlap)
            file_total_chunks = len(chunks)
            total_chunks += file_total_chunks
            
            # Generate embeddings and create documents
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                embeddings = generate_embeddings(
                    openai_client,
                    batch_chunks,
                    cfg.openai_embedding_deployment
                )
                
                for idx, (chunk_text_content, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_index = i + idx
                    # Create safe unique ID
                    doc_id = safe_doc_id(txt_file.name, chunk_index)
                    
                    doc = {
                        "id": doc_id,
                        "content": chunk_text_content,
                        "content_vector": embedding,
                        "blob_path": f"test-files/{txt_file.name}",
                        "file_name": txt_file.name,
                        "folder_path": "test-files",
                        "chunk_index": chunk_index,
                        "total_chunks": file_total_chunks,
                        "source_container": "test-local-files",  # Tag as test files
                    }
                    documents_buffer.append(doc)
                    
                    # Upload in batches of 1000
                    if len(documents_buffer) >= 1000:
                        index_documents_batch(
                            cfg.search_endpoint,
                            cfg.search_key,
                            cfg.search_index,
                            documents_buffer
                        )
                        logging.info(f"Indexed batch of {len(documents_buffer)} documents")
                        documents_buffer = []
                        
        except Exception as e:
            logging.error(f"Error processing {txt_file.name}: {e}")
            continue
    
    # Upload remaining documents
    if documents_buffer:
        index_documents_batch(
            cfg.search_endpoint,
            cfg.search_key,
            cfg.search_index,
            documents_buffer
        )
        logging.info(f"Indexed final batch of {len(documents_buffer)} documents")
    
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE!")
    print(f"{'='*60}")
    print(f"Files indexed: {len(txt_files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Source tag: test-local-files")
    print(f"{'='*60}\n")
    
    # List indexed files
    print("Indexed files:")
    for f in txt_files:
        print(f"  ✓ {f.name}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    index_local_files()
