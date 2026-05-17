"""
Quick FAISS Index Builder from existing chunks.pkl
"""
import os
import sys
import pickle
import time
from langchain_community.vectorstores import FAISS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.services.retrieval import SBertEmbedding
from rank_bm25 import BM25Okapi
from app.services.preprocessing import preprocess_arabic_for_bm25
from tqdm import tqdm

class LegalChunk:
    pass

def dict_to_doc(c):
    if hasattr(c, 'page_content'):
        return c # already a Document
    if hasattr(c, 'content'):
        text = c.content
    elif hasattr(c, 'text'):
        text = c.text
    else:
        text = str(c)
    
    metadata = {}
    if hasattr(c, 'metadata'):
        metadata = c.metadata
    
    from langchain_core.documents import Document
    return Document(page_content=text, metadata=metadata)

def main():
    chunks_path = "/home/marno000onaaa/Desktop/ML_Grad_Project/chunks.pkl"
    output_dir = config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print("loading chunks...")
    import __main__
    __main__.LegalChunk = LegalChunk
    
    with open(chunks_path, "rb") as f:
        old_chunks = pickle.load(f)
        
    limit = 10000 # Safety limit for initial test
    chunks = [dict_to_doc(c) for c in old_chunks[:limit]]
    print(f"Loaded {len(chunks)} chunks (limited from {len(old_chunks)})!")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer
    print(f"Loading local SentenceTransformer model: {config.EMBED_MODEL_NAME} on {device}...")
    model = SentenceTransformer(config.EMBED_MODEL_NAME, device=device)
    wrapper = SBertEmbedding(model)
    
    print(f"Building FAISS using {config.EMBED_MODEL_NAME} on {device}...")
    t0 = time.time()
    # Explicitly use smaller batch size to avoid GPU OOM or UI lag
    vectorstore = FAISS.from_documents(chunks, wrapper)
    faiss_dir = os.path.join(output_dir, "faiss_index")
    vectorstore.save_local(faiss_dir)
    print(f"✅ FAISS built: {vectorstore.index.ntotal} vectors ({time.time()-t0:.1f}s)")
    
    print("Building BM25 index...")
    t0 = time.time()
    tokenized_corpus = [preprocess_arabic_for_bm25(c.page_content) for c in tqdm(chunks, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    with open(os.path.join(output_dir, "tokenized_corpus.pkl"), "wb") as f:
        pickle.dump(tokenized_corpus, f)
    print("✅ Done!")

if __name__ == "__main__":
    main()
