"""
Build FAISS + BM25 indices from the raw dataset.
Run this ONCE before starting the API server.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --dataset-dir /path/to/dataset
"""
import os
import sys
import re
import pickle
import argparse
import glob
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import config
from app.services.preprocessing import (
    clean_arabic_legal_text,
    preprocess_arabic_for_bm25,
    get_document_type,
    get_legal_category,
    get_legal_topic,
    extract_article_references,
)
from app.services.document_loader import load_text as _load_text, load_pdf as _load_pdf
from app.services.retrieval import SBertEmbedding


def load_text_file(path: str) -> str:
    return _load_text(path)


def load_pdf_file(path: str) -> str:
    return _load_pdf(path)


def load_all_documents(dataset_dir: str, limit: int = None):
    """Load all legal documents from the dataset directory."""
    documents = []
    stats = {"txt": 0, "pdf": 0, "skipped": 0, "empty": 0}

    all_files = []
    for ext in ["**/*.txt", "**/*.pdf"]:
        all_files.extend(glob.glob(os.path.join(dataset_dir, ext), recursive=True))

    if limit:
        print(f"📂 Found {len(all_files)} files. Limiting to {limit} for fast startup.")
        all_files = all_files[:limit]
    else:
        print(f"📂 Found {len(all_files)} files in {dataset_dir}")

    for fpath in tqdm(all_files, desc="Loading documents"):
        ext = os.path.splitext(fpath)[1].lower()

        if ext == '.txt':
            raw = load_text_file(fpath)
            stats["txt"] += 1
        elif ext == '.pdf':
            raw = load_pdf_file(fpath)
            stats["pdf"] += 1
        else:
            stats["skipped"] += 1
            continue

        if len(raw.strip()) < 50:
            stats["empty"] += 1
            continue

        # Clean and classify
        cleaned = clean_arabic_legal_text(raw)
        doc_type = get_document_type(fpath)
        legal_cat = get_legal_category(fpath)

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": fpath,
                "filename": os.path.basename(fpath),
                "doc_type": doc_type,
                "legal_category": legal_cat,
                "legal_topic": get_legal_topic(fpath),
                "char_count": len(cleaned),
            }
        )
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents")
    print(f"   TXT: {stats['txt']} | PDF: {stats['pdf']} | Empty: {stats['empty']} | Skipped: {stats['skipped']}")
    return documents


def chunk_documents(documents):
    """Split documents using type-aware chunking."""
    all_chunks = []

    for doc in tqdm(documents, desc="Chunking"):
        doc_type = doc.metadata.get("doc_type", "legal_reference")
        cfg = config.CHUNKING_CONFIGS.get(doc_type, {"size": 1024, "overlap": 150})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["size"],
            chunk_overlap=cfg["overlap"],
            separators=["\n\n", "\n", ".", "،", " "],
        )

        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = len(all_chunks) + i
            chunk.metadata["referenced_articles"] = extract_article_references(chunk.page_content)
        all_chunks.extend(chunks)

    print(f"✅ Created {len(all_chunks):,} chunks")
    return all_chunks


def build_indices(chunks, embed_model_name: str, output_dir: str):
    """Build FAISS and BM25 indices."""
    os.makedirs(output_dir, exist_ok=True)

    # --- FAISS ---
    print(f"\n🔧 Building FAISS index with {embed_model_name}...")
    t0 = time.time()
    
    if config.USE_REMOTE_EMBEDDINGS:
        print("   ☁️ Using remote OpenRouter embeddings API")
        wrapper = SBertEmbedding()
    else:
        print(f"   💻 Loading local SentenceTransformer on CPU (batch={config.EMBED_BATCH_SIZE}, max_seq={config.EMBED_MAX_SEQ_LENGTH})...")
        model = SentenceTransformer(embed_model_name, device='cpu')
        # Cap seq length to bound CPU attention memory — otherwise one long chunk OOMs the whole batch.
        model.max_seq_length = config.EMBED_MAX_SEQ_LENGTH
        wrapper = SBertEmbedding(model)
        
    vectorstore = FAISS.from_documents(chunks, wrapper)
    faiss_dir = os.path.join(output_dir, "faiss_index")
    vectorstore.save_local(faiss_dir)
    print(f"   ✅ FAISS built: {vectorstore.index.ntotal} vectors ({time.time()-t0:.1f}s)")

    # --- BM25 ---
    print("🔧 Building BM25 index...")
    t0 = time.time()
    tokenized_corpus = [preprocess_arabic_for_bm25(c.page_content) for c in tqdm(chunks, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   ✅ BM25 built ({time.time()-t0:.1f}s)")

    # --- Save ---
    with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    with open(os.path.join(output_dir, "tokenized_corpus.pkl"), "wb") as f:
        pickle.dump(tokenized_corpus, f)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, filenames in os.walk(output_dir)
        for f in filenames
    ) / (1024 * 1024)
    print(f"\n✅ All indices saved to {output_dir} ({total_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices from legal dataset")
    parser.add_argument("--dataset-dir", default=config.DATASET_DIR,
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", default=config.DATA_DIR,
                        help="Path to output directory")
    parser.add_argument("--embed-model", default=config.EMBED_MODEL_NAME,
                        help="Embedding model name")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of files to process for fast startup")
    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  Legal AI Assistant — Index Builder")
    print("=" * 60)
    print(f"   Dataset:   {args.dataset_dir}")
    print(f"   Output:    {args.output_dir}")
    print(f"   Embedding: {args.embed_model}")
    if args.limit:
        print(f"   Limit:     {args.limit} files")
    print()

    documents = load_all_documents(args.dataset_dir, limit=args.limit)
    chunks = chunk_documents(documents)
    build_indices(chunks, args.embed_model, args.output_dir)

    print("\n" + "=" * 60)
    print("✅ Index build complete! You can now start the server:")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
