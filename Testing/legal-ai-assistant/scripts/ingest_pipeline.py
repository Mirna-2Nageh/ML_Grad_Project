"""
Data Ingestion Pipeline — CLI Tool
═══════════════════════════════════
Automated pipeline to ingest new legal documents into the vector database.
Accepts a directory of new files, preprocesses, chunks, embeds, and merges
into the existing FAISS + BM25 indices.

Usage:
    python scripts/ingest_pipeline.py --input-dir /path/to/new/files
    python scripts/ingest_pipeline.py --input-dir /path/to/new/files --limit 5 --dry-run
"""
import os
import sys
import pickle
import argparse
import glob
import time
import logging

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
from app.services.document_loader import load_text as _load_text, load_pdf as _load_pdf, load_docx as _load_docx
from app.services.retrieval import SBertEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_text_file(path: str) -> str:
    return _load_text(path)


def load_pdf_file(path: str) -> str:
    return _load_pdf(path)


def load_docx_file(path: str) -> str:
    return _load_docx(path)


def discover_files(input_dir: str, limit: int = None):
    """Find all supported files in the input directory."""
    all_files = []
    for ext in ["**/*.txt", "**/*.pdf", "**/*.docx"]:
        all_files.extend(glob.glob(os.path.join(input_dir, ext), recursive=True))

    if limit:
        logger.info(f"Found {len(all_files)} files, limiting to {limit}")
        all_files = all_files[:limit]
    else:
        logger.info(f"Found {len(all_files)} files")

    return all_files


def process_files(file_paths, dry_run=False):
    """Load, preprocess, and chunk all files."""
    documents = []
    errors = []
    stats = {"txt": 0, "pdf": 0, "docx": 0, "empty": 0, "skipped": 0}

    for fpath in tqdm(file_paths, desc="Processing files"):
        ext = os.path.splitext(fpath)[1].lower()

        if ext == ".txt":
            raw = load_text_file(fpath)
            stats["txt"] += 1
        elif ext == ".pdf":
            raw = load_pdf_file(fpath)
            stats["pdf"] += 1
        elif ext == ".docx":
            raw = load_docx_file(fpath)
            stats["docx"] += 1
        else:
            stats["skipped"] += 1
            continue

        if len(raw.strip()) < 50:
            stats["empty"] += 1
            continue

        cleaned = clean_arabic_legal_text(raw)
        if len(cleaned) < 50:
            stats["empty"] += 1
            continue

        doc_type = get_document_type(fpath)
        legal_cat = get_legal_category(fpath)

        documents.append(Document(
            page_content=cleaned,
            metadata={
                "source": fpath,
                "filename": os.path.basename(fpath),
                "doc_type": doc_type,
                "legal_category": legal_cat,
                "legal_topic": get_legal_topic(fpath),
                "char_count": len(cleaned),
                "ingested": True,
            }
        ))

    logger.info(f"✅ Loaded {len(documents)} documents")
    logger.info(f"   Stats: {stats}")
    if errors:
        logger.warning(f"   Errors: {len(errors)}")

    if dry_run:
        logger.info("🔍 DRY RUN — Skipping chunking and indexing")
        return documents, []

    # Chunk
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        doc_type = doc.metadata.get("doc_type", "legal_reference")
        cfg = config.CHUNKING_CONFIGS.get(doc_type, {"size": 2048, "overlap": 300})
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

    logger.info(f"✅ Created {len(all_chunks):,} chunks")
    return documents, all_chunks


def merge_into_indices(new_chunks, data_dir: str):
    """Merge new chunks into existing FAISS + BM25 indices."""
    faiss_dir = os.path.join(data_dir, "faiss_index")
    bm25_path = os.path.join(data_dir, "bm25.pkl")
    chunks_path = os.path.join(data_dir, "chunks.pkl")
    tokenized_path = os.path.join(data_dir, "tokenized_corpus.pkl")

    # ── Create embedding wrapper ──
    if config.USE_REMOTE_EMBEDDINGS:
        logger.info("☁️ Using remote embeddings")
        wrapper = SBertEmbedding()
    else:
        from sentence_transformers import SentenceTransformer
        logger.info(f"💻 Loading local model: {config.EMBED_MODEL_NAME}")
        model = SentenceTransformer(config.EMBED_MODEL_NAME, device='cpu')
        wrapper = SBertEmbedding(model)

    # ── Merge FAISS ──
    logger.info("🔧 Building FAISS index for new chunks...")
    t0 = time.time()
    new_vs = FAISS.from_documents(new_chunks, wrapper)

    if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
        logger.info("   Merging into existing FAISS index...")
        existing_vs = FAISS.load_local(faiss_dir, wrapper, allow_dangerous_deserialization=True)
        existing_vs.merge_from(new_vs)
        existing_vs.save_local(faiss_dir)
        total_vectors = existing_vs.index.ntotal
    else:
        logger.info("   Creating new FAISS index...")
        new_vs.save_local(faiss_dir)
        total_vectors = new_vs.index.ntotal

    logger.info(f"   ✅ FAISS: {total_vectors} total vectors ({time.time()-t0:.1f}s)")

    # ── Update chunks ──
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
        existing_chunks.extend(new_chunks)
    else:
        existing_chunks = new_chunks

    with open(chunks_path, "wb") as f:
        pickle.dump(existing_chunks, f)

    # ── Rebuild BM25 ──
    logger.info("🔧 Rebuilding BM25 index...")
    t0 = time.time()

    if os.path.exists(tokenized_path):
        with open(tokenized_path, "rb") as f:
            tokenized_corpus = pickle.load(f)
    else:
        tokenized_corpus = []

    new_tokens = [preprocess_arabic_for_bm25(c.page_content) for c in new_chunks]
    tokenized_corpus.extend(new_tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(tokenized_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)

    logger.info(f"   ✅ BM25: {len(tokenized_corpus)} total documents ({time.time()-t0:.1f}s)")

    return total_vectors, len(existing_chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest new legal documents into the vector database")
    parser.add_argument("--input-dir", required=True, help="Directory containing new files to ingest")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="Path to existing index directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--dry-run", action="store_true", help="Process files without updating indices")
    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  كونان — Data Ingestion Pipeline")
    print("=" * 60)
    print(f"   Input:    {args.input_dir}")
    print(f"   Data dir: {args.data_dir}")
    if args.limit:
        print(f"   Limit:    {args.limit} files")
    if args.dry_run:
        print(f"   Mode:     DRY RUN")
    print()

    # Discover files
    file_paths = discover_files(args.input_dir, limit=args.limit)
    if not file_paths:
        print("❌ No supported files found. Exiting.")
        return

    # Process
    documents, chunks = process_files(file_paths, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n🔍 DRY RUN COMPLETE: Would process {len(documents)} docs → ~{len(documents)*5} est. chunks")
        return

    if not chunks:
        print("❌ No chunks generated. Exiting.")
        return

    # Merge
    total_vectors, total_chunks = merge_into_indices(chunks, args.data_dir)

    print(f"\n{'=' * 60}")
    print(f"✅ Ingestion complete!")
    print(f"   New documents: {len(documents)}")
    print(f"   New chunks:    {len(chunks)}")
    print(f"   Total vectors: {total_vectors}")
    print(f"   Total chunks:  {total_chunks}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
