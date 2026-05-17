"""
Data ingestion endpoint — Accept new files and add them to the vector database.
"""
import asyncio
import os
import time
import pickle
import logging
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

import config
from app.models import IngestResponse
from app.services.retrieval import retrieval_service, SBertEmbedding
from app.services.preprocessing import (
    clean_arabic_legal_text,
    preprocess_arabic_for_bm25,
    get_document_type,
    get_legal_category,
    get_legal_topic,
    extract_article_references,
)
from app.services.document_loader import load_text, load_pdf, load_docx

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _load_file_content(filepath: str, ext: str) -> str:
    if ext == ".txt":
        return load_text(filepath)
    if ext == ".pdf":
        return load_pdf(filepath)
    if ext == ".docx":
        return load_docx(filepath)
    return ""


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest New Documents",
    description=(
        "Upload new legal documents to be added to the vector database. "
        "Accepts .txt, .pdf, and .docx files. Documents are preprocessed, "
        "chunked, embedded, and merged into the existing FAISS + BM25 indices."
    ),
)
async def ingest_documents(files: List[UploadFile] = File(...)):
    t0 = time.time()

    if not retrieval_service.is_loaded:
        raise HTTPException(500, "Retrieval service not loaded. Start the server first.")

    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from rank_bm25 import BM25Okapi

    errors = []
    new_chunks = []
    files_processed = 0

    for upload_file in files:
        ext = os.path.splitext(upload_file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{upload_file.filename}: Unsupported format {ext}")
            continue

        # Save to temp file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                content = await upload_file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Extract text
            raw_text = _load_file_content(tmp_path, ext)
            os.unlink(tmp_path)

            if len(raw_text.strip()) < 50:
                errors.append(f"{upload_file.filename}: Content too short ({len(raw_text)} chars)")
                continue

            # Preprocess
            cleaned = clean_arabic_legal_text(raw_text)
            if len(cleaned) < 50:
                errors.append(f"{upload_file.filename}: Cleaned content too short")
                continue

            # Classify
            doc_type = get_document_type(upload_file.filename)
            legal_cat = get_legal_category(upload_file.filename)

            # Chunk
            chunk_cfg = config.CHUNKING_CONFIGS.get(doc_type, {"size": 2048, "overlap": 300})
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_cfg["size"],
                chunk_overlap=chunk_cfg["overlap"],
                separators=["\n\n", "\n", ".", "،", " "],
            )
            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": upload_file.filename,
                    "filename": upload_file.filename,
                    "doc_type": doc_type,
                    "legal_category": legal_cat,
                    "legal_topic": get_legal_topic(upload_file.filename),
                    "char_count": len(cleaned),
                    "ingested": True,
                }
            )
            chunks = splitter.split_documents([doc])
            # chunk_index will be set after merging into retrieval_service.chunks below.
            for c in chunks:
                c.metadata["referenced_articles"] = extract_article_references(c.page_content)
            new_chunks.extend(chunks)
            files_processed += 1
            logger.info(f"✅ {upload_file.filename}: {len(chunks)} chunks")

        except Exception as e:
            errors.append(f"{upload_file.filename}: {str(e)[:100]}")
            # Clean up temp file if it exists
            try:
                os.unlink(tmp_path)
            except:
                pass

    if not new_chunks:
        return IngestResponse(
            status="no_data",
            files_processed=0,
            errors=errors,
            duration_s=round(time.time() - t0, 1),
        )

    # ── Merge into FAISS ──
    try:
        if config.USE_REMOTE_EMBEDDINGS:
            wrapper = SBertEmbedding()
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.EMBED_MODEL_NAME, device='cpu')
            wrapper = SBertEmbedding(model)

        new_vs = FAISS.from_documents(new_chunks, wrapper)
        retrieval_service.vectorstore.merge_from(new_vs)
        retrieval_service.vectorstore.save_local(config.FAISS_INDEX_PATH)
        vectors_added = len(new_chunks)
        logger.info(f"✅ FAISS: +{vectors_added} vectors merged")
    except Exception as e:
        errors.append(f"FAISS merge failed: {str(e)[:100]}")
        vectors_added = 0

    # ── Rebuild BM25 (append chunks, rebuild index) ──
    try:
        retrieval_service.chunks.extend(new_chunks)
        new_tokens = [preprocess_arabic_for_bm25(c.page_content) for c in new_chunks]
        retrieval_service.tokenized_corpus.extend(new_tokens)
        retrieval_service.bm25 = BM25Okapi(retrieval_service.tokenized_corpus)

        # Save updated indices
        with open(config.BM25_PATH, "wb") as f:
            pickle.dump(retrieval_service.bm25, f)
        with open(config.CHUNKS_PATH, "wb") as f:
            pickle.dump(retrieval_service.chunks, f)
        with open(config.TOKENIZED_CORPUS_PATH, "wb") as f:
            pickle.dump(retrieval_service.tokenized_corpus, f)
        logger.info(f"✅ BM25: rebuilt with {len(retrieval_service.chunks)} total chunks")
    except Exception as e:
        errors.append(f"BM25 rebuild failed: {str(e)[:100]}")

    duration = time.time() - t0
    return IngestResponse(
        status="ok" if not errors else "partial",
        files_processed=files_processed,
        chunks_created=len(new_chunks),
        vectors_added=vectors_added,
        errors=errors,
        duration_s=round(duration, 1),
    )


@router.post(
    "/ingest/scan",
    response_model=IngestResponse,
    summary="Scan Inbox & Ingest New Documents",
    description=(
        "Trigger an in-process scan of the configured INGEST_INBOX_DIR. Any files not yet in "
        "the ingest registry (`data/.ingested_files.json`) are preprocessed, chunked, embedded, "
        "and merged into the FAISS + BM25 indices on disk; then the running retrieval service "
        "hot-reloads its indices so subsequent queries see the new data without a restart. "
        "Idempotent — re-running is safe."
    ),
)
async def scan_inbox():
    """Server-side trigger for the same logic the watch_ingest.py CLI runs."""
    t0 = time.time()
    if not retrieval_service.is_loaded:
        raise HTTPException(500, "Retrieval service not loaded.")

    from scripts.watch_ingest import scan_once

    try:
        result = await asyncio.to_thread(scan_once, config.INGEST_INBOX_DIR)
    except Exception as e:
        logger.exception("scan_inbox failed")
        return IngestResponse(
            status="error",
            errors=[str(e)],
            duration_s=round(time.time() - t0, 1),
        )

    # Hot-reload the running service if anything new actually made it in.
    if result.get("chunks_added", 0) > 0:
        try:
            await asyncio.to_thread(retrieval_service.reload_indices)
        except Exception as e:
            logger.warning(f"reload_indices after scan failed: {e}")
            result.setdefault("errors", []).append(f"reload_indices: {e}")

    return IngestResponse(
        status="ok" if not result.get("errors") else "partial",
        files_processed=result.get("new", 0),
        chunks_created=result.get("chunks_added", 0),
        vectors_added=result.get("chunks_added", 0),
        errors=result.get("errors", []),
        duration_s=round(time.time() - t0, 1),
    )
