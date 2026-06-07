"""
Article-aware chunking for Arabic legal documents.

Statute-like documents (penal code, criminal procedure, legal references, rules
collections) are split on article boundaries (المادة N) so each article stays
intact in a single chunk — critical for a citation-driven RAG system where every
cited article must trace back to one retrievable unit. Narrative documents (case
files, cassation rulings, encyclopedia entries) keep recursive character
splitting, since they have no clean article structure to exploit.

Oversized article segments are sub-split with the recursive splitter (carrying the
article number forward). Every emitted chunk carries the metadata the retrieval /
scoring pipeline depends on: chunk_index, referenced_articles, legal_topic, and a
new primary_article (the article this chunk *is*, not just one it references).

Single source of truth for chunking: both scripts/build_index.py and
scripts/ingest_pipeline.py call chunk_document() so the two index-writers can never
drift apart (CLAUDE.md: the four data/ files must stay coherent).
"""
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from app.services.preprocessing import extract_article_references, normalize_arabic_indic_digits

# doc_types whose body is article-numbered statute text worth splitting on المادة.
# Case files / cassation rulings / encyclopedia entries are narrative and keep the
# recursive character splitter (no reliable article scaffold to cut on).
ARTICLE_STRUCTURED_TYPES = {
    "penal_code",
    "criminal_procedure",
    "criminal_law_reference",
    "legal_reference",
    "legal_rules_collection",
}

# An article header at a segment start: "المادة 230", "مادة رقم ٣١٦", "المادة (240)".
# Digits are already Western here because clean_arabic_legal_text() normalized
# Arabic-Indic digits during loading. ة is preserved (NORMALIZE_TA_MARBUTA=False),
# so we match the ة form; the ه fallback covers a marbuta-normalized corpus.
_ARTICLE_HEADER = re.compile(
    r'(?:^|\n)\s*(?:المادة|الماده|مادة|ماده)\s*(?:رقم\s*)?\(?\s*(\d{1,4})\s*\)?'
)

# A sub-split is only triggered when a single article overruns the doc-type chunk
# size by this factor — small headroom keeps most articles whole (the whole point)
# while still bounding the rare very long article so it fits the LLM token budget.
_OVERSIZE_FACTOR = 1.2


def _splitter_for(cfg: dict) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["size"],
        chunk_overlap=cfg["overlap"],
        separators=["\n\n", "\n", ".", "،", " "],
    )


def _split_into_articles(text: str):
    """Yield (article_no, segment_text) for each article block in `text`.

    Returns [] when fewer than 2 article headers are found — a document with one
    or zero headers has no article structure to exploit, so the caller falls back
    to recursive character splitting.
    """
    matches = list(_ARTICLE_HEADER.finditer(text))
    if len(matches) < 2:
        return []
    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append((m.group(1), seg))
    return segments


def chunk_document(doc: Document, start_index: int) -> List[Document]:
    """Chunk one document, returning chunks with retrieval/scoring metadata set.

    `start_index` is the position of the first emitted chunk in the global
    chunks.pkl list; chunk_index is assigned contiguously from there so FAISS hits
    map back to chunks in O(1) (see CLAUDE.md "Chunk indexing").
    """
    doc_type = doc.metadata.get("doc_type", "legal_reference")
    cfg = config.CHUNKING_CONFIGS.get(doc_type, {"size": 2048, "overlap": 300})
    splitter = _splitter_for(cfg)

    pieces: List[Document] = []
    article_segments = (
        _split_into_articles(doc.page_content)
        if doc_type in ARTICLE_STRUCTURED_TYPES
        else []
    )

    if article_segments:
        # Article-aware path: one chunk per article, sub-split only if oversized.
        oversize = cfg["size"] * _OVERSIZE_FACTOR
        for art_no, seg in article_segments:
            if len(seg) <= oversize:
                sub_texts = [seg]
            else:
                holder = Document(page_content=seg, metadata=dict(doc.metadata))
                sub_texts = [c.page_content for c in splitter.split_documents([holder])]
            for txt in sub_texts:
                md = dict(doc.metadata)
                # Normalize defensively: Python's \d matches Arabic-Indic digits too, so a
                # doc that slipped past clean_arabic_legal_text() would otherwise store an
                # Arabic-Indic article number (e.g. '۲۳۹') here. Keep primary_article Western.
                md["primary_article"] = normalize_arabic_indic_digits(art_no)
                pieces.append(Document(page_content=txt, metadata=md))
    else:
        # Narrative path: recursive character splitting (legacy behavior).
        pieces = splitter.split_documents([doc])

    for i, chunk in enumerate(pieces):
        chunk.metadata["chunk_index"] = start_index + i
        chunk.metadata["referenced_articles"] = extract_article_references(chunk.page_content)
        chunk.metadata.setdefault("primary_article", "")
        chunk.metadata.setdefault("legal_topic", doc.metadata.get("legal_topic", ""))
    return pieces


def chunk_documents(documents, progress_iter=None) -> List[Document]:
    """Chunk a list of documents into a single contiguously-indexed chunk list.

    `progress_iter` lets a caller wrap `documents` in tqdm for a progress bar
    without this module depending on tqdm.
    """
    all_chunks: List[Document] = []
    for doc in (progress_iter or documents):
        all_chunks.extend(chunk_document(doc, len(all_chunks)))
    return all_chunks
