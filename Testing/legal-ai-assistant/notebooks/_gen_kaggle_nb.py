"""Generate kaggle_build_bge_m3_index.ipynb (valid nbformat v4 JSON)."""
import json, os

cells = []
def md(src):  cells.append(("markdown", src))
def code(src): cells.append(("code", src))

md(r'''# 🏛️ BGE-M3 Index Builder — Conan (Egyptian Criminal Law RAG)

Builds the **bge-m3 (1024-dim) + article-aware** retrieval index on a **GPU**, from the
HuggingFace corpus, and packages it for download. On the laptop this is a ~3-day CPU job;
here it's a few minutes of GPU compute.

**Before running:** `Settings → Accelerator → GPU T4 ×2` (or P100), then **Run All**.

The output index is byte-compatible with the FastAPI server: it inlines the project's exact
Arabic preprocessing, article-aware chunking, and the `normalize_embeddings=True` embedding
wrapper, and pins the same library versions so the server can unpickle it.

**After running:** download `data.new.zip` from the **Output** panel → on the laptop:
`unzip` into `legal-ai-assistant/data.new/` → `./scripts/swap_new_index.sh` → flip the bge-m3
lines in `.env` → restart the backend.''')

code(r'''# 1) Pinned deps — MUST match the server so it can unpickle the index.
#    torch is left as Kaggle's CUDA build (don't pin it, or you lose the GPU).
!pip install -q \
  "langchain-core==0.3.84" "langchain-community==0.3.25" "langchain-text-splitters==0.3.8" \
  "faiss-cpu==1.8.0" "rank-bm25==0.2.2" "numpy==1.26.4" \
  "sentence-transformers==2.7.0" "transformers==4.44.0" \
  "datasets" "huggingface-hub"
print("deps installed — if pip reported a conflict, do Run → Restart & Run All once.")''')

code(r'''# 2) Imports + config
import os, re, time, pickle, shutil
import numpy as np
import torch
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

DATASET_ID        = "Mirna2Nageh/Egyptian_Criminal_Legal_Assistant_RAG_V2"
EMBED_MODEL       = "BAAI/bge-m3"
EMBED_BATCH_SIZE  = 64       # GPU batch; drop to 32 if you hit CUDA OOM
EMBED_MAX_SEQ_LEN = 1024     # matches server config.EMBED_MAX_SEQ_LENGTH
USE_FP16          = True     # ~2x faster on GPU; negligible effect on normalized cosine retrieval
OUT_DIR           = "/kaggle/working/data.new"
NORMALIZE_TA_MARBUTA = False # matches server config

os.makedirs(OUT_DIR, exist_ok=True)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(no GPU — enable the accelerator!)")''')

code(r'''# 3) Arabic preprocessing — copied verbatim from app/services/preprocessing.py
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩" + "۰۱۲۳۴۵۶۷۸۹", "0123456789" + "0123456789")

def normalize_arabic_indic_digits(text):
    return text.translate(_ARABIC_INDIC_DIGITS)

def clean_arabic_legal_text(text):
    text = re.sub(r'[ً-ٰٟ]', '', text)
    text = normalize_arabic_indic_digits(text)
    text = re.sub(r'[أإآ]', 'ا', text)
    if NORMALIZE_TA_MARBUTA:
        text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'بسم الله الرحمن الرحيم', '', text)
    text = re.sub(r'باسم الشعب', '', text)
    text = re.sub(r'محكمة\s+\S+\s+الابتدائية', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'- \d+ -', '', text)
    text = re.sub(r'\(\s*\d+\s*\)', '', text)
    text = text.replace('؛', '؛ ').replace('،', '، ')
    return text.strip()

def preprocess_arabic_for_bm25(text):
    text = re.sub(r'[ً-ٰٟ]', '', text)
    text = normalize_arabic_indic_digits(text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]

def get_document_type(p):
    if 'جنايات' in p: return 'criminal_case'
    elif 'محكمه النقض' in p: return 'cassation_ruling'
    elif 'موسوعة' in p: return 'cassation_encyclopedia'
    elif 'قانون العقوبات' in p or 'عقوبات' in p: return 'penal_code'
    elif 'إجراءات' in p or 'اجراءات' in p or 'الاجرءات' in p: return 'criminal_procedure'
    elif 'الطب الشرعي' in p: return 'forensic_medicine'
    elif 'القواعد' in p or 'مجموعة' in p: return 'legal_rules_collection'
    elif 'قانون الجنائي' in p or 'الجنائي' in p: return 'criminal_law_reference'
    else: return 'legal_reference'

def get_legal_category(p):
    parts = p.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if 'موسوعة' in part and i + 1 < len(parts):
            nxt = parts[i + 1]
            if not nxt.endswith('.txt') and not nxt.endswith('.pdf'): return nxt
    return 'general'

def get_legal_topic(p):
    parts = p.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if 'موسوعة' in part and i + 1 < len(parts):
            nxt = parts[i + 1]
            if not (nxt.endswith('.txt') or nxt.endswith('.pdf')): return nxt
    return ''

_ARTICLE_BLOCK_PATTERN = re.compile(
    r'(?:المادة|المادتين|المادتان|المواد|مادة|مادتين|مواد)\s*:?\s*(\d+(?:\s*[،,و]\s*\d+)*)')
_NUM_PATTERN = re.compile(r'\d+')

def extract_article_references(text):
    normalized = normalize_arabic_indic_digits(text)
    matches = []
    for block in _ARTICLE_BLOCK_PATTERN.findall(normalized):
        matches.extend(_NUM_PATTERN.findall(block))
    unique = list(set(matches))
    try: unique.sort(key=int)
    except ValueError: unique.sort()
    return unique

CHUNKING_CONFIGS = {
    "criminal_case":          {"size": 3000, "overlap": 400},
    "cassation_ruling":       {"size": 2000, "overlap": 200},
    "cassation_encyclopedia": {"size": 2000, "overlap": 200},
    "penal_code":             {"size": 2000, "overlap": 200},
    "criminal_procedure":     {"size": 2048, "overlap": 300},
    "forensic_medicine":      {"size": 2400, "overlap": 300},
    "legal_rules_collection": {"size": 2048, "overlap": 300},
    "criminal_law_reference": {"size": 2048, "overlap": 300},
    "legal_reference":        {"size": 2048, "overlap": 300},
}
print("preprocessing ready")''')

code(r'''# 4) Article-aware chunking — copied verbatim from app/services/chunking.py
ARTICLE_STRUCTURED_TYPES = {
    "penal_code", "criminal_procedure", "criminal_law_reference",
    "legal_reference", "legal_rules_collection",
}
_ARTICLE_HEADER = re.compile(
    r'(?:^|\n)\s*(?:المادة|الماده|مادة|ماده)\s*(?:رقم\s*)?\(?\s*(\d{1,4})\s*\)?')
_OVERSIZE_FACTOR = 1.2

def _splitter_for(cfg):
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["size"], chunk_overlap=cfg["overlap"],
        separators=["\n\n", "\n", ".", "،", " "])

def _split_into_articles(text):
    matches = list(_ARTICLE_HEADER.finditer(text))
    if len(matches) < 2: return []
    segs = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip()
        if seg: segs.append((m.group(1), seg))
    return segs

def chunk_document(doc, start_index):
    doc_type = doc.metadata.get("doc_type", "legal_reference")
    cfg = CHUNKING_CONFIGS.get(doc_type, {"size": 2048, "overlap": 300})
    splitter = _splitter_for(cfg)
    pieces = []
    article_segments = _split_into_articles(doc.page_content) if doc_type in ARTICLE_STRUCTURED_TYPES else []
    if article_segments:
        oversize = cfg["size"] * _OVERSIZE_FACTOR
        for art_no, seg in article_segments:
            if len(seg) <= oversize:
                sub_texts = [seg]
            else:
                holder = Document(page_content=seg, metadata=dict(doc.metadata))
                sub_texts = [c.page_content for c in splitter.split_documents([holder])]
            for txt in sub_texts:
                md = dict(doc.metadata); md["primary_article"] = art_no
                pieces.append(Document(page_content=txt, metadata=md))
    else:
        pieces = splitter.split_documents([doc])
    for i, chunk in enumerate(pieces):
        chunk.metadata["chunk_index"] = start_index + i
        chunk.metadata["referenced_articles"] = extract_article_references(chunk.page_content)
        chunk.metadata.setdefault("primary_article", "")
        chunk.metadata.setdefault("legal_topic", doc.metadata.get("legal_topic", ""))
    return pieces

def chunk_documents(documents):
    out = []
    for doc in documents:
        out.extend(chunk_document(doc, len(out)))
    return out
print("chunking ready")''')

code(r'''# 5) Load the HF corpus and inspect its schema
ds = load_dataset(DATASET_ID)
print(ds)
split = "train" if "train" in ds else list(ds.keys())[0]
data = ds[split]
print("\nusing split:", split, "| rows:", len(data))
print("columns:", data.column_names)
print("\n--- sample row (truncated) ---")
for k, v in data[0].items():
    print(f"  {k}: {str(v)[:140]}")''')

code(r'''# 6) Map dataset rows -> LangChain Documents.
#    Auto-detects the text + path/metadata columns; override the *_COL names below if needed.
cols = set(data.column_names)
def _first(cands):
    for c in cands:
        if c in cols: return c
    return None

TEXT_COL   = _first(["page_content", "text", "content", "cleaned_text", "document", "body", "chunk"])
PATH_COL   = _first(["source", "path", "file_path", "filepath", "filename", "file", "doc_id", "id"])
DTYPE_COL  = _first(["doc_type", "document_type", "type"])
TOPIC_COL  = _first(["legal_topic", "topic"])
CAT_COL    = _first(["legal_category", "category"])
assert TEXT_COL, f"Could not find a text column in {data.column_names} — set TEXT_COL manually."
print("TEXT_COL=", TEXT_COL, "| PATH_COL=", PATH_COL, "| DTYPE_COL=", DTYPE_COL,
      "| TOPIC_COL=", TOPIC_COL, "| CAT_COL=", CAT_COL)

# Is the dataset already chunked, or document-level? (article-aware chunking only applies to docs)
ALREADY_CHUNKED = ("chunk_index" in cols) or (len(data) > 5000)
print("ALREADY_CHUNKED =", ALREADY_CHUNKED, f"(rows={len(data)})")

docs = []
for i, row in enumerate(data):
    raw = row[TEXT_COL] or ""
    if len(raw.strip()) < 30:
        continue
    path = str(row[PATH_COL]) if PATH_COL and row.get(PATH_COL) else f"doc_{i}.txt"
    cleaned = clean_arabic_legal_text(raw)
    meta = {
        "source": path,
        "filename": os.path.basename(path),
        "doc_type": row[DTYPE_COL] if DTYPE_COL and row.get(DTYPE_COL) else get_document_type(path),
        "legal_category": row[CAT_COL] if CAT_COL and row.get(CAT_COL) else get_legal_category(path),
        "legal_topic": row[TOPIC_COL] if TOPIC_COL and row.get(TOPIC_COL) else get_legal_topic(path),
        "char_count": len(cleaned),
    }
    docs.append(Document(page_content=cleaned, metadata=meta))
print(f"built {len(docs):,} documents")''')

code(r'''# 7) Chunk (article-aware) — or use rows directly if already chunked
t0 = time.time()
if ALREADY_CHUNKED:
    chunks = docs
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata.setdefault("referenced_articles", extract_article_references(c.page_content))
        c.metadata.setdefault("primary_article", "")
    print("dataset was pre-chunked — using rows as chunks (article-aware splitting skipped)")
else:
    chunks = chunk_documents(docs)
article_anchored = sum(1 for c in chunks if c.metadata.get("primary_article"))
print(f"✅ {len(chunks):,} chunks ({article_anchored:,} article-anchored) in {time.time()-t0:.1f}s")''')

code(r'''# 8) GPU embed with bge-m3 (normalize_embeddings=True) -> FAISS, saved exactly like build_index.py
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading {EMBED_MODEL} on {device} (fp16={USE_FP16}) ...")
_model = SentenceTransformer(EMBED_MODEL, device=device)
_model.max_seq_length = EMBED_MAX_SEQ_LEN
if USE_FP16 and device == "cuda":
    _model = _model.half()

class SBertEmbedding:
    # LangChain-compatible; mirrors app/services/retrieval.SBertEmbedding (normalize=True).
    def __init__(self, model): self.model = model
    def embed_documents(self, texts):
        v = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                              batch_size=EMBED_BATCH_SIZE, convert_to_numpy=True)
        return v.astype(np.float32).tolist()
    def embed_query(self, text):
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return v[0].astype(np.float32).tolist()
    def __call__(self, text): return self.embed_query(text)

emb = SBertEmbedding(_model)
t0 = time.time()
vectorstore = FAISS.from_documents(chunks, emb)
faiss_dir = os.path.join(OUT_DIR, "faiss_index")
vectorstore.save_local(faiss_dir)
print(f"✅ FAISS: {vectorstore.index.ntotal:,} vectors in {time.time()-t0:.1f}s -> {faiss_dir}")''')

code(r'''# 9) BM25 + save the remaining three artifacts (same filenames/format as build_index.py)
t0 = time.time()
tokenized_corpus = [preprocess_arabic_for_bm25(c.page_content) for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)
with open(os.path.join(OUT_DIR, "bm25.pkl"), "wb") as f: pickle.dump(bm25, f)
with open(os.path.join(OUT_DIR, "chunks.pkl"), "wb") as f: pickle.dump(chunks, f)
with open(os.path.join(OUT_DIR, "tokenized_corpus.pkl"), "wb") as f: pickle.dump(tokenized_corpus, f)
print(f"✅ BM25 + pickles saved in {time.time()-t0:.1f}s")''')

code(r'''# 10) Verify the four artifacts, then zip for download
print("Index files in", OUT_DIR, ":")
for root, _, files in os.walk(OUT_DIR):
    for fn in files:
        p = os.path.join(root, fn)
        print(f"  {os.path.relpath(p, OUT_DIR):<28} {os.path.getsize(p)/1e6:8.1f} MB")

zip_path = shutil.make_archive("/kaggle/working/data.new", "zip", OUT_DIR)
print(f"\n✅ zipped -> {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
print("Download 'data.new.zip' from the right-hand Output panel.")
try:
    from IPython.display import FileLink, display
    display(FileLink("/kaggle/working/data.new.zip"))
except Exception as e:
    print("(open the Output tab to download)", e)''')

md(r'''## (Optional) Push the index to the HuggingFace Hub instead of downloading

If the zip is large / your connection is slow, upload it to a private HF dataset and `wget` it
on the laptop. Needs an HF token with **write** access (`Settings → Secrets → HF_TOKEN`).''')

code(r'''# OPTIONAL — uncomment to push the zip to a private HF dataset repo
# from huggingface_hub import HfApi
# from kaggle_secrets import UserSecretsClient
# HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
# REPO = "Mirna2Nageh/conan-bge-m3-index"   # change to your repo
# api = HfApi(token=HF_TOKEN)
# api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
# api.upload_file(path_or_fileobj="/kaggle/working/data.new.zip",
#                 path_in_repo="data.new.zip", repo_id=REPO, repo_type="dataset")
# print("pushed to", REPO)''')

md(r'''## On the laptop, after downloading `data.new.zip`

```bash
cd legal-ai-assistant
unzip ~/Downloads/data.new.zip -d data.new          # -> data.new/faiss_index, bm25.pkl, chunks.pkl, tokenized_corpus.pkl
./scripts/swap_new_index.sh                          # data/ -> data.legacy/, data.new/ -> data/ (carries sessions, drops stale cache)
# then set in .env:  EMBED_MODEL=BAAI/bge-m3  EMBED_DIMENSIONS=1024  EMBED_MAX_SEQ_LENGTH=1024  EMBED_BATCH_SIZE=8
systemctl --user restart conan-backend.service       # reload with the bge-m3 index
python scripts/eval_harness.py --mode retrieval --tag bge_m3   # then --compare baseline_minilm bge_m3
```
Roll back anytime: `rm -rf data && mv data.legacy data && systemctl --user restart conan-backend.service`.''')

nb = {
    "cells": [
        {"cell_type": ct, "metadata": {},
         **({"source": src.splitlines(keepends=True)} if ct == "markdown"
            else {"source": src.splitlines(keepends=True), "outputs": [], "execution_count": None})}
        for ct, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_build_bge_m3_index.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "with", len(cells), "cells")
