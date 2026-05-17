# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository scope

Active development lives in `legal-ai-assistant/`. Files at this `Testing/` level (`part1_*.py … part5_*.py`, `streaming_indexer.py`, `*.ipynb`, root-level logs) are earlier experiment/notebook exports that the FastAPI service has replaced — prefer editing `legal-ai-assistant/` unless explicitly working on the experimental pipeline. The parent directory `../` holds the dataset, large model archives, and historical artifacts that are intentionally outside `Testing/`.

## Running the system

The service needs **two pieces** to be functional: a pre-built retrieval index in `legal-ai-assistant/data/` and API keys in `legal-ai-assistant/.env`. The server will refuse to start if the index isn't there (see `app.main.lifespan`).

```bash
# One-time setup
cd legal-ai-assistant
cp .env.example .env                       # then add OPENROUTER_API_KEY and/or GOOGLE_API_KEY
pip install -r requirements.txt            # or use the shipped venv at legal-ai-assistant/LaW/

# One-time index build (~10–30 min, processes ../final_total_dataset by default)
python scripts/build_index.py
python scripts/build_index.py --limit 100  # fast dev path
python scripts/build_from_chunks.py        # alt path: rebuilds from a pre-existing chunks.pkl

# Add documents to an existing index
python scripts/ingest_pipeline.py --input-dir /path/to/new/files [--dry-run]

# Run backend + Streamlit UI together (cleans ports 8000/8501 first)
./start_project.sh                         # at the Testing/ level

# Or individually
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run streamlit_app.py
```

`start_project.sh` uses `legal-ai-assistant/LaW/bin/python` — that venv is committed in-tree and is what the script expects.

There is **no test suite**. The `test_*.py` files in `legal-ai-assistant/` and at the `Testing/` level are ad-hoc smoke scripts (e.g. `test_gemini.py` pings the Gemini API, `test_retrieval.py` calls `retrieval_service.retrieve()` once); run them directly with `python <file>`.

## Architecture

The system is a hybrid-retrieval RAG service for Egyptian Criminal Law, Arabic-first throughout.

**Request flow (chat/QA endpoints):**
`POST /api/v1/{chat,qa,weakness,defense}` → `RetrievalService.retrieve()` does FAISS dense search + BM25 sparse search → results merged by **Reciprocal Rank Fusion** (`RRF_K=60`) → expert-rule contexts (matched by keyword from `data/expert_rules.json`) are **prepended** to the fused context → prompt assembled from `app/core/prompts.py` → `async_call_llm()` tries Google Gemini first (looping through `gemini-2.5-flash` → `2.0-flash` → `2.0-flash-lite`), falls back to OpenRouter Qwen.

**State that must stay coherent.** The retrieval indices are four files in `data/` that are written together and must be loaded together: `faiss_index/`, `bm25.pkl`, `chunks.pkl`, `tokenized_corpus.pkl`. Both `scripts/build_index.py` and `scripts/ingest_pipeline.py` write all four; the ingest endpoint (`POST /api/v1/ingest`) does the same merge in-process and mutates the singleton `retrieval_service`. Don't write to any of them in isolation.

**Chunk indexing.** `chunk.metadata["chunk_index"]` is the position in `chunks.pkl`. Retrieval relies on this to map FAISS hits back to chunks in O(1) — legacy chunks without `chunk_index` fall back to linear scan. New code that creates chunks should set it.

**Context expansion.** After RRF picks top-`k` chunks, retrieval grows each hit by one neighbor on each side **only when the neighbor shares the same `source` filename** (`app/services/retrieval.py:retrieve`). The `seen_indices` set deduplicates so expansion never double-counts.

**Chunking is doc-type-aware.** `clean_arabic_legal_text()` normalizes Arabic (strips diacritics, unifies alef forms, removes boilerplate); `get_document_type()` classifies by Arabic substrings in the file path (e.g. `جنايات` → `criminal_case`, `محكمه النقض` → `cassation_ruling`); chunk size + overlap come from `config.CHUNKING_CONFIGS[doc_type]`. Changing chunking config without rebuilding the index will drift the data out of sync.

**Sessions.** `app/services/session.py` keeps chat history per `session_id` in-memory (lost on restart). When `turn_count > MAX_TURNS` (10), it summarizes older turns via an LLM call and keeps the last `KEEP_RECENT*2` messages — the summary is injected into future prompts as `[ملخص المحادثة السابقة]`.

**Embeddings.** Default is local `BAAI/bge-m3` on CPU via `sentence-transformers` (1024-dim). `USE_REMOTE_EMBEDDINGS=true` switches to remote: Google Gemini Embeddings if `EMBED_PROVIDER=google` (rate-limited to 15 RPM via a 4 s sleep + 20-doc batches), otherwise OpenRouter. Switching providers requires rebuilding the FAISS index because vector dimensionality and semantics differ.

**LLM temperatures** are per-feature in `config.TEMPERATURES` and currently all set to `0.001` (effectively deterministic) despite the docstrings describing higher values for creative tasks — treat the code as source of truth.

**Module layout** (under `legal-ai-assistant/app/`):
- `main.py` — FastAPI entry, `lifespan` loads `retrieval_service` once on startup, registers routers under `/api/v1`.
- `models.py` — Pydantic schemas; these define the **wire contract** with the .NET frontend (see `api_contract.md`). Field names cross the wire as-is.
- `routers/` — thin endpoint handlers, one per feature (`chat`, `qa`, `summarize`, `weakness`, `defense`, `ingest`, `health`).
- `services/retrieval.py` — `RetrievalService` singleton (`retrieval_service`), `SBertEmbedding` LangChain-compatible wrapper, `GoogleEmbedding` for remote Gemini.
- `services/llm.py` — `call_llm` / `async_call_llm` with Gemini→OpenRouter fallback; strips `<think>...</think>` from Qwen output.
- `services/session.py` — `SessionManager` singleton with sliding-window compaction.
- `services/preprocessing.py` — Arabic normalization, BM25 tokenization, doc-type/legal-category classifiers.
- `core/prompts.py` — `SYSTEM_MESSAGES` and `PROMPTS` dicts, keyed by feature; "Nour" persona prompt lives here.

## Conventions

- All user-facing output is Modern Standard Arabic; the persona is "Nour" (نور). The prompts enforce strict textual adherence to provided context — don't add general legal commentary or weaken the "Use provided texts ONLY" rule without reason.
- `config.py` is the single source of truth for tunables (paths, model names, retrieval `k`, chunk sizes, temperatures, char limits). Don't hard-code these values in routers/services — read from `config`.
- `RetrievalService` is a process-wide singleton instantiated at module import. Tests/scripts that need it must call `retrieval_service.load()` once before `retrieve()`.
- The .env loads at config import time; new env vars belong in both `config.py` and `.env.example`.
