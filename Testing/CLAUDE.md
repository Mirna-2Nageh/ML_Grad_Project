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

# Recommended OS dep for cleaner Arabic PDF extraction (the pipeline falls back to PyPDF2
# automatically if pdftotext isn't on PATH).
sudo apt-get install -y poppler-utils      # or: brew install poppler

# Index build (~10–30 min, processes ../final_total_dataset by default).
# After Phase 1 the chunk metadata schema is richer (legal_topic, referenced_articles),
# so existing pre-Phase-1 indices need a rebuild for topic-match + article-validation
# signals to score correctly. The server still runs against a legacy index; it just
# won't get those signals.
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

The system is a hybrid-retrieval RAG service for Egyptian Criminal Law, Arabic-first throughout. The target architecture (multi-phase) is documented in the saved project memory `enterprise-rag-target-vision`; what's in code today is **Phase 1**: structured responses + confidence scoring + evidence validation + cross-encoder reranking + graph-lite topic metadata.

**Request flow (chat/QA/weakness/defense endpoints):**
1. `RetrievalService.retrieve()` — FAISS dense search (top 30) + BM25 sparse search (top 30) → **Reciprocal Rank Fusion** (`RRF_K=60`) → top 30 candidates.
2. **Cross-encoder rerank** (`RerankerService`, `bge-reranker-v2-m3`, sigmoid-bound 0–1) narrows the 30 candidates → final top `k` (default 7). If the reranker fails to load, retrieval silently falls back to RRF-only order (no behavior break).
3. Context expansion: each surviving chunk grows by one same-source neighbor on each side.
4. Expert-rule contexts (matched by keyword from `data/expert_rules.json`) are **prepended** to the final context.
5. Prompt assembled from `app/core/prompts.py` → `async_call_llm()` tries Google Gemini first (`gemini-2.5-flash` → `2.0-flash` → `2.0-flash-lite`), falls back to OpenRouter Qwen.
6. **Post-generation evidence validation** — `app/services/confidence.validate_evidence` regex-extracts every `المادة \d+` (after Arabic-Indic digit normalization) from the LLM's answer and verifies each appears in the retrieved context. Hallucinated articles → warning + 0.3 confidence penalty.
7. **Confidence scoring** — weighted heuristic over `mean(rerank_scores)`, `source_count`, `article_validation`, `topic_match`. Weights in `config.CONFIDENCE_WEIGHTS`. If final confidence < `CONFIDENCE_THRESHOLD_CLARIFY` (default 0.4), the response includes a clarification-prompt warning.

**Response shape** is the same across `/qa`, `/chat`, `/weakness`, `/defense`: `{answer|analysis|memorandum, confidence_score, confidence_factors, sources[{filename, source, doc_type, legal_category, legal_topic, article, referenced_articles, page, retrieval_score, rerank_score}], warnings, conflicts_detected, latency_ms, model}`. `app/models.py` is the wire contract — renaming a field breaks the .NET client.

**State that must stay coherent.** The retrieval indices are four files in `data/` that are written together and must be loaded together: `faiss_index/`, `bm25.pkl`, `chunks.pkl`, `tokenized_corpus.pkl`. Both `scripts/build_index.py` and `scripts/ingest_pipeline.py` write all four; the ingest endpoint (`POST /api/v1/ingest`) does the same merge in-process and mutates the singleton `retrieval_service`. Don't write to any of them in isolation.

**Chunk indexing.** `chunk.metadata["chunk_index"]` is the position in `chunks.pkl` — retrieval relies on this to map FAISS hits back to chunks in O(1) (legacy chunks without it fall back to linear scan). New chunk-creation code must also set:
- `legal_topic` — encyclopedia subdir name (`تزوير`, `قتل عمد`, …) extracted by `get_legal_topic(path)`. Empty for non-encyclopedia files. Used for `topic_match` confidence factor.
- `referenced_articles` — list of Egyptian-law article numbers cited in the chunk text, extracted by `extract_article_references()`. Used to feed `SourceInfo.article` + speed up evidence validation.

**Context expansion.** After rerank picks top-`k` chunks, retrieval grows each hit by one neighbor on each side **only when the neighbor shares the same `source` filename** (`app/services/retrieval.py:retrieve`). The `seen_indices` set deduplicates so expansion never double-counts.

**Chunking is doc-type-aware.** `clean_arabic_legal_text()` normalizes Arabic (strips diacritics, unifies alef forms, **normalizes Arabic-Indic digits ٠-٩ → 0-9**, removes boilerplate); `get_document_type()` classifies by Arabic substrings in the file path (e.g. `جنايات` → `criminal_case`, `محكمه النقض` → `cassation_ruling`); chunk size + overlap come from `config.CHUNKING_CONFIGS[doc_type]`. Changing chunking config or digit normalization without rebuilding the index will drift the data out of sync.

**Sessions.** `app/services/session.py` keeps chat history per `session_id`. When `turn_count > SESSION_MAX_TURNS` (default 6, configurable), older turns are LLM-summarized; the last `SESSION_KEEP_RECENT * 2` messages stay verbatim. The summary is injected into future prompts as `[ملخص المحادثة السابقة]`.

Sessions **persist to disk** as JSON files in `SESSION_PERSIST_DIR` (default `data/sessions/`) — they survive server restarts. Toggle off with `SESSION_PERSIST=False`. Idle sessions older than `SESSION_TTL_HOURS` (default 168 = 7 days) are pruned by a background task started in `main.py:lifespan`. The on-disk filename is the session_id sanitized to `[A-Za-z0-9_-]{1,128}`. Writes are atomic (write-to-`.tmp` + `os.replace`).

**Streaming chat.** `POST /api/v1/chat/stream` returns Server-Sent Events as Qwen generates tokens (`async_stream_llm` in `app/services/llm.py`, OpenRouter-only — Gemini streaming isn't wired yet). Each token chunk is `data: {"chunk": "...", "done": false}\n\n`; the final event carries confidence/sources/warnings: `data: {"done": true, "session_id": "...", "confidence_score": ..., ...}\n\n`. Session persistence + evidence validation run after the stream completes, so the final event has the full grounding info. The non-streaming `POST /api/v1/chat` retains the Gemini-first fallback chain.

**Automated ingest.** `scripts/watch_ingest.py` polls `INGEST_INBOX_DIR` every `INGEST_WATCH_INTERVAL_S` seconds, ingests new files via the same pipeline as `scripts/ingest_pipeline.py`, and tracks processed file paths in `data/.ingested_files.json`. Idempotent. The server exposes the same as `POST /api/v1/ingest/scan` — it runs `scan_once()` in a thread, then calls `retrieval_service.reload_indices()` to hot-reload FAISS+BM25+chunks from disk without re-initializing the embedding model. Use either: cron the script OR have the .NET layer trigger `/ingest/scan` after dropping files.

**Benchmark.** `scripts/benchmark_llms.py` queries our system + `openai/gpt-4o-mini` + `anthropic/claude-sonnet-4.5` (all via OpenRouter — single API key) on the same Arabic legal questions. Two modes: `--mode rag` (all 3 share the same retrieved context, isolates answer quality) or `--mode raw` (each model answers from its own knowledge, tests baseline legal knowledge). Outputs a CSV with per-model latency, answer length, cited article numbers. Models configurable via `BENCHMARK_OPENAI_MODEL` + `BENCHMARK_CLAUDE_MODEL` env vars.

**Embeddings.** Default is local `BAAI/bge-m3` on CPU via `sentence-transformers` (1024-dim). `USE_REMOTE_EMBEDDINGS=true` switches to remote: Google Gemini Embeddings if `EMBED_PROVIDER=google` (rate-limited to 15 RPM via a 4 s sleep + 20-doc batches), otherwise OpenRouter. Switching providers requires rebuilding the FAISS index because vector dimensionality and semantics differ.

**Reranker.** `bge-reranker-v2-m3` via `sentence-transformers.CrossEncoder`, loaded once at startup by `RerankerService.load()`. Sigmoid activation bounds scores to [0,1] so they feed cleanly into confidence scoring. Toggleable via `USE_RERANKER` env var; if it fails to load, retrieval falls back to RRF-only order automatically.

**PDF extraction.** `app/services/document_loader.load_pdf()` tries `pdftotext -layout -enc UTF-8` (poppler-utils) first — substantially cleaner Arabic layout — and falls back to `PyPDF2.PdfReader` if poppler isn't on PATH. Used by `scripts/build_index.py`, `scripts/ingest_pipeline.py`, and `app/routers/ingest.py`.

**LLM temperatures** are per-feature in `config.TEMPERATURES` and currently all set to `0.001` (effectively deterministic) despite the docstrings describing higher values for creative tasks — treat the code as source of truth.

**Module layout** (under `legal-ai-assistant/app/`):
- `main.py` — FastAPI entry; `lifespan` loads `retrieval_service`, `reranker_service`, restores `session_manager` from disk, starts the session pruner; registers routers under `/api/v1`.
- `models.py` — Pydantic schemas; **wire contract** with the .NET frontend (see `api_contract.md`). Field names cross the wire as-is.
- `routers/` — thin endpoint handlers (`chat` [+ `/chat/stream`], `qa`, `summarize`, `weakness`, `defense`, `ingest` [+ `/ingest/scan`], `health`). Each retrieval-backed router runs post-generation evidence validation + confidence scoring inline.
- `services/retrieval.py` — `RetrievalService` singleton with FAISS + BM25 + RRF + reranker pipeline. Exposes `reload_indices()` for in-process hot-reload after out-of-process ingest.
- `services/reranker.py` — `RerankerService` singleton (`CrossEncoder`, sigmoid-bounded). Graceful no-op if model fails to load.
- `services/confidence.py` — pure functions: `validate_evidence`, `topic_match`, `compute_confidence`.
- `services/document_loader.py` — `load_text` / `load_pdf` / `load_docx` / `load_any`. PDF tries poppler `pdftotext` then PyPDF2.
- `services/llm.py` — `call_llm` / `async_call_llm` (sync + threaded async, Gemini→OpenRouter fallback) and `async_stream_llm` (async generator, OpenRouter-only) for the streaming endpoint.
- `services/session.py` — `SessionManager` with sliding-window compaction, atomic JSON-file persistence, TTL-based pruner.
- `services/preprocessing.py` — Arabic normalization (incl. Arabic-Indic digit → Western), BM25 tokenization, doc-type/legal-category classifiers, `get_legal_topic`, `extract_article_references`.
- `core/prompts.py` — `SYSTEM_MESSAGES` and `PROMPTS` dicts (keys: `qa_standard`, `qa_restrictive`, `weakness`, `defense`, `chat`, `summarize`, `compact_history`); "Nour" persona prompt lives here.
- `scripts/watch_ingest.py` — watch-folder ingest CLI. Uses functions from `scripts/ingest_pipeline.py`. Tracks processed files in `data/.ingested_files.json`.
- `scripts/benchmark_llms.py` — compare our RAG vs `openai/gpt-4o-mini` vs `anthropic/claude-sonnet-4.5` (all via OpenRouter).

## Conventions

- All user-facing output is Modern Standard Arabic; the persona is "Nour" (نور). The prompts enforce strict textual adherence to provided context — don't add general legal commentary or weaken the "Use provided texts ONLY" rule without reason. **Every cited article number must trace back to retrieved context** or the evidence validator will flag it.
- `config.py` is the single source of truth for tunables (paths, model names, retrieval `k`, chunk sizes, temperatures, char limits, confidence weights, reranker config). Don't hard-code these values in routers/services — read from `config`.
- `RetrievalService` and `RerankerService` are process-wide singletons instantiated at module import. Tests/scripts that need them must call `.load()` once before use.
- The .env loads at config import time; new env vars belong in both `config.py` and `.env.example`.
- New code that creates chunks must set `chunk_index`, `legal_topic`, and `referenced_articles` on `chunk.metadata` — missing any of these silently degrades retrieval/scoring quality.
