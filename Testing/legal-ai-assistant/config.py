"""
Central configuration for Legal AI Assistant.
All tunable parameters in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
def _key_list(*env_names):
    """Collect API keys from one or more env vars, each possibly comma-separated.
    Enables multi-key rotation: set e.g. GROQ_API_KEYS=k1,k2,k3 (or repeat in GROQ_API_KEY)."""
    keys = []
    for name in env_names:
        for k in os.getenv(name, "").split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


OPENROUTER_API_KEYS = _key_list("OPENROUTER_API_KEY", "OPENROUTER_API_KEYS")
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else ""  # back-compat (first key)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "google") # 'google' or 'openai' (OpenRouter)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Primary LLM provider: "gemini" | "xai" | "groq" (all OpenAI-compatible except gemini).
# The chosen primary is tried first, then Gemini (if GOOGLE_API_KEY set), then OpenRouter.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")  # OpenRouter fallback model

# xAI Grok (OpenAI-compatible API at https://api.x.ai/v1)
XAI_API_KEYS = _key_list("XAI_API_KEY", "XAI_API_KEYS")
XAI_API_KEY = XAI_API_KEYS[0] if XAI_API_KEYS else ""  # back-compat
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.1-fast")  # verify exact id via GET /v1/models

# Groq (free tier, OpenAI-compatible API at https://api.groq.com/openai/v1)
GROQ_API_KEYS = _key_list("GROQ_API_KEY", "GROQ_API_KEYS")
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""  # back-compat
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # default model
# Per-feature model routing: heavy reasoning (defense memo, weakness analysis) gets the
# strong model; high-volume QA/chat can use a cheaper/faster one. Falls back to GROQ_MODEL.
GROQ_MODEL_STRONG = os.getenv("GROQ_MODEL_STRONG", "llama-3.3-70b-versatile")
MODEL_BY_FEATURE = {
    "defense": GROQ_MODEL_STRONG,
    "weakness": GROQ_MODEL_STRONG,
    "summarize": GROQ_MODEL,
    "qa": GROQ_MODEL,
    "default": GROQ_MODEL,
}
# Cap Gemini 2.5 Flash's hidden thinking tokens. Unbounded (default) thinking
# eats max_output_tokens and truncates the visible answer mid-sentence.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "512"))
# Retry transient Gemini failures (503 overloaded / 429 rate-limited) per model tier
# with exponential backoff before falling through to the next tier / OpenRouter.
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BASE_DELAY = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "2.0"))  # seconds; doubles each retry
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
USE_REMOTE_EMBEDDINGS = os.getenv("USE_REMOTE_EMBEDDINGS", "False").lower() == "true"
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024")) # 1024 for BGE-M3
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))           # CPU: 8 keeps attention under ~3 GB even for long chunks
EMBED_MAX_SEQ_LENGTH = int(os.getenv("EMBED_MAX_SEQ_LENGTH", "1024")) # bge-m3 supports 8192, but Arabic legal chunks ~2000 chars rarely need more, and 1024 keeps memory bounded
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(BASE_DIR, "..", "final_total_dataset"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
BM25_PATH = os.path.join(DATA_DIR, "bm25.pkl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")
TOKENIZED_CORPUS_PATH = os.path.join(DATA_DIR, "tokenized_corpus.pkl")

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ──────────────────────────────────────────────
# Per-Feature Temperature Settings
# ──────────────────────────────────────────────
TEMPERATURES = {
    "qa": 0.001,           # Legal Q&A: precise, deterministic
    "summarize": 0.001,    # Summarization: slight creativity
    "weakness": 0.001,     # Weakness detection: creative analysis
    "defense": 0.001,      # Defense memo: persuasive writing
    "default": 0.001,      # Fallback
}

# ──────────────────────────────────────────────
# Retrieval Parameters
# ──────────────────────────────────────────────
RETRIEVAL_K = 7          # Final docs returned
RETRIEVAL_K_DENSE = 30   # FAISS candidates
RETRIEVAL_K_SPARSE = 30  # BM25 candidates
RRF_K = 60               # RRF constant

# ──────────────────────────────────────────────
# Chunking Configs (per document type)
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
NORMALIZE_TA_MARBUTA = False

# ──────────────────────────────────────────────
# Reranker (Cross-Encoder)
# ──────────────────────────────────────────────
USE_RERANKER = os.getenv("USE_RERANKER", "True").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RETRIEVAL_K_RERANK = int(os.getenv("RETRIEVAL_K_RERANK", "30"))  # candidates fed to reranker (top-N from RRF)

# ──────────────────────────────────────────────
# Confidence Scoring (Phase 1 heuristic)
# ──────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "rerank":             0.45,  # mean rerank score over final top-k (sigmoid-bound)
    "source_count":       0.20,  # min(1, n_sources / 5)
    "article_validation": 0.20,  # 1.0 if every cited article appears in retrieved context, else 0
    "topic_match":        0.15,  # 1.0 if any source's legal_topic appears in the question
}
CONFIDENCE_THRESHOLD_CLARIFY = float(os.getenv("CONFIDENCE_THRESHOLD_CLARIFY", "0.4"))

# ──────────────────────────────────────────────
# Agentic memo self-check (draft → verify against context → revise)
# Closes the argument-level hallucination gap: validator catches fabricated article
# numbers; this catches unsupported legal arguments/citations and revises them out.
# ──────────────────────────────────────────────
MEMO_SELF_CHECK = os.getenv("MEMO_SELF_CHECK", "True").lower() == "true"
MEMO_SELF_CHECK_MAX_ITERS = int(os.getenv("MEMO_SELF_CHECK_MAX_ITERS", "1"))

# ──────────────────────────────────────────────
# Session Management (chat history compaction + persistence)
# ──────────────────────────────────────────────
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "6"))         # compact when > N turns
SESSION_KEEP_RECENT = int(os.getenv("SESSION_KEEP_RECENT", "3"))     # keep last N turns after compaction
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "168"))       # 7 days; sessions idle longer get pruned
SESSION_PERSIST = os.getenv("SESSION_PERSIST", "True").lower() == "true"
SESSION_PERSIST_DIR = os.getenv("SESSION_PERSIST_DIR", os.path.join(DATA_DIR, "sessions"))

# ──────────────────────────────────────────────
# Ingest Automation
# ──────────────────────────────────────────────
INGEST_INBOX_DIR = os.getenv("INGEST_INBOX_DIR", os.path.join(DATA_DIR, "inbox"))
INGEST_REGISTRY_PATH = os.path.join(DATA_DIR, ".ingested_files.json")
INGEST_WATCH_INTERVAL_S = int(os.getenv("INGEST_WATCH_INTERVAL_S", "30"))

# ──────────────────────────────────────────────
# Benchmark (compare our system vs other LLMs)
# All routed through OpenRouter so no extra API keys needed.
# ──────────────────────────────────────────────
BENCHMARK_MODELS = {
    "ours":   LLM_MODEL,
    "openai": os.getenv("BENCHMARK_OPENAI_MODEL", "openai/gpt-4o-mini"),
    "claude": os.getenv("BENCHMARK_CLAUDE_MODEL", "anthropic/claude-sonnet-4.5"),
}

# ──────────────────────────────────────────────
# API Limits
# ──────────────────────────────────────────────
# Context budget: trimmed so a full request (input + output reservation) fits free-tier
# token-per-minute caps (e.g. Groq 8B TPM=6000). Expert rules + top chunks still fit.
MAX_CONTEXT_CHARS = 4500
MAX_INPUT_CHARS = 50000
# Output token cap for short-answer features (QA/chat). Answers are typically 200-800 chars,
# so a 4096 reservation needlessly doubled per-request token cost on free tiers.
LLM_MAX_TOKENS_QA = int(os.getenv("LLM_MAX_TOKENS_QA", "1024"))
