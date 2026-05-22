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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "google") # 'google' or 'openai' (OpenRouter)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen-2.5-72b-instruct")
# Cap Gemini 2.5 Flash's hidden thinking tokens. Unbounded (default) thinking
# eats max_output_tokens and truncates the visible answer mid-sentence.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "512"))
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
MAX_CONTEXT_CHARS = 8000
MAX_INPUT_CHARS = 50000
