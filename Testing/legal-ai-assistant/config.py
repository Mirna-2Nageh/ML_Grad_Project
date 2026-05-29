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
# Gemini supports multiple project keys: each Google Cloud project has its own free
# 20-req/day quota, so listing several keys (GOOGLE_API_KEYS=k1,k2) multiplies the daily
# budget — the router rotates to the next project when one hits its daily quota.
GOOGLE_API_KEYS = _key_list("GOOGLE_API_KEY", "GOOGLE_API_KEYS")
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""  # back-compat (first project; used by embeddings)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "google") # 'google' or 'openai' (OpenRouter)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Primary LLM provider: "gemini" | "groq" | "cerebras" | "xai" (all OpenAI-compatible
# except gemini). The chosen primary is tried first, then the other OpenAI-compatible
# backups (groq → cerebras → xai), then Gemini (multi-project), then OpenRouter.
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

# Cerebras (free tier, OpenAI-compatible, very fast — generous free daily budget + a
# llama-3.3-70b model). A strong free backup to Groq. Free key at https://cloud.cerebras.ai
CEREBRAS_API_KEYS = _key_list("CEREBRAS_API_KEY", "CEREBRAS_API_KEYS")
CEREBRAS_API_KEY = CEREBRAS_API_KEYS[0] if CEREBRAS_API_KEYS else ""  # back-compat
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")  # verify exact id at cloud.cerebras.ai

# Route oversized prompts (big case files) past providers that can't serve them in one
# request — a rough char proxy for tokens — straight to a large-context provider (Gemini),
# instead of wasting attempts on a guaranteed 413/over-context error. 0 = no limit.
# Keyed by the provider label used in the chain.
PROVIDER_MAX_PROMPT_CHARS = {
    "Groq":       int(os.getenv("GROQ_MAX_PROMPT_CHARS", "26000")),        # ~12k tokens/min cap
    "Cerebras":   int(os.getenv("CEREBRAS_MAX_PROMPT_CHARS", "16000")),    # ~8k-token context window
    "xAI":        int(os.getenv("XAI_MAX_PROMPT_CHARS", "26000")),
    "OpenRouter": int(os.getenv("OPENROUTER_MAX_PROMPT_CHARS", "80000")),
}  # Gemini intentionally has no cap — its large context is the home for big documents.
# Per-feature model routing: heavy reasoning (defense memo, weakness analysis) gets the
# strong model; high-volume QA/chat can use a cheaper/faster one. Falls back to GROQ_MODEL.
GROQ_MODEL_STRONG = os.getenv("GROQ_MODEL_STRONG", "llama-3.3-70b-versatile")
MODEL_BY_FEATURE = {
    "defense": GROQ_MODEL_STRONG,
    "weakness": GROQ_MODEL_STRONG,
    "forensic": GROQ_MODEL_STRONG,  # consistency analysis is reasoning-heavy
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
# In-process rate gate: minimum seconds between physical LLM provider requests. Spaces out
# the multi-call-per-request pattern (draft → retry → rescue) and concurrent users so the
# free-tier per-minute token/request budget isn't burst-exhausted (413). 0 disables.
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "2.0"))
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
RETRIEVAL_K = 10         # Final docs passed to the LLM (broader context; bounded by LLM token budget)
RETRIEVAL_K_DENSE = 60   # FAISS candidates — wider net over the corpus before rerank (CPU-only, no LLM cost)
RETRIEVAL_K_SPARSE = 60  # BM25 candidates
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
RETRIEVAL_K_RERANK = int(os.getenv("RETRIEVAL_K_RERANK", "60"))  # candidates fed to reranker (top-N from RRF) — wider = ranks over more of the corpus

# ──────────────────────────────────────────────
# Confidence Scoring (Phase 1 heuristic)
# ──────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "rerank":             0.30,  # mean rerank score over final top-k (gamma-calibrated)
    "source_count":       0.20,  # min(1, n_sources / 5)
    "article_validation": 0.30,  # 1.0 if every cited article appears in retrieved context, else 0
    "topic_match":        0.20,  # 1.0 if any source's legal_topic appears in the question
}
# bge-reranker-v2-m3 sigmoid scores sit low (~0.1) on Arabic legal text, which previously
# made well-grounded answers read as "low confidence" even when correct. Gamma<1 lifts the
# squashed mid-range (0.1 -> ~0.32 at 0.5) without saturating strong matches. 1.0 = off.
RERANK_CALIBRATION_GAMMA = float(os.getenv("RERANK_CALIBRATION_GAMMA", "0.5"))
CONFIDENCE_THRESHOLD_CLARIFY = float(os.getenv("CONFIDENCE_THRESHOLD_CLARIFY", "0.5"))

# ──────────────────────────────────────────────
# Answer cache (free-tier budget saver for stateless /qa)
# ──────────────────────────────────────────────
# Serving a cached answer costs 0 LLM tokens. Exact normalized-question match only.
USE_ANSWER_CACHE = os.getenv("USE_ANSWER_CACHE", "True").lower() == "true"
ANSWER_CACHE_MAX = int(os.getenv("ANSWER_CACHE_MAX", "512"))
ANSWER_CACHE_PERSIST = os.getenv("ANSWER_CACHE_PERSIST", "True").lower() == "true"
ANSWER_CACHE_PATH = os.getenv("ANSWER_CACHE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "answer_cache.json"))

# ──────────────────────────────────────────────
# Multi-Query Retrieval + Domain-Aware Routing (Phase 2)
# Heuristic synonym expansion: deterministic, no LLM cost. Targets retrieval gaps
# uncovered in eval(3) — e.g. "التوقيف الاحتياطي" wasn't matching "الحبس الاحتياطي"
# chunks, so the LLM had to refuse or hallucinate.
# ──────────────────────────────────────────────
# NOTE: USE_MULTI_QUERY is OFF by default — eval(v5) showed synonym expansion
# diluted retrieval quality on this index (4 questions regressed from passing
# to hallucinating). Flip to true once a higher-quality index (BGE-M3 +
# article-aware chunking) is in place; for now the single-query path is best.
USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "False").lower() == "true"
# Domain boost: when the question is clearly procedural (التوقيف، التحقيق، الطعن)
# or substantive (عقوبة، أركان، تعريف)، add weight to chunks whose doc_type matches
# the inferred domain. Weight is added to RRF score before reranking.
USE_DOMAIN_BOOST = os.getenv("USE_DOMAIN_BOOST", "True").lower() == "true"
DOMAIN_BOOST_WEIGHT = float(os.getenv("DOMAIN_BOOST_WEIGHT", "0.05"))
# Smarter retry on hallucinated citations:
#   - skip when no articles cited (a refusal has nothing to fix)
#   - up to N attempts, each one passes the growing block-list of bad articles
# 1 retry is the sweet spot on Groq's free tier: a 2nd retry blows the 6000
# token-per-minute budget (~4k tokens per call), causing 429s that cascade into
# fallbacks and stalled requests. Bump to 2+ only on paid tiers.
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "1"))

# Iterative retrieval: when the validate→retry→rescue pipeline still fails
# article validation OR returns a low-confidence refusal, re-run the whole
# pipeline at a larger k. Each step in the sequence is tried at most once per
# request, and we stop as soon as one of them produces a passing answer.
# Trade-off: doubles/triples latency on hard questions to recover them rather
# than returning a hallucination warning. Easy queries (those that pass on
# the first k) cost nothing extra.
USE_ITERATIVE_RETRIEVAL = os.getenv("USE_ITERATIVE_RETRIEVAL", "True").lower() == "true"
def _parse_int_list(env_val: str, fallback) -> list:
    try:
        out = [int(x.strip()) for x in env_val.split(",") if x.strip()]
        return out or fallback
    except ValueError:
        return fallback
ITERATIVE_K_SEQUENCE = _parse_int_list(
    os.getenv("ITERATIVE_K_SEQUENCE", "10,18,28"), [10, 18, 28]
)
# Programmatic post-processing of the LLM answer: strip casual openings
# ("حسناً"، "بالتأكيد"...) and rewrite the leaky template phrase
# "المادة المطلوبة غير متوفرة في السياق المقدم" if the LLM embedded it mid-clause.
USE_ANSWER_POSTPROCESS = os.getenv("USE_ANSWER_POSTPROCESS", "True").lower() == "true"

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
MAX_CONTEXT_CHARS = 6000
MAX_INPUT_CHARS = 50000
# Output token cap for short-answer features (QA/chat). Answers are typically 200-800 chars,
# so a 4096 reservation needlessly doubled per-request token cost on free tiers.
LLM_MAX_TOKENS_QA = int(os.getenv("LLM_MAX_TOKENS_QA", "1024"))
