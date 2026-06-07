"""
Legal AI Assistant — FastAPI Application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from app.services.retrieval import retrieval_service
from app.services.reranker import reranker_service
from app.services.session import session_manager
from app.services.article_lookup import article_lookup_service
from app.routers import qa, summarize, weakness, defense, health, chat, ingest, forensic, parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indices at startup."""
    logger.info("=" * 60)
    logger.info("🏛️  كونان — Legal AI Assistant — Starting up...")
    logger.info("=" * 60)

    try:
        retrieval_service.load()
        reranker_service.load()
        # ArticleLookupService shares the SAME chunks list the retriever just
        # loaded — no second disk read, no duplicate memory.
        article_lookup_service.load(retrieval_service.chunks)
        restored = session_manager.load_from_disk()
        session_manager.start_pruner(interval_s=3600)
        logger.info(f"✅ LLM model: {config.LLM_MODEL}")
        logger.info(f"✅ Embedding model: {config.EMBED_MODEL_NAME}")
        logger.info(
            f"✅ Reranker: {config.RERANKER_MODEL}"
            if reranker_service.is_loaded
            else "⚠️ Reranker: disabled or failed to load (retrieval falls back to RRF-only)"
        )
        if config.SESSION_PERSIST:
            logger.info(f"✅ Sessions persisted to: {config.SESSION_PERSIST_DIR} (restored {restored}, TTL {config.SESSION_TTL_HOURS}h)")
        else:
            logger.info("⚠️ Sessions: in-memory only (SESSION_PERSIST=False)")
        logger.info("✅ All services ready")
    except FileNotFoundError as e:
        logger.error(f"❌ Index files not found: {e}")
        logger.error("   Run 'python scripts/build_index.py' first!")
        raise

    yield
    logger.info("🛑 Shutting down...")


app = FastAPI(
    title="كونان — Legal AI Assistant API",
    description=(
        "RAG-based AI system for Egyptian Criminal Law.\n\n"
        "Features: Chat (with session history), Q&A, Summarization, "
        "Weakness Detection, Defense Memo Generation, Data Ingestion.\n\n"
        "All responses are in Arabic. Powered by the Conan persona."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow frontend to connect ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──
API_PREFIX = "/api/v1"
app.include_router(health.router,    prefix=API_PREFIX, tags=["Health"])
app.include_router(chat.router,      prefix=API_PREFIX, tags=["Chat"])
app.include_router(qa.router,        prefix=API_PREFIX, tags=["Q&A"])
app.include_router(summarize.router, prefix=API_PREFIX, tags=["Summarization"])
app.include_router(weakness.router,  prefix=API_PREFIX, tags=["Weakness Detection"])
app.include_router(defense.router,   prefix=API_PREFIX, tags=["Defense Memo"])
app.include_router(forensic.router,  prefix=API_PREFIX, tags=["Forensic Analysis"])
app.include_router(ingest.router,    prefix=API_PREFIX, tags=["Data Ingestion"])
app.include_router(parse.router,     prefix=API_PREFIX, tags=["Document Parsing"])


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "كونان — Legal AI Assistant",
        "version": "2.0.0",
        "message": (
            f"All API endpoints live under {API_PREFIX}. A path without that prefix "
            f"(e.g. /health or /qa) returns 404. Open /docs for the full interactive spec."
        ),
        "api_base": API_PREFIX,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "health":    f"GET  {API_PREFIX}/health",
            "qa":        f"POST {API_PREFIX}/qa",
            "chat":      f"POST {API_PREFIX}/chat",
            "summarize": f"POST {API_PREFIX}/summarize",
            "weakness":  f"POST {API_PREFIX}/weakness",
            "defense":   f"POST {API_PREFIX}/defense",
            "forensic":  f"POST {API_PREFIX}/forensic",
            "parse":     f"POST {API_PREFIX}/parse",
            "ingest":    f"POST {API_PREFIX}/ingest",
        },
    }


@app.exception_handler(404)
async def not_found_hint(request, exc):
    """Return a helpful pointer instead of a bare 404 for unknown paths."""
    from fastapi.responses import JSONResponse
    detail = getattr(exc, "detail", "Not Found")
    return JSONResponse(
        status_code=404,
        content={
            "detail": detail,
            "hint": (
                f"This path was not found. All endpoints are under {API_PREFIX} "
                f"(e.g. {API_PREFIX}/health, {API_PREFIX}/qa). See /docs for the full list."
            ),
            "api_base": API_PREFIX,
            "docs": "/docs",
        },
    )
