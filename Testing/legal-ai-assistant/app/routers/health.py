"""Health check endpoint."""
from fastapi import APIRouter
import config
from app.models import HealthResponse
from app.services.retrieval import retrieval_service
from app.services.reranker import reranker_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check service health and loaded resources."""
    is_ready = retrieval_service.is_loaded and not retrieval_service._load_errors
    status = "ok" if is_ready else ("degraded" if retrieval_service.is_loaded else "loading")

    return HealthResponse(
        status=status,
        vectors=retrieval_service.vectorstore.index.ntotal if (retrieval_service.is_loaded and retrieval_service.vectorstore) else 0,
        chunks=len(retrieval_service.chunks) if (retrieval_service.is_loaded and retrieval_service.chunks) else 0,
        model=config.LLM_MODEL,
        embedding_model=config.EMBED_MODEL_NAME,
        reranker_loaded=reranker_service.is_loaded,
        reranker_model=config.RERANKER_MODEL if reranker_service.is_loaded else "",
        load_errors=retrieval_service._load_errors + ([f"Reranker: {reranker_service.load_error}"] if reranker_service.load_error else []),
    )
