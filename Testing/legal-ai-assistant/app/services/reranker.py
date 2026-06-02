"""Cross-encoder reranker (bge-reranker-v2-m3) — sits between RRF and final top-k."""
import logging
import threading
from typing import List, Tuple

import config

logger = logging.getLogger(__name__)

# Bounds concurrent cross-encoder inference. Reranks run in threadpool threads (handlers
# offload via asyncio.to_thread), so a threading.Semaphore is the right primitive: excess
# reranks block here instead of all thrashing the CPU and timing out together.
_RERANK_SLOTS = threading.Semaphore(config.RERANK_MAX_CONCURRENCY)


class RerankerService:
    def __init__(self):
        self.model = None
        self._loaded = False
        self.load_error: str = ""

    def load(self):
        if not config.USE_RERANKER:
            logger.info("Reranker disabled (config.USE_RERANKER=False)")
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
            self.model = CrossEncoder(
                config.RERANKER_MODEL, device="cpu", max_length=512
            )
            self._loaded = True
            logger.info("✅ Reranker loaded")
        except Exception as e:
            self.load_error = str(e)
            logger.warning(f"⚠️ Reranker failed to load — retrieval will fall back to RRF-only: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def rerank(self, query: str, passages: List[str], top_k: int) -> List[Tuple[int, float]]:
        """Return [(passage_index, score), ...] sorted desc, length ≤ top_k. Identity order if not loaded."""
        if not passages:
            return []
        if not self.is_loaded:
            return [(i, 0.0) for i in range(min(top_k, len(passages)))]
        import torch  # local import keeps startup cheap when reranker disabled
        pairs = [[query, p[:2000]] for p in passages]  # truncate to cap memory + latency
        # Limit concurrent CPU inference (see _RERANK_SLOTS) so simultaneous heavy requests
        # queue briefly rather than all thrashing the cores and blowing the client timeout.
        with _RERANK_SLOTS:
            scores = self.model.predict(pairs, activation_fct=torch.nn.Sigmoid()).tolist()
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]


reranker_service = RerankerService()
