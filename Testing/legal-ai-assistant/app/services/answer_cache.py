"""In-process answer cache for the stateless /qa endpoint.

Serving a cached answer costs **zero LLM tokens** — the main lever for surviving the
free-tier daily/per-minute budget and for "pre-warming" a demo: run your demo questions
once, then they return instantly and for free during the live session.

Design choices, on purpose:
- **Exact normalized-match only** (no semantic/fuzzy similarity). For a legal tool we never
  want to serve a cached answer that belongs to a *different* question. Normalization only
  folds whitespace + Arabic-Indic digits, so "ما عقوبة السرقة؟ " hits the same key as
  "ما عقوبة السرقة؟".
- **LRU-bounded** (OrderedDict) so memory stays flat.
- **Optional atomic disk persistence** so a pre-warmed cache survives a server restart.

NOTE: rebuilding the retrieval index changes what a question *should* return, so clear the
cache after a rebuild (delete the persist file or call `answer_cache.clear()`).
"""
import os
import json
import threading
import logging
from collections import OrderedDict
from typing import Optional, Dict

import config
from app.services.preprocessing import normalize_arabic_indic_digits

logger = logging.getLogger(__name__)


def _normalize_question(q: str) -> str:
    """Light, reversible-enough normalization for cache keying (not for retrieval)."""
    s = normalize_arabic_indic_digits(q or "").strip()
    return " ".join(s.split())


class AnswerCache:
    def __init__(self, max_entries: int = 512, persist_path: Optional[str] = None):
        self._store: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max(1, max_entries)
        self._persist_path = persist_path
        self.hits = 0
        self.misses = 0
        self._load()

    def _key(self, question: str, k: int, prompt_style: Optional[str]) -> str:
        return f"{prompt_style or 'default'}|{k}|{_normalize_question(question)}"

    def get(self, question: str, k: int, prompt_style: Optional[str]) -> Optional[dict]:
        key = self._key(question, k, prompt_style)
        with self._lock:
            payload = self._store.get(key)
            if payload is not None:
                self._store.move_to_end(key)  # mark most-recently-used
                self.hits += 1
                return dict(payload)  # shallow copy so callers can mutate latency etc.
            self.misses += 1
            return None

    def put(self, question: str, k: int, prompt_style: Optional[str], payload: dict) -> None:
        key = self._key(question, k, prompt_style)
        with self._lock:
            self._store[key] = payload
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # evict least-recently-used
        self._save()

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
        self._save()
        return n

    def stats(self) -> Dict:
        with self._lock:
            return {"entries": len(self._store), "hits": self.hits, "misses": self.misses}

    # ── persistence ──
    def _load(self) -> None:
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, payload in data.get("entries", {}).items():
                self._store[key] = payload
            logger.info(f"✅ Answer cache restored ({len(self._store)} entries) from {self._persist_path}")
        except Exception as e:
            logger.warning(f"Answer cache load failed (starting empty): {e}")

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            tmp = self._persist_path + ".tmp"
            with self._lock:
                snapshot = {"entries": dict(self._store)}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False)
            os.replace(tmp, self._persist_path)
        except Exception as e:
            logger.warning(f"Answer cache save failed: {e}")


answer_cache = AnswerCache(
    max_entries=config.ANSWER_CACHE_MAX,
    persist_path=(config.ANSWER_CACHE_PATH if config.ANSWER_CACHE_PERSIST else None),
)
