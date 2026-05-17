"""Confidence scoring + post-generation evidence validation."""
import re
import logging
from typing import List, Dict, Tuple

import config
from app.services.preprocessing import (
    normalize_arabic_indic_digits,
    extract_article_references,
)

logger = logging.getLogger(__name__)


def validate_evidence(answer: str, contexts: List[str]) -> Tuple[bool, List[str]]:
    """Check every article cited in `answer` appears in `contexts`. Returns (passed, missing_articles)."""
    cited = extract_article_references(answer)
    if not cited:
        return True, []
    blob = normalize_arabic_indic_digits(" ".join(contexts))
    missing = [
        art for art in cited
        if not re.search(rf"(?:المادة|مادة)\s+{art}\b", blob)
    ]
    return len(missing) == 0, missing


def topic_match(question: str, sources: List[Dict]) -> bool:
    """True if any source's `legal_topic` (encyclopedia subdir name) appears in the question."""
    for s in sources:
        topic = s.get("legal_topic") or ""
        # Fallback to legal_category for back-compat (same field, older name).
        if not topic:
            cat = s.get("legal_category", "")
            if cat and cat != "general":
                topic = cat
        if topic and topic in question:
            return True
    return False


def compute_confidence(
    rerank_scores: List[float],
    source_count: int,
    article_validation_pass: bool,
    topic_match_hit: bool,
) -> Tuple[float, Dict]:
    """Heuristic confidence ∈ [0, 1] from rerank + source count + grounding signals."""
    w = config.CONFIDENCE_WEIGHTS

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    rerank_signal = max(0.0, min(1.0, mean(rerank_scores)))
    source_signal = min(1.0, source_count / 5.0)
    article_signal = 1.0 if article_validation_pass else 0.0
    topic_signal = 1.0 if topic_match_hit else 0.0

    score = (
        w["rerank"] * rerank_signal
        + w["source_count"] * source_signal
        + w["article_validation"] * article_signal
        + w["topic_match"] * topic_signal
    )
    score = max(0.0, min(1.0, score))

    return round(score, 3), {
        "rerank_signal": round(rerank_signal, 3),
        "source_count": source_count,
        "article_validation": "passed" if article_validation_pass else "failed",
        "topic_match": topic_match_hit,
    }
