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


# Inputs that are not real legal questions — markdown headings, bullet markers,
# numbered list items left dangling, or single bare words. These were getting
# fabricated answers in evaluation (e.g. '## المستوى الأول', '* المخالفة',
# '1. ما المقصود بمبدأ:'). Catching them up front lets us ask for a real
# question instead of running RAG on garbage and trusting the LLM not to invent.
_MARKDOWN_HEADER_RE = re.compile(r'^\s*#{1,6}\s')
_BULLET_RE = re.compile(r'^\s*[\*\-•]\s')
_NUMBERED_FRAGMENT_RE = re.compile(r'^\s*\d+\s*[\.\)]\s')


def is_meaningful_query(text: str) -> bool:
    """Return False if `text` looks like a fragment, header, or bullet — not a question.

    A real question needs (a) at least a few Arabic letters and (b) no telltale
    markdown-listing prefix UNLESS it also contains a verb/interrogative or an
    explicit question mark. We don't require the question mark itself because
    Arabic legal questions are often phrased as declarative requests.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()

    # Must have enough Arabic content to be a real question. Strip whitespace
    # and punctuation, then require at least 8 Arabic-letter characters.
    arabic_chars = re.findall(r'[؀-ۿ]', stripped)
    if len(arabic_chars) < 8:
        return False

    # Markdown heading is never a question — reject unconditionally. Real users
    # don't type '##' to ask something; these come from pasted document outlines.
    if _MARKDOWN_HEADER_RE.match(stripped):
        return False

    # Strip a leading list/bullet marker for the rest of the checks — we want
    # to know whether anything substantive follows it.
    body = stripped
    if _BULLET_RE.match(body):
        body = _BULLET_RE.sub('', body, count=1)
    elif _NUMBERED_FRAGMENT_RE.match(body):
        body = _NUMBERED_FRAGMENT_RE.sub('', body, count=1)
    body = body.strip()

    # After stripping the marker, the body must itself be a substantive
    # question (≥3 tokens AND ≥12 Arabic chars). '* المخالفة' (1 token) and
    # '## المستوى الأول' (2 tokens, no verb) both fail here.
    body_arabic = re.findall(r'[؀-ۿ]', body)
    if len(body_arabic) < 12:
        return False
    if len(body.split()) < 3:
        return False

    # Sentence ending with a colon is a section header introducing what
    # follows — not a self-contained question (e.g. 'ما المقصود بمبدأ:').
    if body.rstrip().endswith(':') or body.rstrip().endswith('：'):
        return False

    return True


INCOMPLETE_QUERY_MESSAGE_AR = (
    "يبدو أن السؤال غير مكتمل أو غير واضح. يرجى صياغة سؤال قانوني كامل "
    "(مثال: «ما عقوبة السرقة بالإكراه في القانون المصري؟») لأتمكن من تقديم إجابة دقيقة."
)


def validate_evidence(answer: str, contexts: List[str]) -> Tuple[bool, List[str]]:
    """Check every article cited in `answer` appears in `contexts`. Returns (passed, missing_articles).

    Both sides use the same extractor so plural/list article forms ("المواد 211، 212، 213",
    "المواد: 315") are matched symmetrically — avoids false 'uncited article' flags."""
    cited = extract_article_references(answer)
    if not cited:
        return True, []
    context_articles = set(extract_article_references(" ".join(contexts)))
    missing = [art for art in cited if art not in context_articles]
    return len(missing) == 0, missing


# Court-of-Cassation precedent citation, e.g. "الطعن رقم 8875 لسنة 6",
# "طعن 1605 لسنة 55", "نقض رقم 141 لسنة 36". Captures (case_number, year).
_PRECEDENT_RE = re.compile(
    r'(?:ال)?(?:طعن|نقض)\s+(?:رقم\s+)?(\d{1,6})\s+لسن[ةه]\s+(\d{1,4})'
)


def validate_precedents(answer: str, contexts: List[str]) -> Tuple[bool, List[str]]:
    """Check every Court-of-Cassation precedent cited in `answer` (طعن/نقض رقم N لسنة M)
    is grounded in the retrieved `contexts`. Returns (passed, ungrounded_precedents).

    The article validator only checks 'المادة N'; precedent numbers are a separate,
    easily-hallucinated citation form, so they get their own grounding check. A precedent
    is considered grounded when its case number AND year both appear in the same retrieved
    chunk (digit-normalized)."""
    norm_answer = normalize_arabic_indic_digits(answer)
    cited = _PRECEDENT_RE.findall(norm_answer)
    if not cited:
        return True, []
    norm_contexts = [normalize_arabic_indic_digits(c) for c in contexts]
    missing: List[str] = []
    for num, year in cited:
        grounded = any(num in c and year in c for c in norm_contexts)
        label = f"طعن {num} لسنة {year}"
        if not grounded and label not in missing:
            missing.append(label)
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
    # Gamma-calibrate the (low) cross-encoder sigmoid so grounded answers don't read as
    # "low confidence". gamma=1.0 disables. See config.RERANK_CALIBRATION_GAMMA.
    gamma = getattr(config, "RERANK_CALIBRATION_GAMMA", 1.0)
    if gamma != 1.0 and rerank_signal > 0.0:
        rerank_signal = rerank_signal ** gamma
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
