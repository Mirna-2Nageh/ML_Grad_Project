"""Programmatic answer cleanup applied after the LLM returns.

Two purposes:
 1. Strip casual conversational openers (حسناً، بالتأكيد، تمام...) that the
    "Nour" persona prompt forbids but Gemini/Groq still emit ~30% of the time.
 2. Detect when the answer is a refusal (no legal substance, only the leaked
    template phrase) so confidence scoring can downgrade it instead of treating
    it as a grounded answer.

Pure functions, no IO, no LLM. Wired into qa.py + chat.py after the final
answer (post-retry) is selected.
"""
import re
from typing import Tuple

# Casual openers — Nour persona forbids these but the LLM emits them anyway.
# Order matters: longer phrases first so "سوف أجيب" doesn't leave a stranded "سوف".
_CASUAL_OPENERS = [
    'سوف أجاوب على أسئلتك بتفصيل',
    'سأجيب على سؤالك بتفصيل',
    'سوف أجيب على سؤالك',
    'سأجيب على سؤالك',
    'سوف أجيب',
    'سأجيب',
    'سوف أبدأ',
    'سأبدأ',
    'لقد قرأت السؤال',
    'فيما يتعلق بسؤالك',
    'فيما يتعلق بالسؤال',
    'بالنسبة لسؤالك',
    'بالتأكيد',
    'تمامًا',
    'تماماً',
    'تمام',
    'طبعاً',
    'طبعا',
    'حسناً',
    'حسنًا',
    'حسنا',
]

# Standalone-line phrases that signal the answer is a refusal / not-found
# response (rather than a real grounded answer). When detected, confidence
# should be reduced and the user should be told the question's articles
# aren't in the index — not given a false sense of grounding.
_REFUSAL_MARKERS = [
    'يبدو أن السؤال غير مكتمل',
    'يرجى صياغة سؤال قانوني كامل',
    'المادة المطلوبة غير متوفرة في السياق',
    'لا تتضمن النصوص المقدمة',
    'المعلومات المطلوبة غير متوفرة',
    'لا يوجد سياق قانوني',
    'غير متوفرة في قاعدة البيانات',
]

# Awkward sentence-internal occurrences of the refusal template phrase.
# When the LLM embeds "المادة المطلوبة غير متوفرة في السياق المقدم" mid-clause
# (e.g. "وفقاً للمادة المطلوبة غير متوفرة في السياق المقدم، يبدو..."), rewrite
# it to a standalone sentence so the user gets a clean refusal.
#
# Arabic prefix-contraction note: ل + ال → لل (the alef of ال drops). So
# "للمادة" is character-wise ل ل م ا د ة (no alef). The regex must handle BOTH
# explicit-article forms (المادة) and contracted-prefix forms (للمادة / بالمادة
# / والمادة). The verbal lead-in (وفقاً / تنص / بناءً على / طبقاً) is consumed
# along with its trailing whitespace so no stranded "ل" is left behind.
_EMBEDDED_REFUSAL_RE = re.compile(
    r'(?:وفقاً|تنص|بناءً\s+على|طبقاً|استناداً\s+إلى|بناء\s+على)?\s*'
    r'(?:لل|بال|وال|ال|لـ|ل|ب|و)?(?:مادة|مواد)\s+المطلوبة\s+'
    r'غير\s+متوفر[ةت]?\s+في\s+السياق\s*(?:المقدم)?\s*[،,\.]?\s*'
)


def strip_casual_openers(text: str) -> str:
    """Remove leading conversational fillers ('حسناً', 'بالتأكيد', 'سوف أجيب')."""
    if not text:
        return text
    out = text.lstrip()
    # Try removing up to 2 openers in sequence (LLMs sometimes stack them).
    for _ in range(2):
        removed = False
        for opener in _CASUAL_OPENERS:
            if out.startswith(opener):
                rest = out[len(opener):]
                # Trim leading punctuation + whitespace after the opener
                rest = re.sub(r'^[\s،.,؛:!\.]+', '', rest)
                if rest:  # don't strip if it leaves an empty string
                    out = rest
                    removed = True
                    break
        if not removed:
            break
    # Capitalize-equivalent for Arabic: ensure the answer starts with an Arabic letter
    # not a leftover space or punctuation.
    out = re.sub(r'^[\s،.,؛:!\.]+', '', out)
    return out or text


def normalize_refusal_phrase(text: str) -> str:
    """Rewrite the leaky 'المادة المطلوبة غير متوفرة في السياق' template phrase
    when it appears mid-sentence so the answer reads as a clean refusal.

    Replaces the embedded form with a standalone sentence + period.
    """
    if not text:
        return text
    # If the phrase appears mid-clause, replace it with a clean sentence break.
    rewritten, n = _EMBEDDED_REFUSAL_RE.subn(
        'النصوص المقدمة لا تتضمن المادة المطلوبة. ',
        text,
    )
    if n == 0:
        return text
    # Collapse adjacent duplicated punctuation/whitespace left behind by the
    # substitution (e.g. "المطلوبة. . على" -> "المطلوبة. على").
    rewritten = re.sub(r'([،,\.])\s*[،,\.]+', r'\1', rewritten)
    rewritten = re.sub(r'\s{2,}', ' ', rewritten)
    return rewritten


def looks_like_refusal(text: str) -> bool:
    """True if the answer is dominated by refusal-marker phrases, i.e. the LLM
    is saying 'I don't have this' rather than providing legal substance.

    Heuristic: a refusal appears in the first 200 chars AND the total answer
    is short (< 400 chars). Long answers that happen to contain a refusal
    marker in passing are not refusals — those are partial answers, leave alone.
    """
    if not text:
        return True
    head = text[:200]
    has_marker = any(m in head for m in _REFUSAL_MARKERS)
    return has_marker and len(text) < 400


def postprocess_answer(text: str) -> Tuple[str, bool]:
    """Combined cleanup pipeline.

    Returns (cleaned_text, was_refusal). Order:
      1. strip casual openers (no semantic change)
      2. rewrite embedded refusal template phrase
      3. classify the final text as refusal vs. grounded
    """
    cleaned = strip_casual_openers(text)
    cleaned = normalize_refusal_phrase(cleaned)
    return cleaned, looks_like_refusal(cleaned)
