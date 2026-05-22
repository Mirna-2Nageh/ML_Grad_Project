"""
Agentic self-check for generated legal memos.

Pipeline: draft → verify every cited article + legal argument against the retrieved
context → revise out anything unsupported. Closes the argument-level hallucination gap
that post-hoc article-validation alone cannot catch (it only flags fabricated article
*numbers*, not fabricated *reasoning*).

Used by the defense-memo endpoint; reusable for any grounded long-form generation.
"""
import logging
from typing import List, Tuple

import config
from app.services.llm import async_call_llm, is_llm_error
from app.services.confidence import validate_evidence
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logger = logging.getLogger(__name__)


async def self_check_memo(
    draft: str,
    legal_refs: str,
    case_facts: str,
    contexts: List[str],
    max_tokens: int = 2048,
) -> Tuple[str, str, int]:
    """Verify+revise a memo so its citations and arguments are grounded in `contexts`.

    Runs up to config.MEMO_SELF_CHECK_MAX_ITERS revision passes, stopping early once
    article-validation passes after at least one revision. Returns:
        (final_text, model_used_for_last_revision, revisions_applied)
    On any LLM failure it returns the best text so far (never raises).
    """
    text = draft
    model_used = ""
    revisions = 0

    for _ in range(max(1, config.MEMO_SELF_CHECK_MAX_ITERS)):
        article_pass, missing = validate_evidence(text, contexts)
        # Stop once citations are clean AND we've already done one grounding pass
        # (the first pass also re-checks arguments, not just article numbers).
        if article_pass and revisions >= 1:
            break

        flagged = ", ".join(missing) if missing else "لا يوجد مواد مُعلَّمة — تحقّق من صحة الحجج والوقائع فقط"
        prompt = PROMPTS["verify_memo"].format(
            legal_refs=legal_refs[:config.MAX_CONTEXT_CHARS],
            case_facts=case_facts,
            draft=text,
            flagged=flagged,
        )
        revised, _, m = await async_call_llm(
            prompt, feature="defense", system_msg=SYSTEM_MESSAGES["verify_memo"],
            max_tokens=max_tokens,
        )
        if is_llm_error(m) or not revised.strip():
            logger.warning("memo self-check revision failed; keeping prior draft")
            break

        # Strip an echoed prompt cue if the model prefixed it.
        revised = revised.strip()
        for cue in ("المذكرة بعد المراجعة:", "المذكرة بعد المراجعة"):
            if revised.startswith(cue):
                revised = revised[len(cue):].strip()
                break
        text = revised
        model_used = m
        revisions += 1

    return text, model_used, revisions
