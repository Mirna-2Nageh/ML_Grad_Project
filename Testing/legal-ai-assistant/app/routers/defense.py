"""Defense memorandum generation endpoint.

Runs the multi-agent pipeline (Document Analysis → Weakness Detection → Legal
Research/Precedent Retrieval → Defense Strategy → Memorandum synthesis) when
config.USE_AGENTIC_PIPELINE is on, falling back to the single-shot path if any agent
fails. The agentic memo (or the single-shot memo) then passes through the same
self-check + evidence-validation + confidence scoring as before.
"""
import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException

import config
from app.models import (
    DefenseRequest, DefenseResponse, SourceInfo, ConfidenceFactors,
)
from app.services.retrieval import retrieval_service
from app.services.llm import (
    async_call_llm, used_fallback, is_llm_error, FALLBACK_NOTICE_AR, LLM_ERROR_NOTICE_AR,
)
from app.services.confidence import validate_evidence, validate_precedents, topic_match, compute_confidence
from app.services.memo_agent import self_check_memo
from app.services import agents
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logger = logging.getLogger(__name__)
router = APIRouter()


async def _single_shot(req: DefenseRequest):
    """Legacy single-shot path. Returns (memo, contexts, sources, model, legal_refs)."""
    contexts, sources, _ = await asyncio.to_thread(
        retrieval_service.retrieve, req.case_facts, k=7
    )
    legal_refs = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
    prompt = PROMPTS["defense"].format(
        case_facts=req.case_facts,
        evidence=req.evidence or "لم تُقدَّم أدلة إضافية بخلاف ما ورد في الوقائع.",
        defendant_statement=req.defendant_statement or "لم يُقدَّم بيان منفصل للمتهم.",
        weaknesses=req.weaknesses or "لم يتم تحديد نقاط ضعف محددة",
        legal_refs=legal_refs,
    )
    memo, _, model = await async_call_llm(
        prompt, feature="defense", system_msg=SYSTEM_MESSAGES["defense"], max_tokens=2048,
    )
    return memo, contexts, sources, model, legal_refs


@router.post(
    "/defense",
    response_model=DefenseResponse,
    summary="Defense Memorandum Generation",
    description="Generate a formal defense memorandum in Arabic legal language with grounded citations and confidence.",
)
async def generate_defense(req: DefenseRequest):
    t0 = time.time()
    extra_warnings: list = []
    pipeline_used = False
    timeline = None
    weaknesses_detected = None

    if config.USE_AGENTIC_PIPELINE:
        try:
            case_facts = req.case_facts
            if req.weaknesses:
                case_facts = f"{case_facts}\n\n[نقاط ضعف يقترحها المستخدم]:\n{req.weaknesses}"
            res = await agents.run_defense_pipeline(
                case_facts, req.evidence or "", req.defendant_statement or "",
            )
            memorandum, contexts, sources, model_used = (
                res["text"], res["contexts"], res["sources"], res["model"],
            )
            legal_refs = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
            pipeline_used = True
            timeline = (res.get("analysis") or {}).get("timeline")
            weaknesses_detected = res.get("weaknesses")
        except agents.AgentPipelineError as e:
            logger.warning(f"Agentic defense pipeline failed ({e}); falling back to single-shot")
            extra_warnings.append("تعذّر تشغيل خط الصياغة متعدد الوكلاء؛ تم استخدام الصياغة المباشرة.")
            memorandum, contexts, sources, model_used, legal_refs = await _single_shot(req)
    else:
        memorandum, contexts, sources, model_used, legal_refs = await _single_shot(req)

    if is_llm_error(model_used):
        raise HTTPException(status_code=503, detail=LLM_ERROR_NOTICE_AR)

    # Agentic self-check: verify citations + arguments against context, revise out
    # anything unsupported. Validation/confidence below run on the REVISED memo.
    self_check_revisions = 0
    if config.MEMO_SELF_CHECK:
        memorandum, sc_model, self_check_revisions = await self_check_memo(
            memorandum, legal_refs, req.case_facts, contexts,
        )
        if sc_model:
            model_used = sc_model

    article_pass, missing = validate_evidence(memorandum, contexts)
    topic_hit = topic_match(req.case_facts, sources)
    rerank_scores = [s.get("rerank_score", 0.0) for s in sources]
    confidence, factors = compute_confidence(
        rerank_scores=rerank_scores,
        source_count=len(sources),
        article_validation_pass=article_pass,
        topic_match_hit=topic_hit,
    )

    warnings = list(extra_warnings)
    if not article_pass:
        warnings.append(
            "تنبيه: المواد التالية مذكورة في المذكرة لكنها غير موجودة في المراجع المسترجعة: "
            + ", ".join(missing)
        )
        confidence = max(0.0, round(confidence - 0.3, 3))
    # Precedent-citation grounding: flag طعن/نقض numbers not found in the retrieved context.
    prec_pass, missing_precedents = validate_precedents(memorandum, contexts)
    if not prec_pass:
        warnings.append(
            "تنبيه: السوابق القضائية التالية مذكورة في المذكرة لكنها غير موجودة في المراجع المسترجعة "
            "(قد تكون غير دقيقة): " + "، ".join(missing_precedents)
        )
        confidence = max(0.0, round(confidence - 0.2, 3))
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة المذكرة منخفضة — يُنصح بمراجعة بشرية قبل الاعتماد عليها.")
    if used_fallback(model_used):
        warnings.append(FALLBACK_NOTICE_AR)

    return DefenseResponse(
        memorandum=memorandum,
        confidence_score=confidence,
        confidence_factors=ConfidenceFactors(**factors),
        sources=[SourceInfo(**s) for s in sources],
        warnings=warnings,
        conflicts_detected=False,
        latency_ms=round((time.time() - t0) * 1000, 1),
        model=model_used,
        self_check_revisions=self_check_revisions,
        pipeline=pipeline_used,
        timeline=timeline,
        weaknesses_detected=weaknesses_detected,
    )
