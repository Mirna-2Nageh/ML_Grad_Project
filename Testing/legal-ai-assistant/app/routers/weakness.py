"""Case weakness detection endpoint.

Runs the multi-agent pipeline (Document Analysis → Weakness Detection → Legal
Research/Precedent Retrieval → Defense Strategy → Weakness-Analysis synthesis) when
config.USE_AGENTIC_PIPELINE is on, and falls back to the single-shot path automatically
if any agent fails. Both paths share the same evidence-validation + confidence scoring.
"""
import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException

import config
from app.models import (
    WeaknessRequest, WeaknessResponse, SourceInfo, ConfidenceFactors,
)
from app.services.retrieval import retrieval_service
from app.services.llm import (
    async_call_llm, used_fallback, is_llm_error, FALLBACK_NOTICE_AR, LLM_ERROR_NOTICE_AR,
)
from app.services.confidence import validate_evidence, validate_precedents, topic_match, compute_confidence
from app.services import agents
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logger = logging.getLogger(__name__)
router = APIRouter()


async def _single_shot(req: WeaknessRequest):
    """Legacy single-shot path: one retrieve → one LLM call. Returns
    (text, contexts, sources, model)."""
    contexts, sources, _ = await asyncio.to_thread(
        retrieval_service.retrieve, req.case_facts, k=7
    )
    legal_refs = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
    prompt = PROMPTS["weakness"].format(
        case_facts=req.case_facts,
        evidence=req.evidence or "لم تُقدَّم أدلة إضافية بخلاف ما ورد في الوقائع.",
        defendant_statement=req.defendant_statement or "لم يُقدَّم بيان منفصل للمتهم.",
        legal_refs=legal_refs,
    )
    text, _, model = await async_call_llm(
        prompt, feature="weakness", system_msg=SYSTEM_MESSAGES["weakness"], max_tokens=2048,
    )
    return text, contexts, sources, model


@router.post(
    "/weakness",
    response_model=WeaknessResponse,
    summary="Case Weakness Detection",
    description="Analyze an Egyptian criminal case and identify all prosecution weaknesses, with grounded citations and confidence.",
)
async def detect_weakness(req: WeaknessRequest):
    t0 = time.time()
    extra_warnings: list = []
    pipeline_used = False
    timeline = None
    weaknesses_detected = None

    if config.USE_AGENTIC_PIPELINE:
        try:
            res = await agents.run_weakness_pipeline(
                req.case_facts, req.evidence or "", req.defendant_statement or "",
            )
            text, contexts, sources, model_used = (
                res["text"], res["contexts"], res["sources"], res["model"],
            )
            pipeline_used = True
            timeline = (res.get("analysis") or {}).get("timeline")
            weaknesses_detected = res.get("weaknesses")
        except agents.AgentPipelineError as e:
            logger.warning(f"Agentic weakness pipeline failed ({e}); falling back to single-shot")
            extra_warnings.append("تعذّر تشغيل خط التحليل متعدد الوكلاء؛ تم استخدام التحليل المباشر.")
            text, contexts, sources, model_used = await _single_shot(req)
    else:
        text, contexts, sources, model_used = await _single_shot(req)

    if is_llm_error(model_used):
        raise HTTPException(status_code=503, detail=LLM_ERROR_NOTICE_AR)

    # Evidence validation + confidence — identical for both paths.
    article_pass, missing = validate_evidence(text, contexts)
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
            "تنبيه: المواد التالية مذكورة في التحليل لكنها غير موجودة في المراجع: "
            + ", ".join(missing)
        )
        confidence = max(0.0, round(confidence - 0.3, 3))
    # Precedent-citation grounding: flag طعن/نقض numbers not found in the retrieved context.
    prec_pass, missing_precedents = validate_precedents(text, contexts)
    if not prec_pass:
        warnings.append(
            "تنبيه: السوابق القضائية التالية مذكورة في التحليل لكنها غير موجودة في المراجع المسترجعة "
            "(قد تكون غير دقيقة): " + "، ".join(missing_precedents)
        )
        confidence = max(0.0, round(confidence - 0.2, 3))
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة التحليل منخفضة — يُنصح بإضافة مزيد من تفاصيل القضية.")
    if used_fallback(model_used):
        warnings.append(FALLBACK_NOTICE_AR)

    return WeaknessResponse(
        analysis=text,
        confidence_score=confidence,
        confidence_factors=ConfidenceFactors(**factors),
        sources=[SourceInfo(**s) for s in sources],
        warnings=warnings,
        conflicts_detected=False,
        latency_ms=round((time.time() - t0) * 1000, 1),
        model=model_used,
        pipeline=pipeline_used,
        timeline=timeline,
        weaknesses_detected=weaknesses_detected,
    )
