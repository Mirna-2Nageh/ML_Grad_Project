"""Forensic-consistency analysis endpoint.

Cross-references case facts + provided evidence against the retrieved legal corpus to
flag internal contradictions, evidence/fact mismatches, unmet legal elements, and gaps.
This is documentary/logical analysis (not physical forensics). The `evidence` field is
free text today; it will be fed from the per-case evidence store once uploads land.
"""
import time
from fastapi import APIRouter, HTTPException

import config
from app.models import ForensicRequest, ForensicResponse, SourceInfo, ConfidenceFactors
from app.services.retrieval import retrieval_service
from app.services.llm import (
    async_call_llm, used_fallback, is_llm_error, FALLBACK_NOTICE_AR, LLM_ERROR_NOTICE_AR,
)
from app.services.confidence import validate_evidence, topic_match, compute_confidence
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

router = APIRouter()


def _parse_conflict_marker(text: str):
    """Split the leading 'CONFLICT: YES|NO' marker from the analysis body.
    Returns (conflicts_detected, analysis_without_marker)."""
    lines = text.lstrip().splitlines()
    if lines and lines[0].strip().upper().startswith("CONFLICT:"):
        flag = "YES" in lines[0].upper()
        return flag, "\n".join(lines[1:]).strip()
    # Marker missing — fall back to scanning the whole text.
    return ("CONFLICT: YES" in text.upper()), text.strip()


@router.post(
    "/forensic",
    response_model=ForensicResponse,
    summary="Forensic-Consistency Analysis",
    description="Cross-reference case facts and evidence against the legal corpus to flag contradictions, evidence mismatches, unmet legal elements, and gaps.",
)
async def forensic_analysis(req: ForensicRequest):
    t0 = time.time()

    # Retrieve law relevant to both the facts and the evidence.
    retrieval_query = (req.case_facts + " " + req.evidence).strip()
    contexts, sources, _ = retrieval_service.retrieve(retrieval_query, k=7)
    legal_refs = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]

    prompt = PROMPTS["forensic"].format(
        legal_refs=legal_refs,
        case_facts=req.case_facts,
        evidence=req.evidence or "لا توجد أدلة إضافية مقدمة",
    )
    raw, _, model_used = await async_call_llm(
        prompt, feature="forensic", system_msg=SYSTEM_MESSAGES["forensic"], max_tokens=2048,
    )
    if is_llm_error(model_used):
        raise HTTPException(status_code=503, detail=LLM_ERROR_NOTICE_AR)

    conflicts_detected, analysis = _parse_conflict_marker(raw)

    article_pass, missing = validate_evidence(analysis, contexts)
    topic_hit = topic_match(retrieval_query, sources)
    rerank_scores = [s.get("rerank_score", 0.0) for s in sources]
    confidence, factors = compute_confidence(
        rerank_scores=rerank_scores,
        source_count=len(sources),
        article_validation_pass=article_pass,
        topic_match_hit=topic_hit,
    )

    warnings = []
    if not article_pass:
        warnings.append(
            "تنبيه: المواد التالية مذكورة في التحليل لكنها غير موجودة في المراجع: " + ", ".join(missing)
        )
        confidence = max(0.0, round(confidence - 0.3, 3))
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة التحليل منخفضة — يُنصح بإضافة مزيد من تفاصيل القضية أو الأدلة.")
    if used_fallback(model_used):
        warnings.append(FALLBACK_NOTICE_AR)

    return ForensicResponse(
        analysis=analysis,
        conflicts_detected=conflicts_detected,
        confidence_score=confidence,
        confidence_factors=ConfidenceFactors(**factors),
        sources=[SourceInfo(**s) for s in sources],
        warnings=warnings,
        latency_ms=round((time.time() - t0) * 1000, 1),
        model=model_used,
    )
