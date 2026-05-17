"""Case weakness detection endpoint."""
import time
from fastapi import APIRouter

import config
from app.models import (
    WeaknessRequest, WeaknessResponse, SourceInfo, ConfidenceFactors,
)
from app.services.retrieval import retrieval_service
from app.services.llm import async_call_llm
from app.services.confidence import validate_evidence, topic_match, compute_confidence
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

router = APIRouter()


@router.post(
    "/weakness",
    response_model=WeaknessResponse,
    summary="Case Weakness Detection",
    description="Analyze an Egyptian criminal case and identify all prosecution weaknesses, with grounded citations and confidence.",
)
async def detect_weakness(req: WeaknessRequest):
    t0 = time.time()

    contexts, sources, _ = retrieval_service.retrieve(req.case_facts, k=7)
    legal_refs = "\n---\n".join(contexts)

    prompt = PROMPTS["weakness"].format(
        case_facts=req.case_facts,
        legal_refs=legal_refs,
    )
    analysis, _ = await async_call_llm(
        prompt, feature="weakness", system_msg=SYSTEM_MESSAGES["weakness"], max_tokens=1500,
    )

    article_pass, missing = validate_evidence(analysis, contexts)
    topic_hit = topic_match(req.case_facts, sources)
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
            "تنبيه: المواد التالية مذكورة في التحليل لكنها غير موجودة في المراجع: "
            + ", ".join(missing)
        )
        confidence = max(0.0, round(confidence - 0.3, 3))
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة التحليل منخفضة — يُنصح بإضافة مزيد من تفاصيل القضية.")

    return WeaknessResponse(
        analysis=analysis,
        confidence_score=confidence,
        confidence_factors=ConfidenceFactors(**factors),
        sources=[SourceInfo(**s) for s in sources],
        warnings=warnings,
        conflicts_detected=False,
        latency_ms=round((time.time() - t0) * 1000, 1),
        model=config.LLM_MODEL,
    )
