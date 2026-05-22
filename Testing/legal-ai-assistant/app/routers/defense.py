"""Defense memorandum generation endpoint."""
import time
from fastapi import APIRouter

import config
from app.models import (
    DefenseRequest, DefenseResponse, SourceInfo, ConfidenceFactors,
)
from app.services.retrieval import retrieval_service
from app.services.llm import async_call_llm, used_fallback, FALLBACK_NOTICE_AR
from app.services.confidence import validate_evidence, topic_match, compute_confidence
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

router = APIRouter()


@router.post(
    "/defense",
    response_model=DefenseResponse,
    summary="Defense Memorandum Generation",
    description="Generate a formal defense memorandum in Arabic legal language with grounded citations and confidence.",
)
async def generate_defense(req: DefenseRequest):
    t0 = time.time()

    contexts, sources, _ = retrieval_service.retrieve(req.case_facts, k=7)
    legal_refs = "\n---\n".join(contexts)

    prompt = PROMPTS["defense"].format(
        case_facts=req.case_facts,
        weaknesses=req.weaknesses or "لم يتم تحديد نقاط ضعف محددة",
        legal_refs=legal_refs,
    )
    memorandum, _, model_used = await async_call_llm(
        prompt, feature="defense", system_msg=SYSTEM_MESSAGES["defense"], max_tokens=2048,
    )

    article_pass, missing = validate_evidence(memorandum, contexts)
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
            "تنبيه: المواد التالية مذكورة في المذكرة لكنها غير موجودة في المراجع المسترجعة: "
            + ", ".join(missing)
        )
        confidence = max(0.0, round(confidence - 0.3, 3))
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
    )
