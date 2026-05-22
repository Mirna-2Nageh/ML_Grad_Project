"""Legal Q&A endpoint."""
import time
from fastapi import APIRouter, HTTPException

import config
from app.models import (
    QARequest, QAResponse, SourceInfo, ConfidenceFactors, ErrorResponse,
)
from app.services.retrieval import retrieval_service
from app.services.llm import async_call_llm
from app.services.confidence import validate_evidence, topic_match, compute_confidence
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

router = APIRouter()


@router.post(
    "/qa",
    response_model=QAResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Legal Question Answering",
    description="Ask a legal question about Egyptian Criminal Law. Returns a grounded answer with citations and a confidence score.",
)
async def legal_qa(req: QARequest):
    t0 = time.time()

    # 1. Retrieve relevant context
    contexts, sources, timing = retrieval_service.retrieve(req.question, k=req.k)
    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # 2. Build prompt + generate
    context_str = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
    prompt_key = f"qa_{req.prompt_style}" if f"qa_{req.prompt_style}" in PROMPTS else "qa_restrictive"
    prompt = PROMPTS[prompt_key].format(context=context_str, question=req.question)

    answer, _, model_used = await async_call_llm(
        prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
    )

    # 3. Evidence validation + confidence scoring
    article_pass, missing_articles = validate_evidence(answer, contexts)
    topic_hit = topic_match(req.question, sources)
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
            "تنبيه: المواد التالية مذكورة في الإجابة لكنها غير موجودة في المصادر المسترجعة: "
            + ", ".join(missing_articles)
        )
        # Hard penalty for hallucinated articles — drops confidence by up to 0.3.
        confidence = max(0.0, round(confidence - 0.3, 3))
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة الإجابة منخفضة — يُنصح بإعادة صياغة السؤال بمزيد من التفصيل أو إضافة سياق إضافي.")

    total_ms = (time.time() - t0) * 1000
    return QAResponse(
        answer=answer,
        confidence_score=confidence,
        confidence_factors=ConfidenceFactors(**factors),
        sources=[SourceInfo(**s) for s in sources],
        warnings=warnings,
        conflicts_detected=False,
        latency_ms=round(total_ms, 1),
        retrieval_ms=round(timing.get("total_ms", 0), 1),
        model=model_used,
    )
