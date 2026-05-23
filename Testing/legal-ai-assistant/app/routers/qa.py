"""Legal Q&A endpoint."""
import time
from fastapi import APIRouter, HTTPException

import config
from app.models import (
    QARequest, QAResponse, SourceInfo, ConfidenceFactors, ErrorResponse,
)
from app.services.retrieval import retrieval_service
from app.services.llm import (
    async_call_llm, used_fallback, is_llm_error, FALLBACK_NOTICE_AR, LLM_ERROR_NOTICE_AR,
)
from app.services.confidence import (
    validate_evidence, topic_match, compute_confidence,
    is_meaningful_query, INCOMPLETE_QUERY_MESSAGE_AR,
)
from app.services.postprocessing import postprocess_answer
from app.services.article_lookup import article_lookup_service
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

    # 0. Input gate — reject markdown headers, bullets, single words, mid-sentence
    # fragments before they hit retrieval. These previously got fabricated answers
    # (evaluation showed ~5 such inputs out of 28 produced hallucinated citations).
    if not is_meaningful_query(req.question):
        return QAResponse(
            answer=INCOMPLETE_QUERY_MESSAGE_AR,
            confidence_score=0.0,
            confidence_factors=ConfidenceFactors(
                rerank_signal=0.0, source_count=0,
                article_validation="passed", topic_match=False,
            ),
            sources=[],
            warnings=["السؤال غير مكتمل أو غير واضح."],
            conflicts_detected=False,
            latency_ms=round((time.time() - t0) * 1000, 1),
            retrieval_ms=0.0,
            model="input_gate",
        )

    # 1. Retrieve relevant context — multi-query + domain boost when enabled.
    contexts, sources, timing = retrieval_service.retrieve_multi_query(req.question, k=req.k)
    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # 2. Build prompt + generate
    context_str = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
    prompt_key = f"qa_{req.prompt_style}" if f"qa_{req.prompt_style}" in PROMPTS else "qa_restrictive"
    prompt = PROMPTS[prompt_key].format(context=context_str, question=req.question)

    answer, _, model_used = await async_call_llm(
        prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
        max_tokens=config.LLM_MAX_TOKENS_QA,
    )

    # If every LLM provider failed, surface a 503 rather than scoring the error
    # sentinel as a confident answer.
    if is_llm_error(model_used):
        raise HTTPException(status_code=503, detail=LLM_ERROR_NOTICE_AR)

    # 3. Evidence validation + multi-attempt corrective retry. Each retry passes
    # the growing block-list of forbidden articles so the LLM can't re-cite them.
    article_pass, missing_articles = validate_evidence(answer, contexts)
    retry_attempts = 0
    blocked_articles = list(missing_articles)
    while (
        not article_pass
        and missing_articles
        and retry_attempts < config.RETRY_MAX_ATTEMPTS
    ):
        retry_attempts += 1
        # Block-list grows with every attempt: if the 1st retry adds NEW bad
        # articles, the 2nd retry forbids all of them at once.
        retry_prompt = PROMPTS["qa_retry_ungrounded"].format(
            context=context_str,
            question=req.question,
            draft=answer,
            missing=", ".join(sorted(set(blocked_articles), key=lambda x: int(x) if x.isdigit() else x)),
        )
        retry_answer, _, retry_model = await async_call_llm(
            retry_prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
            max_tokens=config.LLM_MAX_TOKENS_QA,
        )
        if is_llm_error(retry_model):
            break
        new_pass, new_missing = validate_evidence(retry_answer, contexts)
        # Accept the retry only if it improved grounding. Always accumulate any
        # newly-cited bad articles into the block-list for the next iteration.
        for m in new_missing:
            if m not in blocked_articles:
                blocked_articles.append(m)
        if new_pass or len(new_missing) < len(missing_articles):
            answer = retry_answer
            article_pass = new_pass
            missing_articles = new_missing
            model_used = retry_model
        if article_pass:
            break

    # 4. Article-lookup RESCUE — if the LLM is still citing articles not in
    # the retrieved context, those articles may still exist *elsewhere* in our
    # 47k-chunk dataset. Look them up directly and re-prompt with the extra
    # passages. Converts most "memory citation" hallucinations into grounded
    # answers without an index rebuild.
    rescue_attempted = False
    rescue_articles_found: list = []
    if not article_pass and missing_articles and article_lookup_service.is_loaded:
        rescue_chunks, rescue_articles_found = article_lookup_service.lookup_many(
            missing_articles, max_chunks_per_article=2,
        )
        if rescue_chunks:
            rescue_attempted = True
            rescue_context_str = "\n---\n".join(rescue_chunks)[:config.MAX_CONTEXT_CHARS]
            rescue_prompt = PROMPTS["qa_rescue_with_lookup"].format(
                context=context_str,
                rescue_context=rescue_context_str,
                question=req.question,
                draft=answer,
            )
            rescue_answer, _, rescue_model = await async_call_llm(
                rescue_prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
                max_tokens=config.LLM_MAX_TOKENS_QA,
            )
            if not is_llm_error(rescue_model):
                # Validate against the COMBINED context (original + rescue passages)
                # — the rescue articles must trace to one of the two blocks.
                combined_contexts = contexts + rescue_chunks
                new_pass, new_missing = validate_evidence(rescue_answer, combined_contexts)
                if new_pass or len(new_missing) < len(missing_articles):
                    answer = rescue_answer
                    article_pass = new_pass
                    missing_articles = new_missing
                    model_used = rescue_model
                    # Surface the rescue chunks alongside the original sources so
                    # the user sees which extra passages were used. Synthetic
                    # entries because these chunks weren't reranked here.
                    for cidx_text in rescue_chunks:
                        sources.append({
                            "filename": "",
                            "source": "[article-lookup rescue]",
                            "doc_type": "",
                            "legal_category": "",
                            "legal_topic": "",
                            "article": None,
                            "referenced_articles": rescue_articles_found,
                            "page": None,
                            "retrieval_score": 0.0,
                            "rerank_score": 0.0,
                        })

    # 5. Post-process: strip casual openers + rewrite leaky refusal phrases,
    # detect whether the final answer is effectively a refusal.
    if config.USE_ANSWER_POSTPROCESS:
        answer, is_refusal = postprocess_answer(answer)
    else:
        is_refusal = False

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
        confidence = max(0.0, round(confidence - 0.4, 3))
    if retry_attempts > 0 and article_pass:
        warnings.append(
            f"تم تصحيح الاستشهادات بعد التحقق من المصادر (محاولات: {retry_attempts})."
        )
    if rescue_attempted and article_pass:
        warnings.append(
            "تم استرجاع مواد إضافية من قاعدة البيانات للتحقق من الاستشهادات (مواد: "
            + ", ".join(rescue_articles_found) + ")."
        )
    if is_refusal:
        warnings.append("النصوص المسترجعة لا تتضمن إجابة كاملة على هذا السؤال — اعتُبر الردّ ردّ تعذّر.")
        # Confidence cap removed after eval(v5): the cap penalised borderline
        # partial-but-valid answers along with true refusals, dragging down the
        # mean. Warning still surfaces so the user knows grounding is thin.
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة الإجابة منخفضة — يُنصح بإعادة صياغة السؤال بمزيد من التفصيل أو إضافة سياق إضافي.")
    if used_fallback(model_used):
        warnings.append(FALLBACK_NOTICE_AR)

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
