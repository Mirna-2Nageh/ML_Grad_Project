"""Legal Q&A endpoint."""
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

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
from app.services.upload_helper import (
    parse_uploaded_file, parse_uploaded_text, format_doc_for_context,
)
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

router = APIRouter()


async def _run_qa(
    question: str,
    k: int,
    prompt_style: str,
    attached_doc_text: Optional[str] = None,
    attached_doc_filename: str = "",
    attached_doc_content_type: str = "",
    extra_warnings: Optional[list] = None,
) -> QAResponse:
    """Core QA pipeline — shared by /qa (JSON) and /qa/upload (multipart).

    When `attached_doc_text` is non-empty, the document is wrapped in the same
    delimited block /chat/attach uses and prepended to the retrieved context,
    so the LLM treats it as authoritative material AND the evidence validator
    will accept article numbers cited from it.

    Skipping the input-gate when an attachment is present is intentional: a
    user uploading a contract and asking "لخّص" is a meaningful request even
    though "لخّص" alone would be rejected as a fragment.
    """
    t0 = time.time()
    has_attachment = bool(attached_doc_text and attached_doc_text.strip())

    # 0. Input gate — only enforced when no attachment provides standalone meaning.
    if not has_attachment and not is_meaningful_query(question):
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
    contexts, sources, timing = retrieval_service.retrieve_multi_query(question, k=k)
    # If retrieval found nothing AND there's no attachment, we have no grounding
    # to work with. With an attachment, we can still answer from that alone.
    if not contexts and not has_attachment:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # 2. Inject the attached doc at the TOP of the context block. The "[المستند
    # المرفق]" delimiter mirrors the session-attachment marker so the LLM can
    # tell where user-supplied material ends and retrieved corpus begins.
    if has_attachment:
        doc_block = format_doc_for_context(
            attached_doc_text, attached_doc_filename, attached_doc_content_type,
        )
        contexts = [doc_block] + contexts

    # 3. Build prompt + generate
    context_str = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
    prompt_key = f"qa_{prompt_style}" if f"qa_{prompt_style}" in PROMPTS else "qa_restrictive"
    prompt = PROMPTS[prompt_key].format(context=context_str, question=question)

    answer, _, model_used = await async_call_llm(
        prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
        max_tokens=config.LLM_MAX_TOKENS_QA,
    )

    # If every LLM provider failed, surface a 503 rather than scoring the error
    # sentinel as a confident answer.
    if is_llm_error(model_used):
        raise HTTPException(status_code=503, detail=LLM_ERROR_NOTICE_AR)

    # 3. Evidence validation.
    article_pass, missing_articles = validate_evidence(answer, contexts)

    # 3a. ITERATIVE RETRIEVAL — if validation flagged articles that aren't in
    # the retrieved chunks, widen the retrieval window before paying for a
    # corrective LLM retry. The original LLM answer often correctly identifies
    # the relevant article; we just didn't show that article to the validator.
    # Expanding k surfaces additional chunks; if any of them contain the
    # missing articles, validation flips to passed for free (no LLM call).
    iter_expansions: list = []  # k values we expanded to (for logging only)
    if (
        not article_pass
        and missing_articles
        and config.USE_ITERATIVE_RETRIEVAL
    ):
        seen_contexts = set(contexts)
        for next_k in [n for n in config.ITERATIVE_K_SEQUENCE if n > k]:
            extra_contexts, extra_sources, _ = retrieval_service.retrieve_multi_query(
                question, k=next_k,
            )
            new_chunks = [c for c in extra_contexts if c not in seen_contexts]
            if not new_chunks:
                continue
            iter_expansions.append(next_k)
            contexts = contexts + new_chunks
            seen_contexts.update(new_chunks)
            # Append the matching source records too so the UI shows where the
            # expansion-found chunks came from. Dedup by (source, retrieval_score)
            # rather than full dict equality because lists aren't hashable.
            seen_src_keys = {(s.get("source"), s.get("retrieval_score")) for s in sources}
            for s in extra_sources:
                key = (s.get("source"), s.get("retrieval_score"))
                if key not in seen_src_keys:
                    sources.append(s)
                    seen_src_keys.add(key)
            # Rebuild context_str so subsequent retry / rescue use the wider window.
            context_str = "\n---\n".join(contexts)[:config.MAX_CONTEXT_CHARS]
            article_pass, missing_articles = validate_evidence(answer, contexts)
            if article_pass:
                break

    # 4. Multi-attempt corrective retry. Each retry passes the growing
    # block-list of forbidden articles so the LLM can't re-cite them.
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
            question=question,
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
                question=question,
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

    topic_hit = topic_match(question, sources)
    rerank_scores = [s.get("rerank_score", 0.0) for s in sources]
    confidence, factors = compute_confidence(
        rerank_scores=rerank_scores,
        source_count=len(sources),
        article_validation_pass=article_pass,
        topic_match_hit=topic_hit,
    )

    warnings = list(extra_warnings or [])
    if has_attachment:
        warnings.append(
            f"تم تضمين المستند المرفق ({attached_doc_filename or 'مستند'}, "
            f"{len(attached_doc_text)} حرفاً) في السياق."
        )
    if iter_expansions:
        warnings.append(
            "تم توسيع نطاق البحث تلقائياً إلى "
            f"{iter_expansions[-1]} مرجعاً للعثور على الاستشهادات المطلوبة."
        )
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


# ──────────────────────────────────────────────
# Public endpoints — both delegate to _run_qa
# ──────────────────────────────────────────────

@router.post(
    "/qa",
    response_model=QAResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Legal Question Answering",
    description="Ask a legal question about Egyptian Criminal Law. Returns a grounded answer with citations and a confidence score.",
)
async def legal_qa(req: QARequest):
    return await _run_qa(
        question=req.question,
        k=req.k,
        prompt_style=req.prompt_style,
    )


@router.post(
    "/qa/upload",
    response_model=QAResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Legal Q&A with attached document (per-question context)",
    description=(
        "Same as `/qa` but accepts an attached document (.txt / .pdf / .docx) "
        "OR a pasted-text string, parsed and prepended to the retrieved legal "
        "context for THIS request only (not persisted; not added to the index).\n\n"
        "Use this when the user has a specific contract / case file / legal "
        "memo they want analyzed in light of Egyptian criminal law. Article "
        "numbers cited in the answer can come from EITHER the attachment or "
        "the retrieved corpus — the evidence validator checks both.\n\n"
        "Send as `multipart/form-data` with form fields: `question` (required), "
        "`file` (one of .txt/.pdf/.docx, optional), `text` (raw string, optional), "
        "`k` (int, default 7), `prompt_style` (default 'restrictive'). "
        "Provide either `file` OR `text`; if both, `file` wins."
    ),
)
async def legal_qa_upload(
    question: str = Form(..., min_length=1, max_length=2000),
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    k: int = Form(default=7, ge=1, le=20),
    prompt_style: str = Form(default="restrictive"),
):
    # Parse whichever payload was given (file wins if both).
    if file is not None and file.filename:
        cleaned, fname, ctype, warnings = await parse_uploaded_file(file)
    elif text:
        cleaned, fname, ctype, warnings = parse_uploaded_text(text)
    else:
        cleaned, fname, ctype, warnings = "", "", "", []

    if (file is not None and file.filename or text) and not cleaned:
        # User TRIED to attach something but parsing failed — surface why.
        raise HTTPException(
            status_code=400,
            detail=" / ".join(warnings) or "تعذّر قراءة المرفق.",
        )

    return await _run_qa(
        question=question,
        k=k,
        prompt_style=prompt_style,
        attached_doc_text=cleaned,
        attached_doc_filename=fname,
        attached_doc_content_type=ctype,
        extra_warnings=warnings,
    )
