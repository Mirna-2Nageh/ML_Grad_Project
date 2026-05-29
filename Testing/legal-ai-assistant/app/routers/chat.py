"""
Chat endpoint — Main conversation interface with session management.
Two flavors: /chat (request-response) and /chat/stream (SSE token stream for frontends).
"""
import json
import time
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

import config
from app.models import (
    ChatRequest, ChatResponse, SourceInfo, ConfidenceFactors, SessionInfoResponse,
    AttachedDocInfo, ChatAttachResponse, ChatAttachmentsListResponse,
)
from app.services.retrieval import retrieval_service
from app.services.llm import (
    async_call_llm, async_stream_llm, used_fallback, is_llm_error,
    FALLBACK_NOTICE_AR, LLM_ERROR_NOTICE_AR,
)
from app.services.session import session_manager
from app.services.confidence import (
    validate_evidence, topic_match, compute_confidence,
    is_meaningful_query, INCOMPLETE_QUERY_MESSAGE_AR,
)
from app.services.postprocessing import postprocess_answer
from app.services.article_lookup import article_lookup_service
from app.services.upload_helper import parse_uploaded_file, parse_uploaded_text
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Conan (Session-Aware)",
    description=(
        "Main chat endpoint with conversation history and auto-compaction. "
        "Send a session_id to maintain context across turns. Returns answer with "
        "confidence score, structured sources, and hallucination warnings."
    ),
)
async def chat(req: ChatRequest):
    t0 = time.time()

    # Resolve session: a missing/blank id gets a fresh UUID (returned to the client) so
    # anonymous callers don't all collide on one shared "default" transcript.
    if not req.session_id:
        req.session_id = uuid.uuid4().hex

    # 0. Input gate — short-circuit fragment/header inputs with a polite ask
    # for a complete question. Still records the turn so chat history is coherent.
    if not is_meaningful_query(req.message):
        session = await session_manager.add_user_message(req.session_id, req.message)
        await session_manager.add_assistant_message(req.session_id, INCOMPLETE_QUERY_MESSAGE_AR)
        return ChatResponse(
            answer=INCOMPLETE_QUERY_MESSAGE_AR,
            session_id=req.session_id,
            confidence_score=0.0,
            confidence_factors=ConfidenceFactors(
                rerank_signal=0.0, source_count=0,
                article_validation="passed", topic_match=False,
            ),
            sources=[],
            warnings=["السؤال غير مكتمل أو غير واضح."],
            conflicts_detected=False,
            latency_ms=round((time.time() - t0) * 1000, 1),
            turn_count=session.turn_count,
            was_compacted=False,
            model="input_gate",
        )

    # 1. Add user message + compact history if needed
    session = await session_manager.add_user_message(req.session_id, req.message)
    was_compacted = False
    if await session_manager.should_compact(req.session_id):
        async def _llm_summarizer(prompt, sys_msg):
            text, _, _ = await async_call_llm(prompt, system_msg=sys_msg, feature="compact")
            return text
        await session_manager.compact(req.session_id, _llm_summarizer)
        was_compacted = True

    history_text = session.format_history()

    # 2. Retrieve relevant legal context — multi-query + domain boost when enabled.
    contexts, sources, _ = retrieval_service.retrieve_multi_query(req.message, k=req.k)

    # 2b. If the session has documents attached, prepend them to the context
    # block so the LLM (and the evidence validator) treat them as authoritative
    # grounding alongside the retrieved corpus. The attachment text is wrapped
    # in the same [المستندات المرفقة] markers used by /qa/upload.
    attachments_block = session.format_attachments()
    if attachments_block:
        contexts = [attachments_block] + contexts

    context_str = "\n---\n".join(contexts) if contexts else "لا يوجد سياق قانوني متاح."

    # 3. Build prompt with context + history
    prompt = PROMPTS["chat"].format(
        context=context_str,
        history=history_text,
        question=req.message,
    )

    # 4. Generate answer
    answer, _, model_used = await async_call_llm(
        prompt, feature="qa", system_msg=SYSTEM_MESSAGES["chat"],
        max_tokens=config.LLM_MAX_TOKENS_QA,
    )

    # If every LLM provider failed, surface a 503 and don't record the error as a turn.
    if is_llm_error(model_used):
        raise HTTPException(status_code=503, detail=LLM_ERROR_NOTICE_AR)

    # 5. Evidence validation.
    article_pass, missing = validate_evidence(answer, contexts)

    # 5a. ITERATIVE RETRIEVAL — widen the retrieval window if validation flagged
    # articles not in the chunks we showed. Mirrors the /qa logic; see qa.py
    # for the rationale.
    iter_expansions: list = []
    if not article_pass and missing and config.USE_ITERATIVE_RETRIEVAL:
        seen_contexts = set(contexts)
        for next_k in [n for n in config.ITERATIVE_K_SEQUENCE if n > req.k]:
            extra_contexts, extra_sources, _ = retrieval_service.retrieve_multi_query(
                req.message, k=next_k,
            )
            new_chunks = [c for c in extra_contexts if c not in seen_contexts]
            if not new_chunks:
                continue
            iter_expansions.append(next_k)
            contexts = contexts + new_chunks
            seen_contexts.update(new_chunks)
            seen_src_keys = {(s.get("source"), s.get("retrieval_score")) for s in sources}
            for s in extra_sources:
                key = (s.get("source"), s.get("retrieval_score"))
                if key not in seen_src_keys:
                    sources.append(s)
                    seen_src_keys.add(key)
            context_str = "\n---\n".join(contexts) if contexts else "لا يوجد سياق قانوني متاح."
            article_pass, missing = validate_evidence(answer, contexts)
            if article_pass:
                break

    # 6. Multi-attempt corrective retry with growing block-list.
    retry_attempts = 0
    blocked = list(missing)
    while (
        not article_pass
        and missing
        and contexts
        and retry_attempts < config.RETRY_MAX_ATTEMPTS
    ):
        retry_attempts += 1
        retry_context_str = context_str[:config.MAX_CONTEXT_CHARS]
        retry_prompt = PROMPTS["qa_retry_ungrounded"].format(
            context=retry_context_str,
            question=req.message,
            draft=answer,
            missing=", ".join(sorted(set(blocked), key=lambda x: int(x) if x.isdigit() else x)),
        )
        retry_answer, _, retry_model = await async_call_llm(
            retry_prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
            max_tokens=config.LLM_MAX_TOKENS_QA,
        )
        if is_llm_error(retry_model):
            break
        new_pass, new_missing = validate_evidence(retry_answer, contexts)
        for m in new_missing:
            if m not in blocked:
                blocked.append(m)
        if new_pass or len(new_missing) < len(missing):
            answer = retry_answer
            article_pass = new_pass
            missing = new_missing
            model_used = retry_model
        if article_pass:
            break

    # 6. Article-lookup RESCUE — same as /qa: if the LLM is still citing
    # articles not in retrieved context, check whether they exist elsewhere
    # in the index and re-prompt with the found passages.
    rescue_attempted = False
    rescue_articles_found: list = []
    if not article_pass and missing and article_lookup_service.is_loaded:
        rescue_chunks, rescue_articles_found = article_lookup_service.lookup_many(
            missing, max_chunks_per_article=2,
        )
        if rescue_chunks:
            rescue_attempted = True
            rescue_context_str = "\n---\n".join(rescue_chunks)[:config.MAX_CONTEXT_CHARS]
            rescue_prompt = PROMPTS["qa_rescue_with_lookup"].format(
                context=context_str[:config.MAX_CONTEXT_CHARS],
                rescue_context=rescue_context_str,
                question=req.message,
                draft=answer,
            )
            rescue_answer, _, rescue_model = await async_call_llm(
                rescue_prompt, feature="qa", system_msg=SYSTEM_MESSAGES["qa"],
                max_tokens=config.LLM_MAX_TOKENS_QA,
            )
            if not is_llm_error(rescue_model):
                combined_contexts = contexts + rescue_chunks
                new_pass, new_missing = validate_evidence(rescue_answer, combined_contexts)
                if new_pass or len(new_missing) < len(missing):
                    answer = rescue_answer
                    article_pass = new_pass
                    missing = new_missing
                    model_used = rescue_model
                    for _ in rescue_chunks:
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

    # 7. Post-process: strip casual openers, rewrite leaky refusals, detect refusal.
    if config.USE_ANSWER_POSTPROCESS:
        answer, is_refusal = postprocess_answer(answer)
    else:
        is_refusal = False

    # 7. Record turn (after possible correction so history stores the grounded answer)
    await session_manager.add_assistant_message(req.session_id, answer)

    # 8. Confidence scoring
    topic_hit = topic_match(req.message, sources)
    rerank_scores = [s.get("rerank_score", 0.0) for s in sources]
    confidence, factors = compute_confidence(
        rerank_scores=rerank_scores,
        source_count=len(sources),
        article_validation_pass=article_pass,
        topic_match_hit=topic_hit,
    )

    warnings = []
    if iter_expansions:
        warnings.append(
            "تم توسيع نطاق البحث تلقائياً إلى "
            f"{iter_expansions[-1]} مرجعاً للعثور على الاستشهادات المطلوبة."
        )
    if not article_pass:
        warnings.append(
            "تنبيه: المواد التالية مذكورة في الإجابة لكنها غير موجودة في المصادر: "
            + ", ".join(missing)
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
        # Cap removed after eval(v5) — see qa.py for the reasoning.
    if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
        warnings.append("ثقة الإجابة منخفضة — هل يمكنك توضيح السؤال أكثر؟")
    if used_fallback(model_used):
        warnings.append(FALLBACK_NOTICE_AR)

    total_ms = (time.time() - t0) * 1000
    return ChatResponse(
        answer=answer,
        session_id=req.session_id,
        confidence_score=confidence,
        confidence_factors=ConfidenceFactors(**factors),
        sources=[SourceInfo(**s) for s in sources],
        warnings=warnings,
        conflicts_detected=False,
        latency_ms=round(total_ms, 1),
        turn_count=session.turn_count,
        was_compacted=was_compacted,
        model=model_used,
    )


@router.post(
    "/chat/stream",
    summary="Streaming Chat (SSE)",
    description=(
        "Streaming version of /chat. Returns Server-Sent Events: one event per token "
        "chunk while the LLM generates, followed by one final event carrying confidence "
        "score, sources, warnings, and timing. Same request body as /chat.\n\n"
        "Event payload shapes (UTF-8 JSON):\n"
        "- During generation:   `data: {\"chunk\": \"...\", \"done\": false}`\n"
        "- Final event:         `data: {\"done\": true, \"session_id\": \"...\", \"confidence_score\": ..., "
        "\"confidence_factors\": {...}, \"sources\": [...], \"warnings\": [...], \"conflicts_detected\": false, "
        "\"turn_count\": ..., \"latency_ms\": ..., \"model\": \"...\"}`\n"
        "- On error:            `data: {\"error\": \"...\", \"done\": true}`\n\n"
        "Streaming tries the configured primary provider (Groq) with multi-key rotation, "
        "then OpenRouter, then falls back to the full non-streaming chain chunked out — "
        "so a rate-limited primary degrades gracefully instead of failing the request."
    ),
)
async def chat_stream(req: ChatRequest):
    t0 = time.time()

    # Resolve session id (fresh UUID when absent) — same as /chat.
    if not req.session_id:
        req.session_id = uuid.uuid4().hex

    # 0. Input gate — emit the polite-ask answer as a single SSE event and stop.
    # Same fragment/header rejection as the non-streaming /chat endpoint.
    if not is_meaningful_query(req.message):
        session = await session_manager.add_user_message(req.session_id, req.message)
        await session_manager.add_assistant_message(req.session_id, INCOMPLETE_QUERY_MESSAGE_AR)

        async def _gated_events():
            yield f"data: {json.dumps({'chunk': INCOMPLETE_QUERY_MESSAGE_AR, 'done': False}, ensure_ascii=False)}\n\n"
            final = {
                "done": True,
                "session_id": req.session_id,
                "confidence_score": 0.0,
                "confidence_factors": {
                    "rerank_signal": 0.0, "source_count": 0,
                    "article_validation": "passed", "topic_match": False,
                },
                "sources": [],
                "warnings": ["السؤال غير مكتمل أو غير واضح."],
                "conflicts_detected": False,
                "turn_count": session.turn_count,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "model": "input_gate",
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            _gated_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 1. Session bookkeeping + compaction (same as /chat)
    session = await session_manager.add_user_message(req.session_id, req.message)
    if await session_manager.should_compact(req.session_id):
        async def _summarizer(prompt, sys_msg):
            text, _, _ = await async_call_llm(prompt, system_msg=sys_msg, feature="compact")
            return text
        await session_manager.compact(req.session_id, _summarizer)

    history_text = session.format_history()

    # 2. Retrieve (synchronous, fast — runs before streaming starts)
    contexts, sources, _ = retrieval_service.retrieve(req.message, k=req.k)
    # Prepend any documents attached to this session so the LLM sees them every turn.
    attachments_block = session.format_attachments()
    if attachments_block:
        contexts = [attachments_block] + contexts
    context_str = "\n---\n".join(contexts) if contexts else "لا يوجد سياق قانوني متاح."

    # 3. Build prompt
    prompt = PROMPTS["chat"].format(
        context=context_str, history=history_text, question=req.message,
    )

    async def event_generator():
        full_answer = ""
        meta: dict = {}
        try:
            async for token in async_stream_llm(
                prompt, feature="qa", system_msg=SYSTEM_MESSAGES["chat"],
                max_tokens=config.LLM_MAX_TOKENS_QA, meta=meta,
            ):
                full_answer += token
                yield f"data: {json.dumps({'chunk': token, 'done': False}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("chat/stream LLM error")

        # If no provider produced any content, emit a clean Arabic error — never leak the
        # raw provider exception/JSON to the client. Don't record a failed turn.
        if not full_answer.strip():
            err_final = {
                "done": True,
                "session_id": req.session_id,
                "error": LLM_ERROR_NOTICE_AR,
                "confidence_score": 0.0,
                "sources": [],
                "warnings": [LLM_ERROR_NOTICE_AR],
                "conflicts_detected": False,
                "turn_count": session.turn_count,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "model": meta.get("model", "error"),
            }
            yield f"data: {json.dumps(err_final, ensure_ascii=False)}\n\n"
            return

        # 4. Post-generation: persist + validate + score + final event
        await session_manager.add_assistant_message(req.session_id, full_answer)

        article_pass, missing = validate_evidence(full_answer, contexts)
        topic_hit = topic_match(req.message, sources)
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
                "تنبيه: المواد التالية مذكورة في الإجابة لكنها غير موجودة في المصادر: " + ", ".join(missing)
            )
            confidence = max(0.0, round(confidence - 0.4, 3))
        if confidence < config.CONFIDENCE_THRESHOLD_CLARIFY:
            warnings.append("ثقة الإجابة منخفضة — هل يمكنك توضيح السؤال أكثر؟")

        # Refresh session for accurate turn_count (we just added an assistant msg above).
        info = await session_manager.get_session_info(req.session_id) or {}
        final_event = {
            "done": True,
            "session_id": req.session_id,
            "confidence_score": confidence,
            "confidence_factors": factors,
            "sources": sources,
            "warnings": warnings,
            "conflicts_detected": False,
            "turn_count": info.get("turn_count", 0),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "model": meta.get("model", ""),
        }
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/chat/{session_id}",
    response_model=SessionInfoResponse,
    summary="Get Session Info",
    description="Get information about a chat session.",
)
async def get_session(session_id: str):
    info = await session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfoResponse(**info)


@router.delete(
    "/chat/{session_id}",
    summary="Delete Session",
    description="Clear a chat session and its history.",
)
async def delete_session(session_id: str):
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ──────────────────────────────────────────────
# Session-attached documents (upload mode #2)
# ──────────────────────────────────────────────
# Use case: the user uploads a contract / case file once, then chats about it
# across multiple turns. Each turn prepends every attached document to the
# retrieved context — see the chat() handler above where session.format_attachments()
# is called. Documents persist with the session JSON on disk so they survive
# restarts. Validation accepts article numbers from attached docs too (they
# become part of the combined context).

@router.post(
    "/chat/attach",
    response_model=ChatAttachResponse,
    summary="Attach a document to a chat session",
    description=(
        "Bind a document to a session so subsequent /chat turns include it in "
        "the LLM context. Accepts .txt, .pdf, .docx files OR pasted text. "
        "Multiple documents can be attached to the same session; each turn "
        "includes ALL attachments.\n\n"
        "Send as `multipart/form-data` with fields: `session_id` (required), "
        "`file` (one of .txt/.pdf/.docx, optional), `text` (raw string, optional). "
        "Provide either `file` OR `text`; if both, `file` wins."
    ),
)
async def chat_attach(
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
):
    # Parse the upload (file wins over text if both given).
    if file is not None and file.filename:
        cleaned, fname, ctype, warnings = await parse_uploaded_file(file)
    elif text:
        cleaned, fname, ctype, warnings = parse_uploaded_text(text)
    else:
        raise HTTPException(
            status_code=400, detail="Provide either a `file` or a `text` field.",
        )

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=" / ".join(warnings) or "تعذّر قراءة المرفق.",
        )

    doc = await session_manager.add_attachment(
        session_id=session_id, filename=fname, content_type=ctype, text=cleaned,
    )

    # Recompute totals for the response
    all_attachments = await session_manager.list_attachments(session_id)
    total_chars = sum(len(a.text) for a in all_attachments)

    return ChatAttachResponse(
        session_id=session_id,
        attached=AttachedDocInfo(**doc.info_dict()),
        attachments_total=len(all_attachments),
        total_chars=total_chars,
        warnings=warnings,
    )


@router.get(
    "/chat/{session_id}/attachments",
    response_model=ChatAttachmentsListResponse,
    summary="List documents attached to a chat session",
)
async def chat_attachments_list(session_id: str):
    docs = await session_manager.list_attachments(session_id)
    return ChatAttachmentsListResponse(
        session_id=session_id,
        attachments=[AttachedDocInfo(**d.info_dict()) for d in docs],
        total_chars=sum(len(d.text) for d in docs),
    )


@router.delete(
    "/chat/{session_id}/attachments/{doc_id}",
    summary="Remove one attached document by id",
)
async def chat_attachment_remove(session_id: str, doc_id: str):
    removed = await session_manager.remove_attachment(session_id, doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return {"status": "removed", "session_id": session_id, "doc_id": doc_id}


@router.delete(
    "/chat/{session_id}/attachments",
    summary="Detach all documents from a chat session",
)
async def chat_attachments_clear(session_id: str):
    n = await session_manager.clear_attachments(session_id)
    return {"status": "cleared", "session_id": session_id, "removed_count": n}
