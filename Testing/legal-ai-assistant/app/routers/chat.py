"""
Chat endpoint — Main conversation interface with session management.
Two flavors: /chat (request-response) and /chat/stream (SSE token stream for frontends).
"""
import json
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import config
from app.models import (
    ChatRequest, ChatResponse, SourceInfo, ConfidenceFactors, SessionInfoResponse,
)
from app.services.retrieval import retrieval_service
from app.services.llm import async_call_llm, async_stream_llm, used_fallback, FALLBACK_NOTICE_AR
from app.services.session import session_manager
from app.services.confidence import validate_evidence, topic_match, compute_confidence
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Nour (Session-Aware)",
    description=(
        "Main chat endpoint with conversation history and auto-compaction. "
        "Send a session_id to maintain context across turns. Returns answer with "
        "confidence score, structured sources, and hallucination warnings."
    ),
)
async def chat(req: ChatRequest):
    t0 = time.time()

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

    # 2. Retrieve relevant legal context
    contexts, sources, _ = retrieval_service.retrieve(req.message, k=req.k)
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
    )

    # 5. Record turn
    await session_manager.add_assistant_message(req.session_id, answer)

    # 6. Evidence validation + confidence scoring
    article_pass, missing = validate_evidence(answer, contexts)
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
            "تنبيه: المواد التالية مذكورة في الإجابة لكنها غير موجودة في المصادر: "
            + ", ".join(missing)
        )
        confidence = max(0.0, round(confidence - 0.3, 3))
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
        "Note: streaming uses OpenRouter (Qwen). Non-streaming /chat retains the Gemini-first chain."
    ),
)
async def chat_stream(req: ChatRequest):
    t0 = time.time()

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
    context_str = "\n---\n".join(contexts) if contexts else "لا يوجد سياق قانوني متاح."

    # 3. Build prompt
    prompt = PROMPTS["chat"].format(
        context=context_str, history=history_text, question=req.message,
    )

    async def event_generator():
        full_answer = ""
        try:
            async for token in async_stream_llm(
                prompt, feature="qa", system_msg=SYSTEM_MESSAGES["chat"],
            ):
                full_answer += token
                yield f"data: {json.dumps({'chunk': token, 'done': False}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("chat/stream LLM error")
            yield f"data: {json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"
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
            confidence = max(0.0, round(confidence - 0.3, 3))
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
            "model": config.LLM_MODEL,
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
