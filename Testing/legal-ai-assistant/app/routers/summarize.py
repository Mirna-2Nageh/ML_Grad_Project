"""Legal text summarization endpoint."""
import time
from fastapi import APIRouter
import config
from app.models import SummarizeRequest, SummarizeResponse
from app.services.llm import async_call_llm
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

router = APIRouter()


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Legal Text Summarization",
    description="Summarize Egyptian legal text, preserving article numbers and penalties.",
)
async def summarize_text(req: SummarizeRequest):
    t0 = time.time()

    prompt = PROMPTS["summarize"].format(text=req.text[:config.MAX_INPUT_CHARS])
    summary, _, model_used = await async_call_llm(
        prompt,
        feature="summarize",
        system_msg=SYSTEM_MESSAGES["summarize"],
        max_tokens=2048,
    )

    return SummarizeResponse(
        summary=summary,
        input_length=len(req.text),
        latency_ms=round((time.time() - t0) * 1000, 1),
        model=model_used,
    )
