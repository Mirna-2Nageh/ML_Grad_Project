"""
OpenRouter LLM client (OpenAI-compatible). Uses Qwen via OpenRouter by default,
with Google Gemini as the primary path when GOOGLE_API_KEY is set.
"""
import time
import logging
from typing import AsyncGenerator, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
import config

logger = logging.getLogger(__name__)

# ── Singleton clients (sync + async) ──
_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None


def get_client() -> OpenAI:
    """Get or create the synchronous OpenRouter client."""
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Get a free key at https://openrouter.ai/keys"
            )
        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
    return _client


def get_async_client() -> AsyncOpenAI:
    """Get or create the async OpenRouter client (used by streaming endpoints)."""
    global _async_client
    if _async_client is None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set.")
        _async_client = AsyncOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
    return _async_client


def call_llm(
    prompt: str,
    temperature: float = None,
    max_tokens: int = 2048,
    system_msg: str = None,
    feature: str = "default",
) -> Tuple[str, float]:
    """
    Call LLM via Gemini (Primary) or OpenRouter (Fallback).
    """
    temp = temperature if temperature is not None else config.TEMPERATURES.get(
        feature, config.TEMPERATURES["default"]
    )

    t0 = time.time()
    
    # --- Try Google Gemini first (using modern native API) ---
    if config.GOOGLE_API_KEY:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=config.GOOGLE_API_KEY)
            
            # Combine system msg and prompt for Gemini
            full_prompt = f"{system_msg}\n\n{prompt}" if system_msg else prompt
            
            # Try a loop of model names
            for model_id in ['models/gemini-2.5-flash', 'models/gemini-2.0-flash', 'models/gemini-2.0-flash-lite']:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=temp,
                            max_output_tokens=max_tokens,
                        )
                    )
                    text = response.text.strip()
                    if text:
                        return text, time.time() - t0
                except Exception as inner_e:
                    logger.warning(f"Attempt with {model_id} failed: {inner_e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Native Gemini call failed: {e}. Falling back to OpenRouter...")

    # --- Fallback to OpenRouter ---
    client = get_client()
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text, time.time() - t0
        except Exception as e:
            logger.warning(f"OpenRouter attempt {attempt+1} failed: {e}")
            time.sleep(1)

    return "[ERROR: LLM Service Unavailable]", time.time() - t0


async def async_call_llm(
    prompt: str,
    temperature: float = None,
    max_tokens: int = 2048,
    system_msg: str = None,
    feature: str = "default",
) -> tuple:
    """Non-blocking async wrapper around call_llm. Offloads the sync HTTP call via asyncio.to_thread."""
    import asyncio
    return await asyncio.to_thread(
        call_llm, prompt, temperature, max_tokens, system_msg, feature
    )


async def async_stream_llm(
    prompt: str,
    temperature: float = None,
    max_tokens: int = 2048,
    system_msg: str = None,
    feature: str = "default",
    model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding LLM output chunks. OpenRouter-only (no Gemini fallback yet).

    Gemini's native streaming API differs enough that we skip it here — the streaming endpoint
    is OpenRouter-Qwen-only. Non-streaming endpoints retain the Gemini-first fallback chain.
    """
    temp = temperature if temperature is not None else config.TEMPERATURES.get(
        feature, config.TEMPERATURES["default"]
    )
    client = get_async_client()
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    stream = await client.chat.completions.create(
        model=model or config.LLM_MODEL,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", None)
        if token:
            yield token
