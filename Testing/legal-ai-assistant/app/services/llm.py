"""
Multi-provider LLM client. The configured primary (config.LLM_PROVIDER) is tried first,
then the remaining OpenAI-compatible backups (groq → cerebras → xAI), then Gemini
(rotating across multiple project keys), then OpenRouter. Every OpenAI-compatible provider
(Groq, Cerebras, xAI, OpenRouter) shares one chat helper with per-key rotation on
rate-limit/quota; Gemini rotates project keys on daily-quota exhaustion. This layered
chain is what keeps the service answering when any single free tier is exhausted.
"""
import re
import time
import random
import logging
import threading
from typing import AsyncGenerator, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
import config

logger = logging.getLogger(__name__)


def _is_transient_gemini_error(err: Exception) -> bool:
    """True for Gemini failures worth retrying: overloaded (503) or per-minute
    rate-limit spikes (429). A *daily* quota exhaustion (PerDay) is NOT transient —
    it won't recover for hours, so retrying just wastes the remaining budget."""
    s = str(err).lower()
    if "perday" in s or "requestsperdayper" in s:
        return False
    markers = ("429", "503", "resource_exhausted", "unavailable",
               "overloaded", "rate limit", "deadline", "timeout")
    return any(m in s for m in markers)

# ── Singleton clients (sync + async) ──
_client: Optional[OpenAI] = None
_async_client: Optional[AsyncOpenAI] = None
_xai_client: Optional[OpenAI] = None
_xai_async_client: Optional[AsyncOpenAI] = None
_groq_client: Optional[OpenAI] = None
_groq_async_client: Optional[AsyncOpenAI] = None


# User-facing note appended when the configured primary model was unavailable and
# the answer came from a backup provider (quota/availability).
FALLBACK_NOTICE_AR = (
    "ملاحظة: تعذّر الوصول إلى النموذج الأساسي مؤقتًا، وتم توليد الإجابة بنموذج احتياطي."
)


def primary_model() -> str:
    """The model name the system prefers, given the configured provider."""
    if config.LLM_PROVIDER == "xai" and config.XAI_API_KEYS:
        return config.XAI_MODEL
    if config.LLM_PROVIDER == "cerebras" and config.CEREBRAS_API_KEYS:
        return config.CEREBRAS_MODEL
    if config.LLM_PROVIDER == "groq" and config.GROQ_API_KEYS:
        return config.GROQ_MODEL
    return "gemini-2.5-flash"


def _oai_provider_chain(feature: str):
    """Ordered OpenAI-compatible providers — configured-primary first, then backups —
    each as (base_url, keys, model, label). Only providers that actually have a key are
    included. OpenRouter is handled separately so it stays the final fallback."""
    providers = {
        "groq": (config.GROQ_BASE_URL, config.GROQ_API_KEYS,
                 config.MODEL_BY_FEATURE.get(feature, config.GROQ_MODEL), "Groq"),
        "cerebras": (config.CEREBRAS_BASE_URL, config.CEREBRAS_API_KEYS,
                     config.CEREBRAS_MODEL, "Cerebras"),
        "xai": (config.XAI_BASE_URL, config.XAI_API_KEYS, config.XAI_MODEL, "xAI"),
    }
    order = []
    if config.LLM_PROVIDER in providers:
        order.append(config.LLM_PROVIDER)
    for name in ("groq", "cerebras", "xai"):
        if name not in order:
            order.append(name)
    return [providers[n] for n in order if providers[n][1]]

# Sentinels returned by call_llm when every provider in the fallback chain failed.
LLM_ERROR_MODEL = "error"
LLM_ERROR_TEXT = "[ERROR: LLM Service Unavailable]"
# User-facing message when no LLM provider could answer.
LLM_ERROR_NOTICE_AR = (
    "تعذّر توليد إجابة حالياً بسبب عدم توفر نموذج اللغة (انشغال أو تجاوز الحصة). "
    "يرجى المحاولة مرة أخرى بعد قليل."
)


def is_llm_error(model_used: str) -> bool:
    """True if call_llm exhausted every provider and returned the error sentinel."""
    return model_used == LLM_ERROR_MODEL


def used_fallback(model_used: str) -> bool:
    """True if the answer came from a backup provider rather than the configured primary."""
    if model_used == LLM_ERROR_MODEL:
        return False
    pm = primary_model()
    # Any Gemini tier counts as "primary" when Gemini is the primary provider.
    if pm.startswith("gemini") and model_used.startswith("gemini"):
        return False
    return model_used != pm


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


def get_xai_client() -> OpenAI:
    """Get or create the synchronous xAI (Grok) client."""
    global _xai_client
    if _xai_client is None:
        if not config.XAI_API_KEY:
            raise ValueError("XAI_API_KEY not set. Get a key at https://console.x.ai")
        _xai_client = OpenAI(base_url=config.XAI_BASE_URL, api_key=config.XAI_API_KEY)
    return _xai_client


def get_xai_async_client() -> AsyncOpenAI:
    """Get or create the async xAI (Grok) client (used by streaming endpoints)."""
    global _xai_async_client
    if _xai_async_client is None:
        if not config.XAI_API_KEY:
            raise ValueError("XAI_API_KEY not set.")
        _xai_async_client = AsyncOpenAI(base_url=config.XAI_BASE_URL, api_key=config.XAI_API_KEY)
    return _xai_async_client


def get_groq_client() -> OpenAI:
    """Get or create the synchronous Groq client."""
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        _groq_client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)
    return _groq_client


def get_groq_async_client() -> AsyncOpenAI:
    """Get or create the async Groq client (used by streaming endpoints)."""
    global _groq_async_client
    if _groq_async_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set.")
        _groq_async_client = AsyncOpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)
    return _groq_async_client


# Cache OpenAI-compatible clients by (base_url, key) so multi-key rotation reuses connections.
_oai_client_cache: dict = {}
_async_oai_client_cache: dict = {}


def _client_for(base_url: str, key: str) -> OpenAI:
    ck = (base_url, key)
    if ck not in _oai_client_cache:
        _oai_client_cache[ck] = OpenAI(base_url=base_url, api_key=key)
    return _oai_client_cache[ck]


def _async_client_for(base_url: str, key: str) -> AsyncOpenAI:
    ck = (base_url, key)
    if ck not in _async_oai_client_cache:
        _async_oai_client_cache[ck] = AsyncOpenAI(base_url=base_url, api_key=key)
    return _async_oai_client_cache[ck]


# ── In-process rate gate ──
# Serializes and spaces out physical LLM provider requests so the free-tier
# per-minute token/request limits aren't blown by the multi-call-per-request
# pattern (draft → retry → rescue) or by concurrent users. Set LLM_MIN_INTERVAL_S=0
# to disable. The lock makes the spacing correct across worker threads (call_llm
# runs under asyncio.to_thread) and the streaming path gates via to_thread too.
_rate_lock = threading.Lock()
_last_llm_call_ts = 0.0


def _rate_gate() -> None:
    interval = getattr(config, "LLM_MIN_INTERVAL_S", 0.0)
    if interval <= 0:
        return
    global _last_llm_call_ts
    with _rate_lock:
        wait = interval - (time.monotonic() - _last_llm_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_llm_call_ts = time.monotonic()


def _is_rate_limited(err: Exception) -> bool:
    """True for limit/quota errors where rotating to a different key is worth trying."""
    s = str(err).lower()
    return any(m in s for m in (
        "429", "413", "rate limit", "rate_limit", "too large",
        "quota", "tokens per", "insufficient_quota",
    ))


def _try_openai_chat(base_url, keys, model, prompt, system_msg, temp, max_tokens, label, retries=2):
    """Call an OpenAI-compatible endpoint (Groq/xAI/OpenRouter), rotating across `keys`
    on rate-limit/quota errors. Returns text or None."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    for ki, key in enumerate(keys):
        client = _client_for(base_url, key)
        for attempt in range(retries):
            _rate_gate()
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temp, max_tokens=max_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if text:
                    return text
                logger.warning(f"{label} key#{ki+1} returned empty content (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"{label} key#{ki+1}/{len(keys)} attempt {attempt+1} failed: {e}")
                if _is_rate_limited(e):
                    break  # rotate to next key immediately — retrying the same key won't help
                time.sleep(1)
    return None


def _try_gemini(prompt, system_msg, temp, max_tokens):
    """Try Gemini across multiple project keys (rotating on daily-quota exhaustion) and
    model tiers (with transient-error backoff). Each Google Cloud project has its own free
    daily quota, so multiple keys multiply the budget. Returns (text, model) or None."""
    if not config.GOOGLE_API_KEYS:
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        logger.warning(f"google-genai import failed: {e}")
        return None

    full_prompt = f"{system_msg}\n\n{prompt}" if system_msg else prompt
    gen_config = types.GenerateContentConfig(
        temperature=temp,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=config.GEMINI_THINKING_BUDGET),
    )
    n_keys = len(config.GOOGLE_API_KEYS)
    for ki, key in enumerate(config.GOOGLE_API_KEYS):
        client = genai.Client(api_key=key)
        daily_exhausted = False
        for model_id in ['models/gemini-2.5-flash', 'models/gemini-2.0-flash', 'models/gemini-2.0-flash-lite']:
            if daily_exhausted:
                break
            for attempt in range(config.GEMINI_MAX_RETRIES):
                _rate_gate()
                try:
                    response = client.models.generate_content(
                        model=model_id, contents=full_prompt, config=gen_config,
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text, model_id.split("/", 1)[-1]
                    logger.warning(f"{model_id} returned empty text; trying next tier")
                    break
                except Exception as inner_e:
                    s = str(inner_e).lower()
                    if "perday" in s or "requestsperdayper" in s:
                        # This project's daily quota is gone — rotate to the next project key.
                        logger.warning(
                            f"Gemini key#{ki+1}/{n_keys} daily quota exhausted; rotating to next project."
                        )
                        daily_exhausted = True
                        break
                    transient = _is_transient_gemini_error(inner_e)
                    last = attempt == config.GEMINI_MAX_RETRIES - 1
                    if transient and not last:
                        delay = config.GEMINI_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"{model_id} transient failure (attempt {attempt+1}/"
                            f"{config.GEMINI_MAX_RETRIES}): {inner_e}. Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    logger.warning(f"{model_id} failed (attempt {attempt+1}): {inner_e}")
                    break
    return None


def call_llm(
    prompt: str,
    temperature: float = None,
    max_tokens: int = 4096,
    system_msg: str = None,
    feature: str = "default",
) -> Tuple[str, float, str]:
    """
    Call the LLM provider chain ordered by config.LLM_PROVIDER.
    Returns (text, elapsed_seconds, model_used). Falls through providers on failure.
    """
    temp = temperature if temperature is not None else config.TEMPERATURES.get(
        feature, config.TEMPERATURES["default"]
    )

    t0 = time.time()

    # OpenAI-compatible providers: configured primary first, then backups (groq → cerebras
    # → xAI), each rotating across its own key list on rate-limit/quota.
    for base_url, keys, model, label in _oai_provider_chain(feature):
        text = _try_openai_chat(
            base_url, keys, model, prompt, system_msg, temp, max_tokens, label,
        )
        if text:
            return text, time.time() - t0, model

    # Gemini (multi-project key rotation on daily-quota exhaustion).
    gemini_res = _try_gemini(prompt, system_msg, temp, max_tokens)
    if gemini_res:
        text, model = gemini_res
        return text, time.time() - t0, model

    # OpenRouter (last-resort backup).
    if config.OPENROUTER_API_KEYS:
        text = _try_openai_chat(
            config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEYS, config.LLM_MODEL,
            prompt, system_msg, temp, max_tokens, "OpenRouter",
        )
        if text:
            return text, time.time() - t0, config.LLM_MODEL

    return LLM_ERROR_TEXT, time.time() - t0, LLM_ERROR_MODEL


async def async_call_llm(
    prompt: str,
    temperature: float = None,
    max_tokens: int = 4096,
    system_msg: str = None,
    feature: str = "default",
) -> Tuple[str, float, str]:
    """Non-blocking async wrapper around call_llm. Offloads the sync HTTP call via asyncio.to_thread.
    Returns (text, elapsed_seconds, model_used)."""
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
    meta: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """Resilient async generator yielding LLM output chunks.

    Tries the configured primary provider (groq/xai) with multi-key rotation, then
    OpenRouter, all streaming. If every streaming attempt fails *before* producing any
    token, falls back to the full non-streaming chain (`call_llm`: primary → Gemini →
    OpenRouter) and chunks the result out so the SSE contract still holds.

    `meta` (if passed) is populated with:
      - meta['model'] — the provider/model that actually produced the answer
      - meta['error'] = True — only if every provider failed (no content)
    Callers should read meta['model'] for the final SSE event and treat meta['error']
    as a signal to emit a clean user-facing error instead of leaking provider internals.
    """
    import asyncio
    if meta is None:
        meta = {}
    temp = temperature if temperature is not None else config.TEMPERATURES.get(
        feature, config.TEMPERATURES["default"]
    )

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    # Streaming attempts mirror the sync chain (primary → groq/cerebras/xai), then
    # OpenRouter. A `model` override, if given, applies to every attempt.
    attempts = [
        (base_url, keys, (model or chain_model), label)
        for (base_url, keys, chain_model, label) in _oai_provider_chain(feature)
    ]
    if config.OPENROUTER_API_KEYS:
        attempts.append((config.OPENROUTER_BASE_URL, config.OPENROUTER_API_KEYS, model or config.LLM_MODEL, "OpenRouter"))

    for base_url, keys, stream_model, label in attempts:
        for ki, key in enumerate(keys):
            await asyncio.to_thread(_rate_gate)
            client = _async_client_for(base_url, key)
            try:
                stream = await client.chat.completions.create(
                    model=stream_model, messages=messages,
                    temperature=temp, max_tokens=max_tokens, stream=True,
                )
            except Exception as e:
                logger.warning(f"{label} stream key#{ki+1}/{len(keys)} open failed: {e}")
                continue  # rotate to next key / provider — nothing emitted yet
            # Stream opened — commit to this provider.
            meta["model"] = stream_model
            produced = False
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    token = getattr(chunk.choices[0].delta, "content", None)
                    if token:
                        produced = True
                        yield token
                return  # completed cleanly
            except Exception as e:
                logger.warning(f"{label} stream broke mid-generation: {e}")
                if produced:
                    return  # already streamed partial output; can't safely restart
                # else fall through to next key / provider

    # Every streaming attempt failed before producing output → non-streaming full chain.
    logger.warning("All streaming providers failed; falling back to non-streaming chain.")
    text, _, model_used = await asyncio.to_thread(
        call_llm, prompt, temp, max_tokens, system_msg, feature
    )
    meta["model"] = model_used
    if is_llm_error(model_used) or not text:
        meta["error"] = True
        return
    # Chunk the full answer so the client still receives an incremental stream.
    step = 60
    for i in range(0, len(text), step):
        yield text[i:i + step]
