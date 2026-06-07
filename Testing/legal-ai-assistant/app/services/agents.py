"""Agentic pipeline for /weakness and /defense.

Replaces the single-shot "case → LLM → output" path with a multi-agent flow:

  Document Analysis ─▶ Weakness Detection ─▶ (Legal Research ‖ Precedent Retrieval)
                    ─▶ Defense Strategy ─▶ Memorandum / Weakness-Analysis synthesis

Each agent is a discrete function. The intermediate agents emit strict JSON; the
research/precedent agents query the existing RetrievalService (hybrid + rerank over
the 47k-chunk legal KB) once PER weakness, so every argument is backed by retrieved
authority instead of the model's memory. The two orchestrators —
`run_weakness_pipeline` / `run_defense_pipeline` — return both the final Arabic text
and the structured intermediates (timeline, weaknesses, authorities, sources).

Latency note: retrieval is CPU-bound (~tens of seconds per rerank). Per-weakness
retrievals are launched concurrently with asyncio.gather; actual CPU concurrency is
bounded by the reranker semaphore (config.RERANK_MAX_CONCURRENCY). AGENT_MAX_WEAKNESSES
caps how many weaknesses get deep research so a memo stays within the client timeout.

If any critical agent fails, the orchestrator raises AgentPipelineError and the router
falls back to the single-shot path — the pipeline is never a hard dependency.
"""
import asyncio
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import config
from app.services.retrieval import retrieval_service
from app.services.llm import async_call_llm, is_llm_error
from app.services.preprocessing import normalize_arabic_indic_digits
from app.core.prompts import PROMPTS, SYSTEM_MESSAGES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Deterministic timing verdict (NOT left to the LLM)
# ──────────────────────────────────────────────
# The single most damaging recurring error is the model mis-reading whether the
# arrest/search happened before or after the prosecution warrant — it keeps inverting
# the chronology even when the timeline is extracted correctly. So we compute the
# verdict in CODE from the structured timeline and hand the agents the conclusion as a
# hard fact they cannot re-derive incorrectly.

def _parse_event_dt(ev: Dict) -> Optional[datetime]:
    """Combine an event's date + Arabic time string into a datetime, or None."""
    d = normalize_arabic_indic_digits(str(ev.get("date") or "")).strip()
    t = normalize_arabic_indic_digits(str(ev.get("time") or "")).strip()
    day: Optional[date] = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            day = datetime.strptime(d, fmt).date()
            break
        except ValueError:
            continue
    if day is None:
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", d) or re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", d)
        if m:
            g = list(map(int, m.groups()))
            day = date(g[0], g[1], g[2]) if len(d) and d[0:4].isdigit() else date(g[2], g[1], g[0])
        else:
            return None
    # time-of-day
    hour, minute = 0, 0
    if "منتصف" in t and "ليل" in t:
        hour, minute = 0, 0
    else:
        hm = re.search(r"(\d{1,2})(?::(\d{2}))?", t)
        if hm:
            hour = int(hm.group(1)) % 24
            minute = int(hm.group(2) or 0)
            is_pm = ("مساء" in t) or ("ظهر" in t) or ("ليل" in t)
            is_am = ("صباح" in t) or ("فجر" in t)
            if is_pm and hour < 12:
                hour += 12
            if is_am and hour == 12:
                hour = 0
    return datetime(day.year, day.month, day.day, hour, minute)


def compute_timing_verdict(timeline: Any) -> Optional[Dict]:
    """From the timeline, deterministically decide whether the seizure preceded the
    warrant, followed it within its validity window, or fell after it expired.
    Returns {flag, gap_hours, verdict (Arabic)} or None if it can't be computed."""
    if not isinstance(timeline, list):
        return None
    warrant_dt = arrest_dt = None
    for ev in timeline:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("event") or "")
        dt = _parse_event_dt(ev)
        if dt is None:
            continue
        if ("إذن" in name or "اذن" in name) and warrant_dt is None:
            warrant_dt = dt
        elif ("ضبط" in name or "تفتيش" in name or "قبض" in name) and arrest_dt is None:
            arrest_dt = dt
    if warrant_dt is None or arrest_dt is None:
        return None
    gap_h = (arrest_dt - warrant_dt).total_seconds() / 3600.0
    w = warrant_dt.strftime("%Y-%m-%d %H:%M")
    a = arrest_dt.strftime("%Y-%m-%d %H:%M")
    if gap_h < 0:
        flag = "before"
        verdict = (f"حُسِب آلياً: الضبط ({a}) وقع قبل صدور إذن النيابة ({w}) بفارق "
                   f"{abs(gap_h):.1f} ساعة. الدفع ببطلان القبض والتفتيش لسبقهما على الإذن صحيح، "
                   f"ويُتصدَّر به الدفاع، مع إعمال قاعدة (ما بُني على باطل فهو باطل).")
    elif gap_h <= 24:
        flag = "after_within"
        verdict = (f"حُسِب آلياً: الضبط ({a}) وقع بعد صدور إذن النيابة ({w}) بفارق "
                   f"{gap_h:.1f} ساعة، أي داخل مدة سريان الإذن (24 ساعة). توقيت الضبط سليم — "
                   f"يُحظر طرح دفع 'القبض سابق على الإذن' لأنه مخالف للأوراق.")
    else:
        flag = "after_expired"
        verdict = (f"حُسِب آلياً: الضبط ({a}) وقع بعد انقضاء مدة سريان الإذن ({w} + 24 ساعة). "
                   f"الدفع المتاح هو بطلان الضبط لتجاوز مدة الإذن — وليس لسبقه على الإذن.")
    return {"flag": flag, "gap_hours": round(gap_h, 1), "verdict": verdict}


class AgentPipelineError(Exception):
    """Raised when a critical agent fails — signals the router to fall back to single-shot."""


# ──────────────────────────────────────────────
# JSON extraction from LLM output
# ──────────────────────────────────────────────
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON object/array from an LLM response.

    Handles ```json fences and leading/trailing prose by extracting the first
    balanced {...} (or [...]) span. Returns the parsed value or None.
    """
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced brace/bracket span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            c = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


async def _agent_json(prompt: str, system_key: str, max_tokens: int, label: str) -> Any:
    """Run one JSON-emitting agent LLM call and parse the result, or raise AgentPipelineError."""
    raw, _, model = await async_call_llm(
        prompt, feature="weakness", system_msg=SYSTEM_MESSAGES[system_key], max_tokens=max_tokens,
    )
    if is_llm_error(model):
        raise AgentPipelineError(f"{label}: LLM provider chain failed")
    data = _extract_json(raw)
    if data is None:
        logger.warning(f"{label}: could not parse JSON from agent output (first 200 chars): {raw[:200]!r}")
        raise AgentPipelineError(f"{label}: non-JSON agent output")
    return data


# ──────────────────────────────────────────────
# Agent 1 — Document Analysis
# ──────────────────────────────────────────────
async def analyze_document(case_facts: str, evidence: str, defendant_statement: str) -> Dict:
    """Extract structured facts, charges, evidence, procedural steps, and an explicit
    chronological timeline (date+time per event) from the case file."""
    prompt = PROMPTS["doc_analysis"].format(
        case_facts=case_facts,
        evidence=evidence or "لم تُقدَّم أدلة إضافية بخلاف ما ورد في الوقائع.",
        defendant_statement=defendant_statement or "لم يُقدَّم بيان منفصل للمتهم.",
    )
    data = await _agent_json(prompt, "doc_analysis", config.AGENT_ANALYSIS_MAX_TOKENS, "DocumentAnalysis")
    if not isinstance(data, dict):
        raise AgentPipelineError("DocumentAnalysis: expected a JSON object")
    return data


# ──────────────────────────────────────────────
# Agent 2 — Weakness Detection
# ──────────────────────────────────────────────
async def detect_weaknesses(analysis: Dict) -> List[Dict]:
    """Turn the structured analysis into a list of typed weakness hypotheses, each with
    a focused legal-research query."""
    prompt = PROMPTS["weakness_hypotheses"].format(
        analysis=json.dumps(analysis, ensure_ascii=False),
    )
    data = await _agent_json(prompt, "weakness_hypotheses", config.AGENT_DETECT_MAX_TOKENS, "WeaknessDetection")
    weaknesses = data.get("weaknesses") if isinstance(data, dict) else None
    if not isinstance(weaknesses, list) or not weaknesses:
        raise AgentPipelineError("WeaknessDetection: no weaknesses produced")
    # Ensure each has an id (downstream keys off it).
    for i, w in enumerate(weaknesses):
        w.setdefault("id", f"w{i + 1}")
    return weaknesses


# ──────────────────────────────────────────────
# Agents 3 & 4 — Legal Research + Precedent Retrieval (per weakness)
# ──────────────────────────────────────────────
def _query_for(w: Dict) -> str:
    return (w.get("research_query") or w.get("legal_question") or w.get("claim")
            or w.get("title") or "").strip()


async def _retrieve(query: str, k: int) -> Tuple[List[str], List[Dict]]:
    """Offload the (CPU-bound, rerank) retrieval to a thread; bounded by the rerank semaphore."""
    contexts, sources, _ = await asyncio.to_thread(retrieval_service.retrieve, query, k=k)
    return contexts, sources


def _is_precedent(s: Dict) -> bool:
    """A retrieved source is a court precedent if its doc_type matches a precedent marker
    (the corpus labels cassation material e.g. 'cassation_encyclopedia')."""
    dt = (s.get("doc_type") or "").lower()
    return any(t.lower() in dt for t in config.AGENT_PRECEDENT_DOC_TYPES)


async def research_weakness(w: Dict) -> Dict:
    """Legal Research + Precedent Retrieval for one weakness in a SINGLE retrieval: the pool
    yields both doctrine/statute and court-precedent chunks, which are split by doc_type.
    One retrieval per weakness (instead of two) keeps a full memo CPU-affordable."""
    query = _query_for(w)
    if not query:
        return {"weakness": w, "contexts": [], "sources": [], "precedent_sources": []}

    contexts, sources = await _retrieve(query, config.AGENT_RESEARCH_K)
    precedent_sources = [s for s in sources if _is_precedent(s)]
    return {
        "weakness": w,
        "contexts": contexts,
        "sources": sources,
        "precedent_sources": precedent_sources,
    }


# ──────────────────────────────────────────────
# Agent 5 — Defense Strategy
# ──────────────────────────────────────────────
def _researched_for_strategy(researched: List[Dict]) -> List[Dict]:
    """Compact view of each researched weakness for the strategy agent (drops full text,
    keeps the weakness + the article numbers / filenames that were actually retrieved)."""
    out = []
    for r in researched:
        w = r["weakness"]
        arts, refs = [], []
        for s in r.get("sources", []) + r.get("precedent_sources", []):
            if s.get("article"):
                arts.append(str(s["article"]))
            refs.extend(str(a) for a in (s.get("referenced_articles") or []))
        authorities = sorted(set(arts + refs), key=lambda x: (len(x), x))
        # A short authority excerpt list so the strategist can judge support.
        excerpts = [c[:600] for c in r.get("contexts", [])[:4]]
        out.append({
            "id": w.get("id"),
            "type": w.get("type"),
            "title": w.get("title"),
            "claim": w.get("claim"),
            "supporting_facts": w.get("supporting_facts"),
            "retrieved_articles": authorities,
            "authority_excerpts": excerpts,
        })
    return out


async def build_strategy(analysis: Dict, researched: List[Dict]) -> Dict:
    """Rank weaknesses by real strength against the retrieved authorities, keep the strong
    ones, discard the weak/disproven ones."""
    prompt = PROMPTS["defense_strategy"].format(
        analysis=json.dumps(analysis, ensure_ascii=False),
        researched=json.dumps(_researched_for_strategy(researched), ensure_ascii=False),
    )
    data = await _agent_json(prompt, "defense_strategy", config.AGENT_STRATEGY_MAX_TOKENS, "DefenseStrategy")
    if not isinstance(data, dict) or not isinstance(data.get("strategy"), list):
        raise AgentPipelineError("DefenseStrategy: malformed strategy")
    return data


# ──────────────────────────────────────────────
# Agent 6 — Synthesis (Memorandum / Weakness analysis)
# ──────────────────────────────────────────────
def _format_authorities(researched: List[Dict], strategy: Dict) -> str:
    """Build the grounded-authorities text block fed to the synthesis agent, restricted to
    the weaknesses the strategy KEPT, capped at MAX_CONTEXT_CHARS."""
    kept_ids = {s.get("weakness_id") for s in strategy.get("strategy", []) if s.get("keep", True)}
    by_id = {r["weakness"].get("id"): r for r in researched}
    blocks: List[str] = []
    for wid in (strategy.get("ordered_argument_ids") or list(kept_ids)):
        r = by_id.get(wid)
        if not r:
            continue
        title = r["weakness"].get("title") or r["weakness"].get("claim") or wid
        parts = [f"### {title}"]
        prec_files = {(s.get("filename") or s.get("source")) for s in r.get("precedent_sources", [])}
        parts.extend(r.get("contexts", [])[:4])
        if prec_files:
            parts.append("[سوابق قضائية ذات صلة]: " + "، ".join(sorted(f for f in prec_files if f)))
        blocks.append("\n".join(parts))
    text = "\n\n---\n\n".join(blocks)
    return text[:config.MAX_CONTEXT_CHARS]


async def synthesize(kind: str, analysis: Dict, strategy: Dict, authorities: str) -> Tuple[str, str]:
    """Final synthesis: kind='weakness' → weakness analysis; kind='defense' → memorandum.
    Returns (text, model_used)."""
    prompt_key = "weakness_agentic" if kind == "weakness" else "defense_agentic"
    system_key = "weakness" if kind == "weakness" else "defense"
    prompt = PROMPTS[prompt_key].format(
        analysis=json.dumps(analysis, ensure_ascii=False),
        strategy=json.dumps(strategy, ensure_ascii=False),
        authorities=authorities or "لا توجد نصوص قانونية مسترجعة.",
    )
    text, _, model = await async_call_llm(
        prompt, feature=kind, system_msg=SYSTEM_MESSAGES[system_key],
        max_tokens=config.AGENT_SYNTH_MAX_TOKENS,
    )
    if is_llm_error(model):
        raise AgentPipelineError("Synthesis: LLM provider chain failed")
    return text, model


# ──────────────────────────────────────────────
# Source aggregation (for the wire contract)
# ──────────────────────────────────────────────
def _aggregate_sources(researched: List[Dict]) -> List[Dict]:
    """Dedup the per-weakness retrieved sources into one list for the response, by
    (filename, article)."""
    seen = set()
    out: List[Dict] = []
    for r in researched:
        for s in r.get("sources", []) + r.get("precedent_sources", []):
            key = (s.get("filename") or s.get("source"), s.get("article"))
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def _all_contexts(researched: List[Dict]) -> List[str]:
    ctx: List[str] = []
    for r in researched:
        ctx.extend(r.get("contexts", []))
    return ctx


# ──────────────────────────────────────────────
# Orchestrators
# ──────────────────────────────────────────────
async def _run_pipeline(kind: str, case_facts: str, evidence: str, defendant_statement: str) -> Dict:
    t0 = time.time()
    analysis = await analyze_document(case_facts, evidence, defendant_statement)
    # Deterministic timing verdict computed in CODE from the extracted timeline, injected
    # as a hard fact so the downstream agents cannot re-derive the arrest/warrant order
    # incorrectly (the recurring "القبض سابق على الإذن" inversion).
    timing = compute_timing_verdict(analysis.get("timeline"))
    if timing:
        analysis["timing_verdict"] = timing["verdict"]
        analysis["timing_flag"] = timing["flag"]
        logger.info(f"Timing verdict: {timing['flag']} (gap {timing['gap_hours']}h)")
    weaknesses = await detect_weaknesses(analysis)
    top = weaknesses[:config.AGENT_MAX_WEAKNESSES]

    # Agents 3+4: research every top weakness concurrently (CPU concurrency capped by the
    # rerank semaphore). A single weakness failing to retrieve doesn't sink the pipeline.
    researched = await asyncio.gather(*[research_weakness(w) for w in top], return_exceptions=True)
    researched = [r for r in researched if isinstance(r, dict)]
    if not researched:
        raise AgentPipelineError("Research: no weakness could be researched")

    strategy = await build_strategy(analysis, researched)
    authorities = _format_authorities(researched, strategy)
    final_text, model_used = await synthesize(kind, analysis, strategy, authorities)

    return {
        "text": final_text,
        "model": model_used,
        "analysis": analysis,
        "weaknesses": weaknesses,
        "strategy": strategy,
        "sources": _aggregate_sources(researched),
        "contexts": _all_contexts(researched),
        "researched_count": len(researched),
        "pipeline_ms": round((time.time() - t0) * 1000, 1),
    }


async def run_weakness_pipeline(case_facts: str, evidence: str, defendant_statement: str) -> Dict:
    """Full agentic weakness analysis. Returns dict with `text`, `sources`, `contexts`,
    `analysis`, `weaknesses`, `strategy`, `model`. Raises AgentPipelineError on failure."""
    return await _run_pipeline("weakness", case_facts, evidence, defendant_statement)


async def run_defense_pipeline(case_facts: str, evidence: str, defendant_statement: str) -> Dict:
    """Full agentic defense memorandum. Same shape as run_weakness_pipeline."""
    return await _run_pipeline("defense", case_facts, evidence, defendant_statement)
