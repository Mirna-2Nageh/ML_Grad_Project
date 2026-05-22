# Nour — Egyptian Criminal-Law AI Assistant: Production & Research Plan

A single reference for architecture, operations, evaluation, risks, and the research
paper. Pairs with `CLAUDE.md` (run-book) and `api_contract.md` (wire contract).

---

## 1. What this system is

A **hybrid-retrieval RAG** service for Egyptian Criminal Law, Arabic-first. It answers
legal questions, summarizes texts, analyzes prosecution weaknesses, and drafts defense
memoranda — every claim grounded in a retrieved corpus, with post-hoc citation
validation and confidence scoring.

**Architecture choice (and recommendation): Traditional Hybrid RAG + an expert-rules
knowledge layer, evolving selectively toward Agentic RAG for the memo workflow.**

- **Not** full Graph RAG — retrieval already achieves 13/13 article recall on the eval
  set; entity/relation extraction over Arabic legal text isn't worth the build cost yet.
- **Not** a free-roaming agent — but the defense-memo flow benefits from a *structured*
  agentic pipeline (classify charge → retrieve per charge → draft → self-verify citations
  → revise). Recommended as the next architectural step for memos only.

---

## 2. Request flow & RAG pipeline

```
Client (.NET / Streamlit)
   │  HTTPS JSON
   ▼
FastAPI backend  (app/main.py → routers/)
   │
   ├─ RetrievalService.retrieve(query, k)
   │     FAISS dense (top 30) + BM25 sparse (top 30)
   │       → Reciprocal Rank Fusion (RRF_K=60)
   │       → cross-encoder rerank (bge-reranker-v2-m3) → top k
   │       → same-source neighbor expansion
   │       → prepend matched expert-rule contexts
   │
   ├─ Prompt assembly (app/core/prompts.py, per feature)
   │
   ├─ LLM call (app/services/llm.py)
   │     provider chain: [primary] → Gemini → OpenRouter
   │     multi-key rotation on rate-limit; per-feature model routing
   │
   ├─ Post-generation evidence validation (regex article extraction vs context)
   └─ Confidence scoring (rerank, source_count, article_validation, topic_match)
   ▼
Structured JSON response (app/models.py — stable wire contract)
```

**Per-feature model routing** (`config.MODEL_BY_FEATURE`):
- `defense`, `weakness` → `GROQ_MODEL_STRONG` (70B) — heavy reasoning.
- `qa`, `chat`, `summarize` → `GROQ_MODEL` (can be cheaper/faster, e.g. 8B).

---

## 3. LLM layer: providers, models, multi-key rotation

**Important:** the system uses the **OpenAI-compatible SDK**, not OpenAI the company.
Today's working provider is **Groq (free tier)** — no billing, only rate limits.

**Provider chain** (`LLM_PROVIDER` = `groq` | `xai` | `gemini`): primary tried first,
then Gemini, then OpenRouter. All OpenAI-compatible providers share one chat helper.

**Multi-key rotation** (`config._key_list`): set `GROQ_API_KEYS=k1,k2,k3` (comma-separated).
On a rate-limit/quota error (`429/413/quota/tokens per`), the router rotates to the next
key before falling through to other providers — multiplying the free daily token budget.
*Caveat:* rotating multiple free accounts to bypass per-account limits may violate the
provider's ToS; clean only when keys are legitimately the team's or paid.

**Model recommendation for Arabic legal:**

| Model | Quality | Hallucination | Free budget | Use for |
|---|---|---|---|---|
| llama-3.3-70b-versatile | Good | Low (grounded) | ~100k tok/day | defense, weakness, hard QA |
| llama-3.1-8b-instant | Weak | Higher | High | high-volume simple QA only |
| qwen3-32b | Good Arabic | Low | High | A/B candidate (emits `<think>`) |
| gpt-oss-120b | Strong | Low | Medium | A/B candidate |
| Claude / GPT-4 (paid) | Best | Lowest | Paid | if budget available |

**Token economics (Groq free tier):** the binding limit is *tokens*, not requests.
70B ≈ 100k tokens/day (~29 RAG queries after trims). Per-request cost was cut ~55% via
`MAX_CONTEXT_CHARS=4500` + `LLM_MAX_TOKENS_QA=1024`.

---

## 4. Failover & resilience (implemented)

- **Per-tier retry/backoff** on transient Gemini 503/429; skip retries on daily-quota.
- **Multi-key rotation** within a provider on rate-limit.
- **Cross-provider fallback**: primary → Gemini → OpenRouter.
- **Hard failure → HTTP 503** with an Arabic "try again" message (never a fake-confident
  error answer). Chat does not record a failed turn.

**Still to add for production scale:** a per-key **request queue + rate-limiter** in the
backend so concurrent users don't collide on the shared 30 req/min limit.

---

## 5. Risk register (shared/group keys & ops)

| Risk | Severity (current/paid) | Mitigation |
|---|---|---|
| Key leak | Med | `.env` gitignored; server env vars / secret manager in prod; never commit |
| Key abuse | Med | one key per service; rotate on leak |
| Token exhaustion | High (free) | multi-key rotation + per-feature routing + caching |
| Unexpected billing | **None (Groq free)** / High (paid) | hard spend caps if paid |
| Key revocation | Med | multi-key + cross-provider fallback → one dead key ≠ outage |
| Concurrency collision | Med | **TODO**: backend request queue + per-key limiter |
| Rate-limit failure | High | mitigated: backoff + rotation + fallback + 503 |
| Hallucinated law (memos) | Med-High | grounding prompts + article validation; **TODO**: argument-level self-check |

---

## 6. Prompt design

All prompts live in `app/core/prompts.py`, written in English, output strict MSA Arabic.
Principles enforced: **use provided texts only**, cite article numbers, **no disclaimers /
no fabrication**, ambiguity protocol asks **crime-relevant** clarifiers, defense/weakness
prompts separate fact vs argument vs request and handle edge cases (thin facts, clear
guilt, multiple charges/defendants).

**Open work:** stress-test prompts under adversarial inputs (prompt injection in case
facts, mixed-language input, out-of-domain questions).

---

## 7. Legal-document workflow (defense memo) — quality assessment

- **Strengths:** structured (الوقائع/الإطار القانوني/أوجه الدفاع/الطلبات), grounded,
  article citations validated.
- **Weaknesses:** synthesis across many articles needs the 70B+; case-law citations not
  validated; degrades on multi-charge/multi-defendant and very long facts.
- **Hallucination risk:** moderate — validator catches fabricated *article numbers* but
  not fabricated *legal arguments*. Biggest open gap.
- **Recommended hardening:** agentic self-check pass (draft → verify each article &
  argument against context → revise) before returning the memo.

---

## 8. Evaluation methodology

1. **Retrieval** (LLM-free, `scripts/eval_retrieval.py`): Recall@k / MRR on labeled
   query→article set. Currently 13/13; expand to 50–100 queries.
2. **Answer quality** (`scripts/eval_battery.py` + extensions):
   - Faithfulness / citation grounding (% cited articles in context).
   - Hallucination rate (held-out set, manual or LLM-judge).
   - Correctness vs an expert-graded gold set.
3. **LLM-as-judge**: strong model scores answers on a rubric (correctness, grounding,
   completeness, Arabic quality).
4. **Comparative benchmark** (`scripts/benchmark_llms.py`): our RAG vs raw LLMs.
5. **Human expert evaluation**: small gold set graded by a lawyer/professor — the paper's
   credibility anchor.

Formalize into an `evaluation/` module with versioned gold sets.

---

## 9. Testing strategy

- **Retrieval regression**: `eval_retrieval.py` (CI-runnable, no LLM).
- **Answer regression**: `eval_battery.py` (needs LLM budget; run incrementally).
- **Unit tests** (TODO): preprocessing (article extraction, digit normalization),
  confidence scoring, expert-rule matching, provider-fallback logic (mock the SDK).
- **Contract tests** (TODO): response schemas vs `api_contract.md`.
- **Load/concurrency tests** (TODO) before production.

---

## 10. Deployment & infrastructure

**Current = development**: everything on a laptop. For production the laptop must NOT be
the server.

```
.NET frontend ──HTTPS──> FastAPI backend (always-on cloud VM / container)
                              ├─ local: FAISS + BM25 + embeddings + reranker (in RAM)
                              └─ outbound: Groq/LLM API
```

- **VM sizing:** ≥4–8 GB RAM (index + models), CPU-only OK (~15s retrieval latency).
- **Path:** Dockerize backend → run on always-on VM behind nginx + HTTPS → `/health`
  monitoring. Sessions persist to disk (restart-safe).
- **Scaling:** stateless API replicas behind a load balancer; shared index volume or
  per-replica copy; centralized rate-limiter for the LLM key pool.

---

## 11. Monitoring (TODO)

Log per request: latency (retrieval vs LLM), model used, token usage, confidence,
warnings, provider/key used, error type. Alert on: 503 rate, p95 latency, daily-token
budget nearing exhaustion, confidence-distribution drift.

---

## 12. Roadmap (phased)

- **A — Stabilize (DONE):** multi-key rotation, per-feature model routing, retry/backoff,
  cross-provider fallback, 503-on-failure, token trims.
- **B — Harden:** request queue + rate-limiter; agentic memo self-check; expand expert
  rules; prompt adversarial-hardening; input validation.
- **C — Evaluate:** `evaluation/` module, expert gold set, full benchmark run.
- **D — Deploy:** Dockerize, cloud VM, nginx+HTTPS, monitoring.
- **E — Paper:** write up using real eval results.

---

## 13. Research paper direction

Structure: Abstract → Intro (Arabic legal-NLP gap) → Related Work (team refs) → System
Architecture (hybrid RAG + expert rules + confidence/validation + multi-provider failover)
→ Methodology → **Evaluation** (Section 8 metrics) → Results → **Limitations** (free-tier
budget, retrieval ceiling outside expert-rule topics, argument-level hallucination in
memos, no expert gold set yet) → Future Work (GraphRAG, agentic memo, expert eval) →
Conclusion.

**Inputs needed from the team:** reference papers, member names/roles, target venue.

---

## 14. Known limitations (be honest in the paper)

- Free-tier LLM token budget caps throughput (~29 70B queries/day per key).
- Retrieval surfaces the right article for covered crimes; penalty *text* is only
  guaranteed for the 5 expert-rule families (forgery, theft, homicide, assault,
  fraud/breach-of-trust).
- Embedding model is a small multilingual one; a better Arabic embedding + doc-aware
  re-chunking (index rebuild, 3–7h CPU) would raise the ceiling.
- ~15s latency is CPU retrieval+rerank, not the LLM.
- Memo legal *arguments* are not yet validated (only article numbers are).
