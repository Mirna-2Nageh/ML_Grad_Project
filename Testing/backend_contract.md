# Conan — Backend API Contract (post-v8)

**Service:** Legal AI Assistant for Egyptian Criminal Law
**Base URL:** `http://localhost:8000/api/v1` (dev) — production base set by the deployment team
**Auth:** None at the moment. CORS is permissive (`*`). Lock down in production.
**Response language:** Modern Standard Arabic in user-facing fields; English in keys.
**OpenAPI spec:** live at `GET /docs` (Swagger UI) and `GET /openapi.json`.
**Last updated:** 2026-05-29 (branch `llm-reliability-expert-rules`, tip `0c927a3`) — system renamed Nour → Conan; LLM reliability + multi-provider + answer-cache + confidence-recalibration series. See §13 for the full changelog. Verify any deployment with `python legal-ai-assistant/scripts/contract_selftest.py --full`.

This document supersedes `legal-ai-assistant/api_contract.md` (pre-v8) and reflects every endpoint added or modified during the v3 → v8 accuracy work plus the document-upload + iterative-retrieval rollouts.

---

## 0. Integration stability — what is frozen vs. what changes under the hood

**You can integrate now.** The integration surface is the **JSON shape** in this document —
endpoint paths, field **names**, and field **types**. That is frozen; we will only add
fields, never rename or remove them.

**These change between versions and are NOT part of the contract — treat them as dynamic
data, never hard-code, assert, or branch on exact values:**
- `model` (which LLM answered — may be groq / cerebras / gemini depending on fallback)
- `confidence_score` and `confidence_factors` (recalibrated over time)
- `answer` / `analysis` / `memorandum` **text**, `sources[]` (filenames, articles, scores)
- `latency_ms`, `warnings[]` contents

So our planned upgrades — **index rebuild (bge-m3), confidence recalibration, new LLM
providers, answer caching** — require **zero code changes on your side**. Your client keeps
working automatically. The only coordination needed: after we swap in a rebuilt index we
restart the service and clear the answer cache, and you should re-run your golden-path
checks because the *values* above will shift (the *shapes* won't).

**To integrate cleanly, your client MUST:**
1. **Echo `session_id`** — omit it on turn 1, read it from the response, send it back on
   every later turn. (Do not assume a shared `"default"` session; that changed.)
2. **Use a long HTTP timeout — ≥ 120 s.** Retrieval+rerank run on CPU (~16–26 s/request,
   more for memos). A default 30 s timeout will fail. This is the #1 integration gotcha.
3. **Handle `503`** gracefully — it carries an Arabic "try again later" message and means
   every LLM provider was momentarily unavailable (free-tier budget). Show it, allow retry.
4. **Drive the UI off structured fields** — render the `warnings[]` array and
   `confidence_score`; do **not** parse the Arabic `answer` text or apply your own
   confidence thresholds (the server already decides when to warn).
5. **Stream** (`/chat/stream`): use an SSE client; the final `done:true` event carries
   confidence/sources/warnings; an error arrives as `{"error": "...", "done": true}`.

**Production (before exposing beyond localhost):** add auth and restrict CORS (currently
`*`); put the FastAPI service behind your backend/gateway rather than exposing it directly.

---

## 1. Conventions (read this first)

### 1.1 The standard answer envelope

`POST /qa`, `POST /qa/upload`, `POST /chat`, `POST /weakness`, `POST /defense`, and `POST /forensic` all return an envelope with the same set of fields:

| Field | Type | Notes |
|---|---|---|
| **answer** (or `analysis` / `memorandum`) | string (Arabic, MSA) | The user-facing answer. `/qa` & `/chat` use `answer`; `/weakness` & `/forensic` use `analysis`; `/defense` uses `memorandum`. |
| `confidence_score` | float `[0, 1]` | Weighted heuristic (Section 2.4). Round/display as percent. |
| `confidence_factors` | object | Per-signal breakdown (see 1.3). |
| `sources` | array of `SourceInfo` | Retrieved chunks the answer was built on. See 1.2. |
| `warnings` | array of strings (Arabic) | User-facing notices (see 1.4). |
| `conflicts_detected` | bool | Currently always `false`. Phase 2 will populate this when retrieved sources contradict each other. |
| `latency_ms` | float | Total handler latency. |
| `retrieval_ms` | float | Retrieval-only latency. Only on `/qa` and `/qa/upload`. |
| `model` | string | Identifier of the model that produced the final answer (e.g. `llama-3.3-70b-versatile`, `gemini-2.5-flash`, `input_gate`). |

### 1.2 `SourceInfo` (per retrieved chunk)

```jsonc
{
  "filename": "الباب_الأول_العقوبات.txt",
  "source": "../final_total_dataset/قانون العقوبات/الباب_الأول.txt",
  "doc_type": "penal_code",
  "legal_category": "general",
  "legal_topic": "تزوير",
  "article": "240",
  "referenced_articles": ["240", "241"],
  "page": null,
  "retrieval_score": 0.8312,
  "rerank_score": 0.9211
}
```

Special source entry — surfaces when the **article-lookup rescue** triggers (Section 4):

```jsonc
{ "source": "[article-lookup rescue]", "referenced_articles": ["87"], ... }
```

The .NET frontend should treat `source == "[article-lookup rescue]"` as a marker to optionally display a separate "extra references found" indicator.

### 1.3 `ConfidenceFactors`

```jsonc
{
  "rerank_signal":      0.87,           // mean rerank score over final top-k, [0,1]
  "source_count":       7,              // count of retrieved chunks
  "article_validation": "passed",       // "passed" | "failed" | "not_applicable"
  "topic_match":        true            // bool — true when a source's legal_topic appears in the question
}
```

`article_validation == "failed"` is the strongest signal of an ungrounded citation. UI should flag this prominently.

### 1.4 Canonical warning strings (for i18n / handling)

These exact Arabic strings appear in the `warnings` array. The frontend can pattern-match on prefixes to decide UI treatment.

| Prefix | Meaning | Frontend action |
|---|---|---|
| `تنبيه: المواد التالية مذكورة في الإجابة لكنها غير موجودة في المصادر` | LLM hallucinated article(s). | Show red banner. Strongly recommend treating answer as suspect. |
| `تم تصحيح الاستشهادات بعد التحقق من المصادر (محاولات: N)` | Corrective retry repaired the citations. | Optional info chip. |
| `تم استرجاع مواد إضافية من قاعدة البيانات للتحقق من الاستشهادات (مواد: ...)` | Article-lookup rescue fired and succeeded. | Optional info chip. |
| `تم توسيع نطاق البحث تلقائياً إلى N مرجعاً` | Iterative retrieval auto-expanded k. | Optional info chip. |
| `تم تضمين المستند المرفق (...)` | An uploaded doc was used as context. | Info chip noting the attachment. |
| `النصوص المسترجعة لا تتضمن إجابة كاملة على هذا السؤال — اعتُبر الردّ ردّ تعذّر` | Answer is effectively a refusal. | Show low-confidence banner. |
| `ثقة الإجابة منخفضة — يُنصح بإعادة صياغة السؤال` | Confidence below 0.5 threshold. | Show low-confidence banner. |
| `السؤال غير مكتمل أو غير واضح` | Input gate rejected fragment. | Treat answer as polite ask-for-clarification. |
| `ملاحظة: تعذّر الوصول إلى النموذج الأساسي مؤقتًا، وتم توليد الإجابة بنموذج احتياطي` | Primary LLM unavailable; fallback used. | Optional info chip. |
| `الملفات بصيغة .doc (Word 97-2003 binary) غير مدعومة مباشرة` | `.doc` upload rejected. | Show error explaining to save as `.docx` or `.pdf`. |

### 1.5 Standard error shape

Errors use FastAPI's `{"detail": "..."}` shape with appropriate HTTP status:

| Status | Meaning |
|---|---|
| `400` | Bad input (missing required form fields, unsupported file format, empty extraction). |
| `404` | Resource not found (session doesn't exist, no chunks retrieved). |
| `500` | Internal error. |
| `503` | All LLM providers exhausted (rate-limited or quota). Standard message: `تعذّر توليد إجابة حالياً بسبب عدم توفر نموذج اللغة (انشغال أو تجاوز الحصة). يرجى المحاولة مرة أخرى بعد قليل.` |

---

## 2. Health

### `GET /api/v1/health`

Smoke-test endpoint. Cheap, no LLM calls.

**Response (`200 OK`):**
```jsonc
{
  "status": "ok",                                          // "ok" | "degraded" | "loading"
  "vectors": 47028,                                        // count in FAISS
  "chunks": 47028,                                         // count in chunks.pkl
  "model": "llama-3.3-70b-versatile",                      // ACTIVE primary model — dynamic; may be groq/cerebras/gemini on fallback. Do not hard-code.
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "reranker_loaded": true,
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "load_errors": []
}
```

---

## 3. Q&A

### 3.1 `POST /api/v1/qa` — standard Q&A (JSON body)

**Request:**
```jsonc
{
  "question": "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",  // 5–1000 chars
  "k": 7,                                                  // 1–20, default 7
  "prompt_style": "restrictive"                            // "standard" | "restrictive", default "restrictive"
}
```

**Response (`200 OK`):** standard envelope with `answer`, `confidence_score`, `confidence_factors`, `sources`, `warnings`, `latency_ms`, `retrieval_ms`, `model`.

**Behaviour notes:**
- **Input gate** (v4+): if the question looks like a markdown header, bullet, single word, or sentence ending in a colon, the endpoint returns immediately (~9 ms) with `model == "input_gate"`, `confidence_score: 0.0`, and warning `السؤال غير مكتمل أو غير واضح.` — no LLM call.
- **Iterative retrieval** (v8.1+): when evidence validation flags missing articles, the retrieval window is auto-expanded through `[7, 14, 21]` before any retry. Surfaces the "توسيع نطاق البحث" warning when triggered.
- **Corrective retry** (v4+): if validation still fails, one retry with explicit block-list of forbidden article numbers. Surfaces the "تم تصحيح الاستشهادات" warning on success.
- **Article-lookup rescue** (v6+): if retry still fails, the missing article numbers are looked up directly in the 1,494-article index. Up to 2 chunks per missing article are appended to the context for one more LLM pass. Surfaces the "تم استرجاع مواد إضافية" warning on success and adds synthetic source entries with `source: "[article-lookup rescue]"`.
- **Post-processing** (v6+): strips casual openers (`حسناً`, `بالتأكيد`, …) and rewrites mid-clause refusal-template leakage.
- **Hard penalty:** when validation ultimately fails, `confidence_score` is reduced by 0.4.

**Example call:**
```
curl -X POST http://localhost:8000/api/v1/qa \
  -H 'Content-Type: application/json' \
  -d '{"question":"ما هي شروط التوقيف الاحتياطي؟","k":7}'
```

### 3.2 `POST /api/v1/qa/upload` — Q&A with attached document (NEW in upload rollout)

Same pipeline as `/qa`, but accepts a per-question document attachment. The attached text is prepended to the retrieved context for **this request only** (not persisted, not added to the index). Article numbers cited in the answer can come from the attachment OR the retrieved corpus — the evidence validator accepts both.

**Request** (`multipart/form-data`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | 1–2000 chars |
| `file` | file | one of | `.txt` / `.pdf` / `.docx`. `.doc` legacy binary is NOT supported. |
| `text` | string | one of | Pasted raw text (alternative to `file`). |
| `k` | int | no | 1–20, default 7 |
| `prompt_style` | string | no | default `"restrictive"` |

If both `file` and `text` are provided, `file` wins.

**Response (`200 OK`):** same standard envelope.

**Special behaviour:**
- The **input gate is bypassed** when an attachment is present (the question "لخّص" alone is a fragment, but "لخّص" + an attached PDF is a meaningful request).
- A warning is always surfaced: `تم تضمين المستند المرفق ({filename}, N حرفاً) في السياق.` so the user sees the attachment was used.

**Error (`400`):** returned when parsing fails — `detail` field contains the Arabic reason (e.g. `.doc` rejection, OCR-failed PDF, content too short).

**Example call:**
```
curl -X POST http://localhost:8000/api/v1/qa/upload \
  -F 'question=ما عقوبة الفعل في المستند المرفق؟' \
  -F 'file=@case.pdf;type=application/pdf' \
  -F 'k=7'
```

---

## 4. Chat (session-aware)

Multi-turn dialogue with sliding-window compaction and disk persistence. Sessions also support attached documents (Section 4.4) that ride along in every turn.

### 4.1 `POST /api/v1/chat`

**Request:**
```jsonc
{
  "message": "ما عقوبة السرقة بالإكراه؟",
  "session_id": "user-123",            // OPTIONAL. Omit on the FIRST turn → the server
                                       //   generates a fresh UUID and returns it in the
                                       //   response. Read it there and echo it back on
                                       //   every later turn to keep the conversation.
                                       //   (Older builds defaulted to a shared "default"
                                       //   session — do NOT rely on that anymore.)
  "k": 7                                // 1–20, default 7
}
```

**Response (`200 OK`):**
```jsonc
{
  "answer":              "...",                  // Arabic
  "session_id":          "user-123",
  "confidence_score":    0.64,
  "confidence_factors":  { ... },
  "sources":             [ ... ],
  "warnings":            [ ... ],
  "conflicts_detected":  false,
  "latency_ms":          24123,
  "turn_count":          3,                       // total user-turns this session
  "was_compacted":       false,                   // true when history was summarised this turn
  "model":               "llama-3.3-70b-versatile"
}
```

Behaviour shared with `/qa`: input gate, iterative retrieval, retry, article-lookup rescue, post-processing. Plus:
- The session's compact summary + recent message history is included in the prompt.
- When the session has attached documents (Section 4.4), they are prepended to the retrieved context every turn.

### 4.2 `POST /api/v1/chat/stream`

Same body as `/chat`. Returns `text/event-stream` (SSE):

- During generation: `data: {"chunk": "...", "done": false}\n\n` per token chunk.
- Final event: `data: {"done": true, "session_id": "...", "confidence_score": ..., "confidence_factors": {...}, "sources": [...], "warnings": [...], "conflicts_detected": false, "turn_count": N, "latency_ms": ..., "model": "..."}\n\n`
- On error: `data: {"error": "...", "done": true}\n\n`

Notes:
- Streaming tries the configured primary (Groq) with multi-key rotation, then OpenRouter, then falls back to the full non-streaming chain (chunked out) — a rate-limited primary degrades gracefully instead of failing. On total failure it emits one clean error event (`{"error": "<Arabic message>", "done": true, ...}`), never a raw provider error. The `model` in the final event reflects whichever provider actually answered.
- The streaming endpoint applies the input gate but **does not** run the corrective retry or article-lookup rescue (those are incompatible with token-by-token streaming). For maximum grounding, use `/chat` not `/chat/stream`.

### 4.3 `GET /api/v1/chat/{session_id}`

**Response (`200 OK`):**
```jsonc
{
  "session_id":    "user-123",
  "turn_count":    3,
  "message_count": 6,
  "has_summary":   false,
  "created_at":    1779559123.45,    // unix timestamp
  "last_active":   1779559842.10
}
```

**Response (`404`):** session not found.

### 4.4 `DELETE /api/v1/chat/{session_id}`

Clears the session from memory AND disk. Cascades to all attachments.

**Response (`200 OK`):** `{"status": "deleted", "session_id": "user-123"}`
**Response (`404`):** session not found in memory or on disk.

### 4.5 Session-attached documents (NEW in upload rollout)

Three companion endpoints let a frontend bind a document to a session so subsequent `/chat` turns automatically include it.

#### `POST /api/v1/chat/attach`

**Request** (`multipart/form-data`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | yes | Will be created if doesn't exist. |
| `file` | file | one of | `.txt` / `.pdf` / `.docx` |
| `text` | string | one of | Pasted text |

**Response (`200 OK`):**
```jsonc
{
  "session_id":         "user-123",
  "attached":           {
    "doc_id":           "94a3f7246ea9",      // unique within the session
    "filename":         "contract.pdf",
    "content_type":     ".pdf",
    "char_count":       8421,
    "attached_at":      1779559123.45
  },
  "attachments_total":  1,                    // total docs attached after this one
  "total_chars":        8421,                 // sum across all attached docs
  "warnings":           []                    // non-fatal parsing notes
}
```

After attaching, every subsequent `POST /chat` for that `session_id` includes ALL attached docs in the context block under the `[المستندات المرفقة]` delimiter. Attachments persist with the session to disk.

#### `GET /api/v1/chat/{session_id}/attachments`

**Response (`200 OK`):**
```jsonc
{
  "session_id":  "user-123",
  "attachments": [
    {
      "doc_id":       "94a3f7246ea9",
      "filename":     "contract.pdf",
      "content_type": ".pdf",
      "char_count":   8421,
      "attached_at":  1779559123.45
    }
  ],
  "total_chars": 8421
}
```

Empty `attachments` (with `200 OK`) is returned for any unknown session — never `404` (lets the frontend poll without special-casing).

#### `DELETE /api/v1/chat/{session_id}/attachments/{doc_id}`

Remove a single attachment.
- `200 OK`: `{"status": "removed", "session_id": "...", "doc_id": "..."}`
- `404`: attachment not found.

#### `DELETE /api/v1/chat/{session_id}/attachments`

Clear all attachments for a session.
- `200 OK`: `{"status": "cleared", "session_id": "...", "removed_count": N}`

---

## 5. Summarization

### `POST /api/v1/summarize`

**Request:**
```jsonc
{
  "text": "المادة الأولى: ... المادة الثانية: ..."   // 50–15000 chars
}
```

**Response (`200 OK`):**
```jsonc
{
  "summary":    "...",                       // Arabic, numbered list
  "input_length": 1842,
  "latency_ms": 3210,
  "model":      "llama-3.3-70b-versatile"
}
```

No retrieval, no validation, no confidence — purely a text-in / text-out endpoint.

**Tip for the UI:** combine with `/parse` to support file-driven summarization (upload → parse → submit cleaned text to `/summarize`). The Streamlit Tab 2 implementation in this repo is the reference flow.

---

## 6. Weakness Analysis

### `POST /api/v1/weakness`

Analyses a criminal case from the defence perspective and identifies weaknesses in the prosecution.

**Request:**
```jsonc
{
  "case_facts":          "المتهم متهم بالسرقة بالإكراه...",  // 20–10000 chars, required
  "evidence":            "التقرير الطبي: لا توجد إصابات...", // 0–15000 chars, optional
  "defendant_statement": "ادعى المتهم الدفاع الشرعي..."     // 0–5000 chars, optional
}
```

**Response (`200 OK`):** standard envelope with `analysis` (not `answer`), `confidence_score`, `confidence_factors`, `sources`, `warnings`, `latency_ms`, `model`.

Note: separate `evidence` and `defendant_statement` fields are part of the contract — the prompts use them explicitly. UI should expose them as separate inputs (the Streamlit Tab 3 reference implementation does).

---

## 7. Defense Memorandum

### `POST /api/v1/defense`

Drafts a formal Egyptian defense memorandum (`مذكرة دفاع`) under the headings الوقائع / الإطار القانوني / أوجه الدفاع / الطلبات.

**Request:**
```jsonc
{
  "case_facts":          "...",   // 20–10000 chars, required
  "weaknesses":          "...",   // 0–5000 chars, optional (e.g. paste from /weakness)
  "evidence":            "...",   // 0–15000 chars, optional
  "defendant_statement": "..."    // 0–5000 chars, optional
}
```

**Response (`200 OK`):** standard envelope with `memorandum` (not `answer`), confidence/sources/warnings, plus:

| Field | Type | Notes |
|---|---|---|
| `self_check_revisions` | int | Number of agentic self-check passes (draft → verify → revise) applied to the memo. `0` if disabled or pass-on-first-try. Controlled by `MEMO_SELF_CHECK` env var. |

---

## 8. Forensic-Consistency Analysis

### `POST /api/v1/forensic`

Documentary/logical consistency check. NOT physical forensics — analyses the case file for internal contradictions, evidence-fact mismatches, and unmet legal elements.

**Request:**
```jsonc
{
  "case_facts": "...",          // 20–10000 chars, required
  "evidence":   "..."            // 0–15000 chars, optional
}
```

**Response (`200 OK`):** standard envelope with `analysis` (not `answer`) plus a meaningful `conflicts_detected` boolean — `true` when the LLM found a contradiction / evidence mismatch / unmet element.

---

## 9. Document Parsing (NEW)

### `POST /api/v1/parse`

One-shot file/text → extracted text. **No LLM call, no retrieval.** Used by frontends to preview/edit parsed content before submitting to `/weakness`, `/defense`, `/summarize`, etc.

**Request** (`multipart/form-data`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | one of | `.txt` / `.pdf` / `.docx` |
| `text` | string | one of | Pasted text |

If both provided, `file` wins.

**Response (`200 OK`):**
```jsonc
{
  "text":        "...",                  // extracted + Arabic-cleaned
  "filename":    "case.pdf",             // "pasted-text" for raw strings
  "content_type": ".pdf",                // ".txt" / ".pdf" / ".docx" / ".text"
  "char_count":  4321,
  "warnings":    []                       // truncation / encoding-fallback notes
}
```

**Response (`400`):** Arabic-readable `detail` explaining the parsing failure (unsupported format, content too short, OCR-failed PDF).

---

## 10. Data Ingestion (permanent index)

### 10.1 `POST /api/v1/ingest`

Adds new documents to the permanent FAISS + BM25 indices. Visible in all future searches.

**Request** (`multipart/form-data`): one or more `files` (`.txt` / `.pdf` / `.docx`).

**Response (`200 OK`):**
```jsonc
{
  "status":          "ok",            // "ok" | "partial" | "no_data" | "error"
  "files_processed": 3,
  "chunks_created":  127,
  "vectors_added":   127,
  "errors":          [],
  "duration_s":      18.4
}
```

### 10.2 `POST /api/v1/ingest/scan`

Scans the configured `INGEST_INBOX_DIR` for files not yet in the registry and ingests them. Same response shape as `/ingest`. Idempotent.

---

## 11. Sample multi-step flows (for the .NET team)

### Flow A — "Analyse this case file" (one-shot, no session)

```
1. POST /parse   (file)           → preview text, let user edit
2. POST /weakness (case_facts)    → analysis + sources + confidence
```

### Flow B — "Multi-turn chat about a contract"

```
1. POST /chat/attach              (session_id, file)  → attached.doc_id
2. POST /chat                     (session_id, message)
3. POST /chat                     (session_id, message)  // attachment persists
4. ...
5. DELETE /chat/{session_id}/attachments  // when done with that doc
```

### Flow C — "Quick legal Q&A, no context to share"

```
1. POST /qa  (question, k=7)      → answer + sources + confidence
```

### Flow D — "Quick legal Q&A about a specific file (one-shot)"

```
1. POST /qa/upload  (question, file)  → answer + sources + confidence
                                       // attachment NOT persisted; one-shot
```

### Flow E — "Add a new statute to the searchable corpus"

```
1. POST /ingest  (files=[law.pdf])  → chunks_created, vectors_added
   // After this, /qa, /chat, etc. will find this content
```

---

## 12. Latency expectations (paid-tier estimates)

Free-tier (Groq llama-3.3-70b) introduces variance from rate-limit fallbacks. The numbers below assume a paid LLM tier with no TPM throttling.

| Endpoint | Cold | Warm | Notes |
|---|---:|---:|---|
| `/health` | 5 ms | 3 ms | No LLM, no retrieval. |
| `/parse` (text) | 30 ms | 20 ms | Pure parsing. |
| `/parse` (PDF) | 800 ms | 500 ms | Dominated by `pdftotext`. |
| `/qa` (no rescue needed) | 22 s | 18 s | Retrieve + 1 LLM call. |
| `/qa` (with retry) | 35 s | 30 s | +1 LLM call. |
| `/qa` (with rescue) | 50 s | 42 s | +2 LLM calls. |
| `/qa/upload` | +0.5 s | +0.5 s | Parsing overhead. |
| `/chat` | similar to `/qa` |  |  |
| `/chat/stream` | first-token ~3 s | ~3 s | Full response streams over ~20–60 s. |
| `/ingest` (1 medium PDF) | 30 s | 30 s | Embed + FAISS merge. |
| `/qa` input-gated | 9 ms | 9 ms | No LLM, no retrieval. |

---

## 13. Versioning & change tracking

This contract documents the state of the API as of the **v8 enhancement series** (commits `14f1d45`, `03249a4`, `2e05967`, `c66f74f`, `85865c7` on branch `llm-reliability-expert-rules`).

Endpoints added since the previous (pre-v8) contract:

| Endpoint | Added in commit |
|---|---|
| `POST /qa/upload` | `03249a4` |
| `POST /chat/attach` | `03249a4` |
| `GET /chat/{sid}/attachments` | `03249a4` |
| `DELETE /chat/{sid}/attachments/{doc_id}` | `03249a4` |
| `DELETE /chat/{sid}/attachments` | `03249a4` |
| `POST /parse` | `85865c7` |

Behavioural changes to **existing** endpoints since the previous contract:

| Endpoint | What changed | Commit |
|---|---|---|
| `POST /qa` | input gate, corrective retry, article-lookup rescue, post-processing, iterative retrieval | `14f1d45`, `85865c7` |
| `POST /chat` | same defence pipeline + reads session attachments every turn | `14f1d45`, `03249a4`, `85865c7` |
| `POST /chat/stream` | input gate added; rescue not applied (streaming-incompatible) | `14f1d45`, `03249a4` |
| `DELETE /chat/{sid}` | now also handles disk-only sessions (lazy-load) | `cded4e0` |

### Latest series — reliability, multi-provider & rename (2026-05-29, branch `llm-reliability-expert-rules`)

| Change | Effect on the contract | Commit |
|---|---|---|
| **System renamed Nour → Conan** (Arabic persona نور → كونان) | Cosmetic — appears in `answer` text & banners; **no field changes**. | `0c927a3` |
| `session_id` is now **OPTIONAL** | Omit on turn 1 → server returns a fresh UUID (previously a shared `"default"`). Echo it back on later turns. **This is the only client-visible semantic change to adopt.** | `f6da853` |
| `POST /chat/stream` hardened | primary → OpenRouter → non-streaming fallback; clean Arabic error event on total failure (never a raw provider error). Same SSE event shapes. | `f6da853` |
| Confidence recalibrated | `confidence_score` shifts upward for grounded answers (still `[0,1]`, threshold 0.5). Treat as **dynamic**. | `f6da853` |
| Answer cache for `/qa` | Repeated identical questions return instantly (`latency_ms`≈0); **identical response shape**. | `f6da853` |
| Multi-provider chain (Cerebras + multi-project Gemini) + rate gate | `model` may now report `cerebras`/`gemini`/`groq`. Treat `model` as **dynamic**. | `0636740` |
| `/health` `model` field | Reports the **active** primary model, not a static default. | `f6da853` |

A runnable **contract self-test** ships at `legal-ai-assistant/scripts/contract_selftest.py`: it asserts every response shape against this document (`--full` exercises the LLM endpoints too; a `503` is reported as SKIP, not a failure). Both teams should run it in CI to catch drift automatically.

The `confidence_factors` and `SourceInfo` schemas are **stable wire contract** — adding fields is backwards-compatible (frontend should tolerate unknown keys); renaming or removing is a breaking change.
