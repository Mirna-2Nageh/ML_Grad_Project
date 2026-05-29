# Conan API — Backend Integration Quickstart

One page to get the backend/frontend client talking to **Conan** (Arabic Egyptian-criminal-law
assistant). Full reference: **`backend_contract.md`**. Live, always-current API docs: **`GET /docs`**
(Swagger UI) and **`/openapi.json`**.

- **Base URL (dev):** `http://localhost:8000/api/v1` — production base is set by the deployment team.
- **Transport:** JSON over HTTP. Responses are **Modern Standard Arabic** in user-facing fields; JSON **keys are English**.

---

## 5 rules that will save you hours (read first)

1. **Echo `session_id`.** It's optional. Omit it on the first `/chat` turn → the response returns a
   fresh **UUID** → send that same id back on every later turn. Do **not** hard-code `"default"`.
2. **Set HTTP timeout ≥ 120 s.** Retrieval runs on CPU (~16–26 s/request, more for memos). A default
   30 s timeout **will** fail. This is the #1 integration bug.
3. **Handle `503`.** It carries an Arabic "try again later" message and means every LLM provider was
   momentarily out (free-tier budget). Show it, allow retry — it's normal, not a crash.
4. **Treat these as DYNAMIC** — never hard-code, assert on, or branch by exact value:
   `model`, `confidence_score`, `sources[]`, the Arabic `answer`/`analysis`/`memorandum` text, `latency_ms`.
   Only **field names + types** are the contract.
5. **Drive the UI off structured fields.** Render the `warnings[]` array and `confidence_score`; do
   **not** parse the Arabic answer text, and don't apply your own confidence threshold — the server
   already decides when to warn.

---

## Endpoints you'll actually call

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | — | readiness `{status, vectors, model, ...}` |
| `POST /qa` | `{ "question": "…" }` | answer envelope |
| `POST /chat` | `{ "message": "…", "session_id"?: "…" }` | answer envelope + `session_id`, `turn_count` |
| `POST /chat/stream` | same as `/chat` | **SSE** token stream + final event |
| `POST /weakness` | `{ "case_facts": "…" }` | envelope (`analysis`) |
| `POST /defense` | `{ "case_facts": "…" }` | envelope (`memorandum`) |
| `POST /forensic` | `{ "case_facts": "…", "evidence"?: "…" }` | envelope (`analysis`) |
| `POST /summarize` | `{ "text": "…" }` | `{ summary, input_length, latency_ms, model }` |
| `POST /parse` | multipart `file` **or** `text` | extracted text (no LLM, fast) |
| `POST /qa/upload` | multipart `question` + `file` | answer envelope (doc used as context) |

`question` must be ≥ 5 chars (else `422`). `.doc` uploads are rejected (`400`) — send `.docx`/`.pdf`/`.txt`.

---

## The shared answer envelope (`/qa`, `/chat`, `/weakness`, `/defense`, `/forensic`)

```jsonc
{
  "answer": "…",                 // Arabic. /weakness & /forensic use "analysis"; /defense uses "memorandum"
  "confidence_score": 0.84,      // float [0,1]; < 0.5 means the server flagged low confidence (see warnings)
  "confidence_factors": { "rerank_signal": 0.7, "source_count": 7, "article_validation": "passed", "topic_match": true },
  "sources": [ { "filename": "...", "article": "240", "referenced_articles": ["240"], "retrieval_score": 0.83, "rerank_score": 0.91, "page": null, "...": "..." } ],
  "warnings": [ "…Arabic notices…" ],
  "conflicts_detected": false,
  "latency_ms": 18450,
  "model": "llama-3.3-70b-versatile"   // DYNAMIC
}
```

`/chat` also returns `session_id`, `turn_count`, `was_compacted`. `/qa` also returns `retrieval_ms`.

---

## Copy-paste examples

```bash
# One-shot Q&A
curl -X POST http://localhost:8000/api/v1/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هي عقوبة السرقة بالإكراه في القانون المصري؟"}'

# Chat turn 1 — omit session_id, read it back from the response
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" -d '{"message":"ما عقوبة الرشوة؟"}'
# → response includes  "session_id": "a1b2c3..."

# Chat turn 2 — echo that session_id
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"وماذا لو كان الموظف عاماً؟","session_id":"a1b2c3..."}'
```

```csharp
// .NET — note the 120s timeout and 503 handling
var http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };

var resp = await http.PostAsJsonAsync("http://localhost:8000/api/v1/qa",
    new { question = "ما هي عقوبة السرقة بالإكراه؟" });

if ((int)resp.StatusCode == 503) { /* LLM busy — show retry, don't crash */ return; }
resp.EnsureSuccessStatusCode();

var qa = await resp.Content.ReadFromJsonAsync<QaResponse>();
// Render qa.Answer (Arabic), qa.ConfidenceScore, qa.Warnings; show qa.Sources as citations.

record QaResponse(string answer, double confidence_score, string[] warnings,
                  Source[] sources, bool conflicts_detected, double latency_ms, string model);
record Source(string filename, string? article, string[] referenced_articles,
              double retrieval_score, double rerank_score);
```

**Streaming** (`/chat/stream`): consume `text/event-stream`. Each line is `data: {json}`.
Token chunks: `{"chunk":"…","done":false}`. Final event: `{"done":true, …full envelope…}`.
Error: `{"error":"…","done":true}`.

---

## Status codes

| Code | Meaning | Client action |
|---|---|---|
| `200` | OK | render envelope |
| `400` | bad input / unsupported file (`.doc`) / empty extraction | show the Arabic `detail` |
| `404` | session not found / nothing retrieved | handle gracefully |
| `422` | request validation (e.g. `question` < 5 chars) | fix the request |
| `503` | all LLM providers momentarily out (budget) | show "try again", allow retry |

---

## Warnings worth special UI treatment (prefix-match the Arabic string)

| Prefix (Arabic) | Meaning | Suggested UI |
|---|---|---|
| `تنبيه: المواد التالية مذكورة في الإجابة لكنها غير موجودة في المصادر` | hallucinated citation | **red** banner, treat answer as suspect |
| `ثقة الإجابة منخفضة` | confidence < 0.5 | low-confidence banner |
| `السؤال غير مكتمل أو غير واضح` | input gate rejected a fragment | treat as "please rephrase" |
| `ملاحظة: تعذّر الوصول إلى النموذج الأساسي … نموذج احتياطي` | a backup LLM answered | optional info chip |

---

## Verify your integration (CI-friendly)

```bash
# Asserts every response SHAPE against this contract. Exit 0 = good, 1 = drift.
python legal-ai-assistant/scripts/contract_selftest.py --base-url http://localhost:8000/api/v1
#   --quick (default) = free, no-LLM checks — run on every commit
#   --full            = also hits LLM endpoints (a 503 is reported SKIP, not a failure)
```

---

## Before going to production

- **Add authentication** and **restrict CORS** (currently `*`).
- Put the Conan FastAPI service **behind your backend/gateway** — don't expose it to the internet raw.
- Keep an eye on `503` rate (free-tier budget); production should use funded LLM keys.
