# 🔗 API Contract — Legal AI Assistant ("كونان")

> **For:** .NET Backend Team
> **Version:** 2.0.0 (breaking changes vs. 1.0.0 — see "Migration" below)
> **Base URL:** `http://<server>:8000/api/v1`
> **Interactive docs:** `http://<server>:8000/docs` (Swagger) · `/redoc`

## Migration from 1.0.0

If you previously used the 1.0.0 contract, your existing endpoints still respond, but the response shape has expanded:

| Endpoint | What changed |
|---|---|
| `/qa`, `/chat`, `/weakness`, `/defense` | Added `confidence_score`, `confidence_factors`, `warnings`, `conflicts_detected`. `sources[]` items gained `legal_topic`, `article`, `referenced_articles`, `page`, `retrieval_score`, `rerank_score`. |
| `/health` | Added `reranker_loaded`, `reranker_model`, `load_errors[]`. |
| `/chat/stream` | **New** — SSE token streaming. |
| `/ingest/scan` | **New** — trigger server-side inbox scan. |
| `/chat/{session_id}` (GET/DELETE) | Already existed; documented here for completeness. |

All new fields are additive — existing 1.0.0 clients will receive them but can ignore them. The breaking part is that **field names on `sources[]` have not changed**, so old DTOs still bind, but new fields will be missing unless DTOs are extended.

---

## 1. Conventions

- All requests/responses are JSON, UTF-8. Arabic strings are stored and returned as UTF-8.
- All timestamps are server-side wall-clock seconds.
- All latencies are in **milliseconds** (`*_ms` fields) or **seconds** (`*_s` fields, only for `IngestResponse.duration_s`).
- Property names are **snake_case** on the wire. Configure your .NET `JsonSerializerOptions`:
  ```csharp
  new JsonSerializerOptions {
      PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
      WriteIndented = false,
      Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping  // preserve Arabic
  }
  ```
- Errors follow `{"error": "...", "detail": "..."}` for 4xx/5xx unless noted.

---

## 2. Common Schemas

### `SourceInfo`

Returned inside every retrieval-backed response. One entry per retrieved chunk.

| Field | Type | Notes |
|---|---|---|
| `filename` | string | Source filename (e.g. `قانون العقوبات.txt`) |
| `source` | string | Full source path on the server's filesystem |
| `doc_type` | string | One of: `penal_code`, `criminal_procedure`, `criminal_case`, `cassation_ruling`, `cassation_encyclopedia`, `legal_rules_collection`, `forensic_medicine`, `criminal_law_reference`, `legal_reference` |
| `legal_category` | string | Legacy field; same as `legal_topic` for encyclopedia files, `"general"` otherwise |
| `legal_topic` | string | Encyclopedia folder taxonomy (e.g. `"تزوير"`, `"قتل عمد"`); empty for non-encyclopedia |
| `article` | string \| null | First article number cited in the chunk (e.g. `"314"`) — Western digits, normalized from any Arabic/Eastern variant |
| `referenced_articles` | string[] | All article numbers in the chunk, sorted numerically |
| `page` | int \| null | Source page number when available (PDF only); currently always `null` until OCR pipeline lands |
| `retrieval_score` | float | Dense (FAISS) cosine score, 0–1. May be `0.0` for sources that surfaced only via BM25 |
| `rerank_score` | float | Cross-encoder relevance score (sigmoid-bounded), 0–1. **The primary quality signal.** |

### `ConfidenceFactors`

Per-component breakdown of `confidence_score`.

| Field | Type | Notes |
|---|---|---|
| `rerank_signal` | float | Mean rerank score over the returned sources, 0–1 |
| `source_count` | int | Number of `sources[]` returned |
| `article_validation` | string | `"passed"`, `"failed"`, or `"not_applicable"` (when the answer doesn't cite any article) |
| `topic_match` | bool | True if any source's `legal_topic` substring appears in the question text |

### Confidence score interpretation

| Range | Meaning | UI suggestion |
|---|---|---|
| ≥ 0.7 | High — grounded, multiple aligned sources | Show answer normally |
| 0.4 – 0.69 | Medium — some uncertainty | Show answer with a "verify before relying on this" hint |
| < 0.4 | Low — should ask the user to clarify | Surface `warnings[]` prominently; consider not showing the answer |

`warnings[]` may already contain Arabic-language clarification prompts. Forward them to the user verbatim.

---

## 3. Endpoints

### 3.1  `GET /api/v1/health`

Returns service readiness. Use to check before sending real queries.

**Response 200:**
```json
{
  "status": "ok",
  "vectors": 47028,
  "chunks": 47028,
  "model": "qwen/qwen-2.5-72b-instruct",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "reranker_loaded": true,
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "load_errors": []
}
```

`status` is `ok` if fully ready, `degraded` if running with `load_errors`, `loading` during startup.

---

### 3.2  `POST /api/v1/qa`  — Legal Q&A (stateless)

Single-turn legal question. No conversation memory.

**Request:**
```json
{
  "question": "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
  "k": 7,
  "prompt_style": "restrictive"
}
```

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `question` | string | ✅ | — | 5–1000 chars, Arabic |
| `k` | int | ❌ | 10 | 1–30 |
| `prompt_style` | string | ❌ | `"restrictive"` | `"standard"` (lenient) or `"restrictive"` (strict context-only) |

**Response 200:**
```json
{
  "answer": "وفقاً للمادة 314 من قانون العقوبات...",
  "confidence_score": 0.87,
  "confidence_factors": {
    "rerank_signal": 0.85,
    "source_count": 5,
    "article_validation": "passed",
    "topic_match": true
  },
  "sources": [
    {
      "filename": "قانون العقوبات.txt",
      "source": "/.../قانون العقوبات.txt",
      "doc_type": "penal_code",
      "legal_category": "general",
      "legal_topic": "",
      "article": "314",
      "referenced_articles": ["313", "314", "317"],
      "page": null,
      "retrieval_score": 0.91,
      "rerank_score": 0.98
    }
  ],
  "warnings": [],
  "conflicts_detected": false,
  "latency_ms": 2450.3,
  "retrieval_ms": 120.5,
  "model": "qwen/qwen-2.5-72b-instruct"
}
```

**Possible 4xx/5xx:**
- `404` — no relevant documents found in retrieval
- `422` — request validation failed
- `500` — LLM service unavailable or internal error

---

### 3.3  `POST /api/v1/chat`  — Multi-turn Chat (session-aware)

Maintains conversation history per `session_id`. Auto-compacts at 6 turns.

**Request:**
```json
{
  "message": "وماذا لو كان من شخصين فأكثر؟",
  "session_id": "user-123",
  "k": 7
}
```

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `message` | string | ✅ | — | Min 1 char |
| `session_id` | string | ❌ | `"default"` | Sanitized to `[A-Za-z0-9_-]{1,128}` server-side |
| `k` | int | ❌ | 10 | 1–30 |

**Response 200:** Same shape as `/qa` plus:
```json
{
  "session_id": "user-123",
  "turn_count": 3,
  "was_compacted": false
}
```

Sessions are persisted to disk and survive server restarts. Idle sessions older than 168 h (7 days) are pruned. Use `session_id` to keep a user's conversation coherent across requests.

---

### 3.4  `POST /api/v1/chat/stream`  — Streaming Chat (SSE)

Same input and behaviour as `/chat`, but the response streams as Server-Sent Events while the LLM generates tokens. Use this for UI responsiveness.

**Request:** identical to `/chat`. Set `Accept: text/event-stream`.

**Response 200 — `Content-Type: text/event-stream`:**

Per token chunk while generating:
```
data: {"chunk":"وفقاً","done":false}

data: {"chunk":" للمادة","done":false}

data: {"chunk":" 314","done":false}
```

Final event — full metadata, ends the stream:
```
data: {
  "done": true,
  "session_id": "user-123",
  "confidence_score": 0.87,
  "confidence_factors": { ... },
  "sources": [ ... ],
  "warnings": [],
  "conflicts_detected": false,
  "turn_count": 3,
  "latency_ms": 4820.1,
  "model": "qwen/qwen-2.5-72b-instruct"
}
```

On error:
```
data: {"error":"...","done":true}
```

> **Provider note:** streaming runs OpenRouter-only (Qwen). Non-streaming `/chat` retains the Gemini→OpenRouter fallback chain. If Gemini key isn't set, both behave identically.

---

### 3.5  `GET /api/v1/chat/{session_id}`  — Session Info

**Response 200:**
```json
{
  "session_id": "user-123",
  "turn_count": 3,
  "message_count": 6,
  "has_summary": false,
  "created_at": 1736942400.123,
  "last_active": 1736944800.456
}
```

`404` if session doesn't exist.

---

### 3.6  `DELETE /api/v1/chat/{session_id}`  — Clear Session

**Response 200:** `{"status": "deleted", "session_id": "user-123"}`

`404` if session doesn't exist.

---

### 3.7  `POST /api/v1/summarize`  — Legal Text Summarization

Stateless summarization of a provided text (no retrieval).

**Request:**
```json
{
  "text": "المادة الأولى: يعاقب بالحبس..."
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `text` | string | ✅ | 50–50000 chars |

**Response 200:**
```json
{
  "summary": "• المادة 1: عقوبة الحبس...",
  "input_length": 1523,
  "latency_ms": 1890.2,
  "model": "qwen/qwen-2.5-72b-instruct"
}
```

> No `confidence_score` / `sources` — there's no retrieval and no citations to validate.

---

### 3.8  `POST /api/v1/weakness`  — Case Weakness Detection

Analyses case facts for prosecution-side weaknesses. Same response shape as `/qa` but with `analysis` instead of `answer`.

**Request:**
```json
{
  "case_facts": "المتهم متهم بالسرقة بالإكراه ليلاً. تم القبض عليه بناءً على بلاغ مجهول."
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `case_facts` | string | ✅ | 20–50000 chars |

**Response 200:**
```json
{
  "analysis": "1. المخالفات الإجرائية: القبض بناءً على بلاغ مجهول...",
  "confidence_score": 0.74,
  "confidence_factors": { ... },
  "sources": [ ... ],
  "warnings": [],
  "conflicts_detected": false,
  "latency_ms": 3200.1,
  "model": "qwen/qwen-2.5-72b-instruct"
}
```

---

### 3.9  `POST /api/v1/defense`  — Defense Memorandum

Generates a formal court defense memorandum. Same response shape as `/weakness` but with `memorandum` instead of `analysis`.

**Request:**
```json
{
  "case_facts": "المتهم متهم بالسرقة بالإكراه.",
  "weaknesses": "عدم وجود شهود عيان"
}
```

| Field | Type | Required | Default | Constraints |
|---|---|---|---|---|
| `case_facts` | string | ✅ | — | 20–50000 chars |
| `weaknesses` | string | ❌ | `""` | Max 5000 chars |

**Response 200:**
```json
{
  "memorandum": "بسم الله الرحمن الرحيم\nمذكرة دفاع...",
  "confidence_score": 0.69,
  "confidence_factors": { ... },
  "sources": [ ... ],
  "warnings": [],
  "conflicts_detected": false,
  "latency_ms": 5100.7,
  "model": "qwen/qwen-2.5-72b-instruct"
}
```

---

### 3.10  `POST /api/v1/ingest`  — Upload Documents

Multipart file upload. Adds new legal documents to the running index in-process. Accepts `.txt`, `.pdf`, `.docx`.

**Request:** `multipart/form-data` with one or more `files`.

**Response 200:**
```json
{
  "status": "ok",
  "files_processed": 3,
  "chunks_created": 47,
  "vectors_added": 47,
  "errors": [],
  "duration_s": 12.4
}
```

`status` is `"ok"`, `"partial"` (some files failed; see `errors[]`), `"no_data"` (no valid content), or `"error"`.

---

### 3.11  `POST /api/v1/ingest/scan`  — Scan Inbox

Server-side trigger for the watch-folder pipeline. Picks up new files dropped into the server's configured `INGEST_INBOX_DIR`, processes them, merges into FAISS+BM25 on disk, then hot-reloads the running retrieval service. **Idempotent** — uses `data/.ingested_files.json` to skip already-processed files.

No request body.

**Response 200:** Same shape as `/ingest` above. Use this when your deployment drops files into a known directory and wants to avoid HTTP file upload.

---

## 4. Error Reference

| HTTP | When | Body |
|---|---|---|
| 200 | Success | endpoint-specific |
| 404 | No retrieved docs / no such session | `{"detail": "..."}` (FastAPI default) |
| 422 | Request validation failed | `{"detail": [{...}, ...]}` (FastAPI default) |
| 429 | (Future) rate-limited | n/a |
| 500 | Internal error / LLM unavailable | `{"detail": "..."}` |

LLM-side failures are returned in the `answer`/`analysis`/`memorandum` field as the literal string `"[ERROR: LLM Service Unavailable]"` rather than HTTP 500, so the client can still see partial result and surface a friendly message.

---

## 5. C# Integration

### 5.1  DTOs

```csharp
public record SourceInfo(
    string Filename,
    string Source,
    string DocType,
    string LegalCategory,
    string LegalTopic,
    string? Article,
    List<string> ReferencedArticles,
    int? Page,
    double RetrievalScore,
    double RerankScore
);

public record ConfidenceFactors(
    double RerankSignal,
    int SourceCount,
    string ArticleValidation,   // "passed" | "failed" | "not_applicable"
    bool TopicMatch
);

public record QAResponse(
    string Answer,
    double ConfidenceScore,
    ConfidenceFactors ConfidenceFactors,
    List<SourceInfo> Sources,
    List<string> Warnings,
    bool ConflictsDetected,
    double LatencyMs,
    double RetrievalMs,
    string Model
);

public record ChatResponse(
    string Answer,
    string SessionId,
    double ConfidenceScore,
    ConfidenceFactors ConfidenceFactors,
    List<SourceInfo> Sources,
    List<string> Warnings,
    bool ConflictsDetected,
    int TurnCount,
    bool WasCompacted,
    double LatencyMs,
    string Model
);

public record WeaknessResponse(
    string Analysis,
    double ConfidenceScore,
    ConfidenceFactors ConfidenceFactors,
    List<SourceInfo> Sources,
    List<string> Warnings,
    bool ConflictsDetected,
    double LatencyMs,
    string Model
);

public record DefenseResponse(
    string Memorandum,
    double ConfidenceScore,
    ConfidenceFactors ConfidenceFactors,
    List<SourceInfo> Sources,
    List<string> Warnings,
    bool ConflictsDetected,
    double LatencyMs,
    string Model
);

public record SummarizeResponse(
    string Summary,
    int InputLength,
    double LatencyMs,
    string Model
);

public record HealthResponse(
    string Status,
    int Vectors,
    int Chunks,
    string Model,
    string EmbeddingModel,
    bool RerankerLoaded,
    string RerankerModel,
    List<string> LoadErrors
);

public record IngestResponse(
    string Status,
    int FilesProcessed,
    int ChunksCreated,
    int VectorsAdded,
    List<string> Errors,
    double DurationS
);
```

### 5.2  Client

```csharp
using System.Net.Http.Json;
using System.Text.Encodings.Web;
using System.Text.Json;

public sealed class LegalAIClient
{
    private static readonly JsonSerializerOptions Json = new() {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };
    private readonly HttpClient _http;

    public LegalAIClient(string baseUrl = "http://localhost:8000")
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromMinutes(5) };
    }

    public async Task<HealthResponse?> HealthAsync(CancellationToken ct = default) =>
        await _http.GetFromJsonAsync<HealthResponse>("/api/v1/health", Json, ct);

    public async Task<QAResponse?> AskAsync(string question, int k = 7, string promptStyle = "restrictive", CancellationToken ct = default)
    {
        var resp = await _http.PostAsJsonAsync("/api/v1/qa", new { question, k, prompt_style = promptStyle }, Json, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<QAResponse>(Json, ct);
    }

    public async Task<ChatResponse?> ChatAsync(string message, string sessionId, int k = 7, CancellationToken ct = default)
    {
        var resp = await _http.PostAsJsonAsync("/api/v1/chat", new { message, session_id = sessionId, k }, Json, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<ChatResponse>(Json, ct);
    }

    public async Task<SummarizeResponse?> SummarizeAsync(string text, CancellationToken ct = default)
    {
        var resp = await _http.PostAsJsonAsync("/api/v1/summarize", new { text }, Json, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<SummarizeResponse>(Json, ct);
    }

    public async Task<WeaknessResponse?> WeaknessAsync(string caseFacts, CancellationToken ct = default)
    {
        var resp = await _http.PostAsJsonAsync("/api/v1/weakness", new { case_facts = caseFacts }, Json, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<WeaknessResponse>(Json, ct);
    }

    public async Task<DefenseResponse?> DefenseAsync(string caseFacts, string weaknesses = "", CancellationToken ct = default)
    {
        var resp = await _http.PostAsJsonAsync("/api/v1/defense", new { case_facts = caseFacts, weaknesses }, Json, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<DefenseResponse>(Json, ct);
    }

    public async Task<IngestResponse?> ScanInboxAsync(CancellationToken ct = default)
    {
        var resp = await _http.PostAsync("/api/v1/ingest/scan", null, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<IngestResponse>(Json, ct);
    }
}
```

### 5.3  Streaming Chat (SSE)

```csharp
using System.Net.Http.Headers;
using System.Text.Json;

public async IAsyncEnumerable<string> StreamChatAsync(
    string message,
    string sessionId,
    int k = 7,
    [EnumeratorCancellation] CancellationToken ct = default)
{
    using var req = new HttpRequestMessage(HttpMethod.Post, "/api/v1/chat/stream")
    {
        Content = JsonContent.Create(new { message, session_id = sessionId, k })
    };
    req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

    using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
    resp.EnsureSuccessStatusCode();
    using var stream = await resp.Content.ReadAsStreamAsync(ct);
    using var reader = new StreamReader(stream);

    while (!reader.EndOfStream && !ct.IsCancellationRequested)
    {
        var line = await reader.ReadLineAsync(ct);
        if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data:")) continue;

        var json = line["data:".Length..].Trim();
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("error", out var err))
            throw new InvalidOperationException(err.GetString());

        if (root.GetProperty("done").GetBoolean())
        {
            // Final event: parse confidence_score, sources, warnings, etc. for your UI here.
            yield break;
        }

        yield return root.GetProperty("chunk").GetString() ?? string.Empty;
    }
}
```

### 5.4  DI registration (ASP.NET Core)

```csharp
builder.Services.AddHttpClient<LegalAIClient>((sp, http) =>
{
    http.BaseAddress = new Uri(builder.Configuration["LegalAI:BaseUrl"] ?? "http://localhost:8000");
    http.Timeout = TimeSpan.FromMinutes(5);  // /qa cold-call can take 60-120s with current latency
});
```

---

## 6. Operational Notes

- **Cold-call latency** is significant on the current dev hardware (CPU-only, no GPU). The first `/qa` after server start can take 60–300 s while models warm up. Subsequent calls drop to 5–15 s. Plan timeouts accordingly (`HttpClient.Timeout` ≥ 5 min for now).
- **Streaming hides latency**: prefer `/chat/stream` over `/chat` for any human-facing UI.
- **Session compaction** triggers automatically at 6 turns. Clients don't need to manage it.
- **Ingest concurrency**: `/ingest` and `/ingest/scan` mutate global state; serialize them server-side. (Currently no lock — first one wins on race; refactor on the roadmap.)
- **Health gating**: poll `/health` until `status: "ok"` before sending real traffic after a server restart.
- **Auth**: none in dev. Add `X-API-Key` middleware or upstream gateway auth before production.

---

## 7. Examples — Full Round-Trip

### Q&A
```bash
curl -s -X POST http://localhost:8000/api/v1/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هي عقوبة السرقة بالإكراه؟","k":7}' | jq
```

### Streaming chat
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message":"ما عقوبة السرقة بالإكراه؟","session_id":"demo-1"}'
```

### Trigger inbox ingest
```bash
curl -s -X POST http://localhost:8000/api/v1/ingest/scan | jq
```

---

## 8. Out of Scope (Phase 2+)

These are planned but **not yet shipped**:
- OCR fallback for scanned PDFs
- Page/paragraph preservation in `SourceInfo.page`
- Real conflict detection (`conflicts_detected` is currently always `false`)
- Agentic clarification-loop endpoint
- Neo4j GraphRAG retrieval channel
- Authentication / rate-limiting middleware

Track progress against these in the project's `CLAUDE.md` and Phase-roadmap memos.
