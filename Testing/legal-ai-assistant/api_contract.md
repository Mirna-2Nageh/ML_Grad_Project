# 🔗 API Contract — Legal AI Assistant

> **For:** .NET Backend Team  
> **Version:** 1.0.0  
> **Base URL:** `http://<server>:8000/api/v1`  

---

## Authentication

Currently: **None** (add API key middleware for production).

## Response Format

All responses are JSON. Arabic text is UTF-8 encoded.  
All error responses follow: `{"error": "message", "detail": "info"}`

---

## Endpoints

### 1. Health Check

```
GET /api/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "vectors": 45230,
  "chunks": 45230,
  "model": "qwen/qwen3-8b:free",
  "embedding_model": "BAAI/bge-m3"
}
```

---

### 2. Legal Q&A

```
POST /api/v1/qa
Content-Type: application/json
```

**Request:**
```json
{
  "question": "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
  "k": 7,
  "prompt_style": "restrictive"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | ✅ | — | Legal question in Arabic (min 3 chars) |
| `k` | int | ❌ | 7 | Number of context documents (1-20) |
| `prompt_style` | string | ❌ | "restrictive" | "standard" or "restrictive" |

**Response:**
```json
{
  "answer": "وفقاً للمادة 314 من قانون العقوبات المصري...",
  "sources": [
    {
      "filename": "penal_code_part3.txt",
      "doc_type": "penal_code",
      "legal_category": "general"
    }
  ],
  "latency_ms": 2450.3,
  "retrieval_ms": 120.5,
  "model": "qwen/qwen3-8b:free"
}
```

---

### 3. Text Summarization

```
POST /api/v1/summarize
Content-Type: application/json
```

**Request:**
```json
{
  "text": "المادة الأولى: يعاقب بالحبس كل من ارتكب جريمة..."
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✅ | — | Legal text to summarize (min 50 chars) |

**Response:**
```json
{
  "summary": "• المادة 1: عقوبة الحبس لجريمة...",
  "input_length": 1523,
  "latency_ms": 1890.2,
  "model": "qwen/qwen3-8b:free"
}
```

---

### 4. Weakness Detection

```
POST /api/v1/weakness
Content-Type: application/json
```

**Request:**
```json
{
  "case_facts": "المتهم متهم بالسرقة بالإكراه ليلاً. تم القبض عليه بناءً على بلاغ مجهول."
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `case_facts` | string | ✅ | — | Case facts in Arabic (min 20 chars) |

**Response:**
```json
{
  "analysis": "1. المخالفات الإجرائية: القبض بناءً على بلاغ مجهول...",
  "sources": [ ... ],
  "latency_ms": 3200.1,
  "model": "qwen/qwen3-8b:free"
}
```

---

### 5. Streaming Chat (SSE)

```
POST /api/v1/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

Same request body as `/api/v1/chat` (uses session history + retrieval + evidence validation + confidence). Returns Server-Sent Events as Qwen generates tokens.

**Request:**
```json
{
  "message": "ما عقوبة السرقة بالإكراه؟",
  "session_id": "user-123",
  "k": 7
}
```

**Event stream (UTF-8 JSON payloads):**

While generating — one event per token chunk:
```
data: {"chunk": "وفقاً", "done": false}

data: {"chunk": " للمادة", "done": false}

data: {"chunk": " 314...", "done": false}
```

Final event — confidence, sources, warnings, timing:
```
data: {
  "done": true,
  "session_id": "user-123",
  "confidence_score": 0.87,
  "confidence_factors": {
    "rerank_signal": 0.85,
    "source_count": 5,
    "article_validation": "passed",
    "topic_match": true
  },
  "sources": [
    {
      "filename": "...",
      "doc_type": "penal_code",
      "legal_topic": "تزوير",
      "article": "211",
      "referenced_articles": ["211"],
      "retrieval_score": 0.91,
      "rerank_score": 0.94,
      ...
    }
  ],
  "warnings": [],
  "conflicts_detected": false,
  "turn_count": 3,
  "latency_ms": 4820.1,
  "model": "qwen/qwen-2.5-72b-instruct"
}
```

On error:
```
data: {"error": "...", "done": true}
```

> **Note:** streaming runs **OpenRouter-only** (no Gemini fallback). Non-streaming `/chat` retains the Gemini→OpenRouter fallback chain. Sessions persist server-side (`SESSION_PERSIST=True`).

**C# example** using `System.Net.Http`:
```csharp
public async IAsyncEnumerable<string> StreamChatAsync(string message, string sessionId)
{
    using var req = new HttpRequestMessage(HttpMethod.Post, "/api/v1/chat/stream")
    {
        Content = JsonContent.Create(new { message, session_id = sessionId, k = 7 })
    };
    req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
    using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
    using var stream = await resp.Content.ReadAsStreamAsync();
    using var reader = new StreamReader(stream);
    while (!reader.EndOfStream)
    {
        var line = await reader.ReadLineAsync();
        if (line is null || !line.StartsWith("data:")) continue;
        var payload = JsonDocument.Parse(line.Substring(5).Trim()).RootElement;
        if (payload.GetProperty("done").GetBoolean()) break;
        yield return payload.GetProperty("chunk").GetString()!;
    }
}
```

---

### 6. Scan Inbox & Ingest

```
POST /api/v1/ingest/scan
```

No body. Triggers an in-process scan of the server's configured `INGEST_INBOX_DIR`. Any files not in the ingest registry are preprocessed, chunked, embedded, merged into FAISS+BM25 on disk, then the running retrieval service hot-reloads. Idempotent.

**Response:**
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

Use this when your pipeline drops new legal docs into the server's inbox directory and you want them visible to search without restarting. Alternative: run `python scripts/watch_ingest.py` on a cron / systemd timer instead of hitting this endpoint.

---

### 7. Defense Memo

```
POST /api/v1/defense
Content-Type: application/json
```

**Request:**
```json
{
  "case_facts": "المتهم متهم بالسرقة بالإكراه.",
  "weaknesses": "عدم وجود شهود عيان"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `case_facts` | string | ✅ | — | Case facts in Arabic (min 20 chars) |
| `weaknesses` | string | ❌ | "" | Previously identified weaknesses |

**Response:**
```json
{
  "memorandum": "بسم الله الرحمن الرحيم\nمذكرة دفاع...",
  "sources": [ ... ],
  "latency_ms": 5100.7,
  "model": "qwen/qwen3-8b:free"
}
```

---

## C# Integration Example

```csharp
using System.Net.Http.Json;

public class LegalAIClient
{
    private readonly HttpClient _http;

    public LegalAIClient(string baseUrl = "http://localhost:8000")
    {
        _http = new HttpClient { BaseAddress = new Uri(baseUrl) };
    }

    // Q&A
    public async Task<QAResponse?> AskQuestionAsync(string question, int k = 7)
    {
        var response = await _http.PostAsJsonAsync("/api/v1/qa", new
        {
            question,
            k,
            prompt_style = "restrictive"
        });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<QAResponse>();
    }

    // Summarize
    public async Task<SummarizeResponse?> SummarizeAsync(string text)
    {
        var response = await _http.PostAsJsonAsync("/api/v1/summarize", new { text });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<SummarizeResponse>();
    }

    // Weakness Detection
    public async Task<WeaknessResponse?> DetectWeaknessAsync(string caseFacts)
    {
        var response = await _http.PostAsJsonAsync("/api/v1/weakness", new
        {
            case_facts = caseFacts
        });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<WeaknessResponse>();
    }

    // Defense Memo
    public async Task<DefenseResponse?> GenerateDefenseAsync(string caseFacts, string weaknesses = "")
    {
        var response = await _http.PostAsJsonAsync("/api/v1/defense", new
        {
            case_facts = caseFacts,
            weaknesses
        });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<DefenseResponse>();
    }
}

// ── Response DTOs ──
public record SourceInfo(string Filename, string DocType, string LegalCategory);
public record QAResponse(string Answer, List<SourceInfo> Sources, double LatencyMs, double RetrievalMs, string Model);
public record SummarizeResponse(string Summary, int InputLength, double LatencyMs, string Model);
public record WeaknessResponse(string Analysis, List<SourceInfo> Sources, double LatencyMs, string Model);
public record DefenseResponse(string Memorandum, List<SourceInfo> Sources, double LatencyMs, string Model);
```

> **⚠️ Note:** JSON property names use `snake_case`. Configure `JsonSerializerOptions` with `PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower` in .NET 8+, or use `[JsonPropertyName]` attributes.

---

## Error Codes

| HTTP Code | Meaning |
|---|---|
| 200 | Success |
| 404 | No relevant documents found |
| 422 | Validation error (check request body) |
| 429 | Rate limited (retry after backoff) |
| 500 | Internal server error |

---

## Interactive Docs

When the server is running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
