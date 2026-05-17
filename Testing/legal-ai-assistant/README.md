# ⚖️ Legal AI Assistant — Egyptian Criminal Law

RAG-based AI system for Egyptian Criminal Law analysis, powered by **Qwen (via OpenRouter)** and **BGE-M3** embeddings.

## ✨ Features

| Feature | Endpoint | Description |
|---|---|---|
| **Q&A** | `POST /api/v1/qa` | Ask legal questions with article citations |
| **Summarize** | `POST /api/v1/summarize` | Summarize legal documents |
| **Weakness** | `POST /api/v1/weakness` | Detect prosecution case weaknesses |
| **Defense** | `POST /api/v1/defense` | Generate defense memorandums |
| **Eval UI** | Streamlit app | Interactive testing + metrics dashboard |

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  .NET App   │────▶│  FastAPI      │────▶│  OpenRouter     │
│  (Frontend) │     │  /api/v1/*    │     │  (Qwen 3-8B)    │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────┴───────┐
                    │  Retrieval   │
                    │  FAISS+BM25  │
                    │  (BGE-M3)    │
                    └──────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd legal-ai-assistant
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env → add your OPENROUTER_API_KEY
# Get free key at: https://openrouter.ai/keys
```

### 3. Build Index (one-time)

```bash
python scripts/build_index.py --dataset-dir ../final_total_dataset
```

> ⚠️ This processes the full dataset and builds FAISS + BM25 indices.  
> Takes ~10-30 minutes depending on hardware. Requires ~8GB RAM.

### 4. Start API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Test with Streamlit UI

```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
legal-ai-assistant/
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── models.py              # API request/response schemas
│   ├── routers/               # Endpoint handlers
│   │   ├── qa.py              # POST /api/v1/qa
│   │   ├── summarize.py       # POST /api/v1/summarize
│   │   ├── weakness.py        # POST /api/v1/weakness
│   │   ├── defense.py         # POST /api/v1/defense
│   │   └── health.py          # GET  /api/v1/health
│   ├── services/
│   │   ├── llm.py             # OpenRouter client
│   │   ├── retrieval.py       # FAISS + BM25 hybrid
│   │   └── preprocessing.py   # Arabic text processing
│   └── core/
│       └── prompts.py         # Prompt templates
├── scripts/
│   └── build_index.py         # Index builder
├── streamlit_app.py           # Evaluation UI
├── api_contract.md            # .NET integration guide
├── Dockerfile                 # Docker deployment
└── data/                      # Generated indices
```

## 🔗 .NET Integration

See [`api_contract.md`](api_contract.md) for:
- Full endpoint documentation
- Request/response schemas
- C# `HttpClient` code examples
- Response DTOs

## 🐳 Docker

```bash
# Build index first, then:
docker build -t legal-ai .
docker run -p 8000:8000 --env-file .env legal-ai
```

## 📊 API Docs (auto-generated)

When running, visit:
- **Swagger:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## ⚙️ Configuration

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen/qwen3-8b:free` | OpenRouter model |
| `EMBED_MODEL_NAME` | `BAAI/bge-m3` | Embedding model (1024-dim) |
| `RETRIEVAL_K` | 7 | Context documents returned |
| `TEMPERATURES["qa"]` | 0.1 | Q&A temperature |
| `TEMPERATURES["defense"]` | 0.5 | Defense memo temperature |

## 📝 License

Academic use — Egyptian Criminal Law AI Research Project.
