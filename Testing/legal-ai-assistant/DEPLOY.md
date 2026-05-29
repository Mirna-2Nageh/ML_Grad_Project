# Deploying Conan (RAG API)

How to run the Conan FastAPI service in a container, off your laptop. The image is **code
only**; the index and API keys are supplied at runtime.

---

## 1. What it needs (resource reality)

| Resource | Requirement | Why |
|---|---|---|
| **RAM** | **~6–8 GB** (≈3–4 GB if `USE_RERANKER=False`) | embedding model + cross-encoder reranker (~2.3 GB) + index in memory |
| **CPU** | 2+ cores, **CPU-only is fine** | no GPU used; retrieval+rerank is the latency (~16–26 s/request) |
| **Disk** | ~4 GB | image (~1.5 GB) + models (~2.7 GB, downloaded on first boot) + index (~0.7 GB) |
| **Network** | outbound HTTPS | LLM calls to Groq/Cerebras/Gemini/OpenRouter |

> 512 MB / 1 GB "free tiers" (Render free, etc.) **cannot run this** — it OOMs on startup.
> See §7 for hosts that fit.

You must provide **two things the image does not contain**:
1. **The prebuilt index** → mounted at `/app/data` (`faiss_index/`, `bm25.pkl`, `chunks.pkl`, `tokenized_corpus.pkl`, `expert_rules.json`). The server refuses to start without it.
2. **API keys** → an `.env` file (see `.env.example`).

---

## 2. Quick start (docker compose)

From `legal-ai-assistant/`, with `.env` and `data/` present:

```bash
cp .env.example .env          # then fill in GROQ_API_KEY (+ optionally CEREBRAS / GOOGLE / OPENROUTER)
docker compose up -d --build
docker compose logs -f conan  # watch first boot (model download + index load)
```

First boot takes a few minutes (downloads ~2.7 GB of models). When ready:

```bash
curl http://localhost:8000/api/v1/health        # {"status":"ok","vectors":47028,...}
```

---

## 3. Plain `docker run` (no compose)

```bash
docker build -t conan-legal-rag:latest .

docker run -d --name conan -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -v conan_hf_cache:/root/.cache/huggingface \
  --restart unless-stopped \
  conan-legal-rag:latest
```

- `--env-file .env` injects the keys as env vars (no `.env` baked into the image).
- `-v .../data:/app/data` mounts the index (and is where sessions + the answer cache are written).
- `-v conan_hf_cache:...` persists the models so they download only once.

---

## 4. Getting the index onto the host

The image has **no index** (it's ~0.7 GB and not in git). Two ways to provide it:

**A. Copy the prebuilt index from your dev machine (fast, recommended).**
```bash
# from your laptop, to a server you can ssh to:
rsync -avz legal-ai-assistant/data/  user@host:/srv/conan/data/
# then on the host, mount /srv/conan/data at /app/data (compose: change ./data to /srv/conan/data)
```
Copy `faiss_index/`, `bm25.pkl`, `chunks.pkl`, `tokenized_corpus.pkl`, `expert_rules.json`.
(You can skip `sessions/`, `inbox/`, `answer_cache.json` — they're runtime-created.)

**B. Rebuild it on the host (slow — 3–7 h CPU; only if you can't copy).**
Requires the dataset too. Inside the container or a venv:
```bash
python scripts/build_index.py            # processes ../final_total_dataset → data/
```

> The models (embedding + reranker) are **not** your problem — they download automatically
> from HuggingFace on first boot into the mounted HF cache.

---

## 5. Verify the deployment

```bash
# Shape/contract check against the running container (free, no LLM):
python scripts/contract_selftest.py --base-url http://localhost:8000/api/v1
# Full sweep incl. LLM endpoints (a 503 is reported SKIP, not a failure):
python scripts/contract_selftest.py --full --base-url http://localhost:8000/api/v1
```

---

## 6. Updating the index later

The index files are written together and loaded together. To swap in a new build:
1. Replace the files in the mounted `data/` dir.
2. **Delete `data/answer_cache.json`** (cached answers belong to the old index).
3. Restart: `docker compose restart conan` (or hit `POST /api/v1/ingest/scan` for in-process hot-reload after dropping files in the inbox).

---

## 7. Where to host it (free / student-friendly)

| Platform | Fit | Notes |
|---|---|---|
| **Cloudflare Tunnel / ngrok** (laptop) | Best for **integration now** | Free public URL → `localhost:8000`. Zero migration. Laptop must stay on. |
| **Oracle Cloud "Always Free"** | Best free **always-on** | ARM VM up to 4 cores / 24 GB RAM, free forever. ARM wheels install fine. |
| **Hugging Face Spaces** (free CPU) | Easy demo | 2 vCPU / 16 GB RAM, Docker SDK. Sleeps when idle (cold start reloads models). |
| **DigitalOcean / Azure** via **GitHub Student Pack** | Production-like | $200 / $100 credits → a 4–8 GB droplet/VM for months. |

Expose the tunnel for the backend team in minutes:
```bash
cloudflared tunnel --url http://localhost:8000      # prints a public https URL
# or:  ngrok http 8000
```

---

## 8. Before exposing to the internet (production hardening)

- **Auth:** there is none today. Put Conan **behind your .NET backend / an API gateway**; don't expose it directly.
- **CORS:** currently `allow_origins=["*"]` in `app/main.py` — restrict it to your frontend origin.
- **HTTPS:** terminate TLS at nginx / Caddy / the platform's load balancer in front of the container.
- **Secrets:** keep `.env` out of the image and out of git (already handled); use the platform's secret manager in prod.
- **Budget:** add `CEREBRAS_API_KEY` and extra `GOOGLE_API_KEYS` (see `.env.example`) so a single free tier running out doesn't 503 you. The answer cache also cuts repeat-question cost to zero.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Container exits on boot, logs mention index/`data` | `data/` not mounted or missing files — see §4. |
| `OOMKilled` / killed during startup | Not enough RAM — use a ≥8 GB host, or set `USE_RERANKER=False` in `.env`. |
| Every request → `503` | All LLM providers out of budget — add Cerebras/Gemini keys (§8). |
| First request very slow / healthcheck "starting" for minutes | Normal first boot: models downloading. The healthcheck has a 300 s start period. |
| Build sends gigabytes of context | `.dockerignore` missing/edited — it must exclude `LaW/` and `data/`. |
