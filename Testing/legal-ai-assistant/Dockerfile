# Conan — Egyptian Criminal-Law RAG API
#
# Build context = this directory (legal-ai-assistant/). The 5.8GB `LaW/` venv and the
# ~0.7GB `data/` index are EXCLUDED via .dockerignore — the image is CODE ONLY.
#
# At RUNTIME you must provide (see DEPLOY.md):
#   1. the prebuilt index, mounted at /app/data
#      (faiss_index/, bm25.pkl, chunks.pkl, tokenized_corpus.pkl, expert_rules.json)
#   2. API keys via --env-file .env
#   3. (recommended) a persistent HF cache volume at /root/.cache/huggingface so the
#      ~2.7GB embedding+reranker models aren't re-downloaded on every container start.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# poppler-utils → clean Arabic PDF extraction (pdftotext); falls back to PyPDF2 if absent.
# (All Python wheels are prebuilt for linux x86_64/aarch64, so no compiler is needed. If a
#  wheel ever fails to build on an exotic host, add: build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first — the default PyPI `torch` pulls multi-GB CUDA libs this
# CPU service can't use. This wheel satisfies the `torch>=2.1.0` line in requirements.txt.
RUN pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# App code only (no data/, no .env, no venv — see .dockerignore)
COPY config.py .
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY streamlit_app.py .

EXPOSE 8000

# First boot downloads the models (~2.7GB) and loads the 47k-chunk index — hence the long
# start period before health failures count.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=8).status==200 else 1)" || exit 1

# One worker: the index + models load into RAM once; extra workers would multiply memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
