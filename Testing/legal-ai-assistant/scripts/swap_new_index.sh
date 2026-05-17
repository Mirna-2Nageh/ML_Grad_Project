#!/usr/bin/env bash
# Swap the freshly-built index in `data.new/` into place as `data/`,
# preserving the old `data/` as `data.legacy/` and carrying over expert_rules.json.
# Run after `scripts/build_index.py --output-dir data.new` completes.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f data.new/chunks.pkl ] || [ ! -f data.new/bm25.pkl ] || [ ! -f data.new/tokenized_corpus.pkl ] || [ ! -d data.new/faiss_index ]; then
    echo "❌ data.new/ is incomplete — refusing to swap. Expected chunks.pkl, bm25.pkl, tokenized_corpus.pkl, faiss_index/."
    exit 1
fi

echo "📦 Stopping backend on :8000 (if running)..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

if [ -d data.legacy ]; then
    echo "🧹 Removing prior data.legacy/ (already replaced once)..."
    rm -rf data.legacy
fi

echo "♻️  data/ → data.legacy/"
mv data data.legacy

echo "♻️  data.new/ → data/"
mv data.new data

if [ -f data.legacy/expert_rules.json ] && [ ! -f data/expert_rules.json ]; then
    echo "📋 Carrying expert_rules.json forward"
    cp data.legacy/expert_rules.json data/
fi

echo ""
echo "✅ Swap complete. Index files now in place:"
ls -la data/ | head -10
echo ""
echo "▶️  Start the backend with:"
echo "   nohup ./LaW/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &"
echo ""
echo "↩️  To roll back: rm -rf data && mv data.legacy data && restart the backend."
