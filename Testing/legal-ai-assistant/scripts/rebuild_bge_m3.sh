#!/usr/bin/env bash
# Full-corpus BGE-M3 index rebuild into a STAGING dir (data.new/), leaving the live
# index in data/ untouched so the running server keeps serving. After this finishes,
# run scripts/swap_new_index.sh to put it live, then flip the embedding lines in .env.
#
# Why staging + explicit env overrides: the embedding model in .env and the on-disk
# FAISS index must agree on dimensionality, so we keep .env on the OLD model until the
# swap and pass the bge-m3 settings inline here instead of relying on .env.
#
# Heavy + long: bge-m3 (1024-dim) on CPU over the full corpus is an overnight job and
# will compete with the live server for CPU. Run it when query traffic is low.
#
# Usage:  nohup ./scripts/rebuild_bge_m3.sh > rebuild_bge_m3.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="$([ -x LaW/bin/python ] && echo LaW/bin/python || echo python3)"

echo "▶️  BGE-M3 rebuild starting $(date)"
echo "    python: $PYBIN   output: data.new/"

EMBED_MODEL=BAAI/bge-m3 \
EMBED_DIMENSIONS=1024 \
EMBED_MAX_SEQ_LENGTH=1024 \
EMBED_BATCH_SIZE=8 \
USE_REMOTE_EMBEDDINGS=False \
EMBED_PROVIDER=local \
  "$PYBIN" scripts/build_index.py \
    --output-dir data.new \
    --embed-model BAAI/bge-m3

echo "✅ BGE-M3 rebuild finished $(date)"
echo ""
echo "Next:"
echo "  1) python scripts/eval_harness.py --mode retrieval --tag bge_m3   # build the new scorecard"
echo "     (point DATA_DIR at data.new first, or run after the swap)"
echo "  2) ./scripts/swap_new_index.sh                                    # data.new/ → data/"
echo "  3) set in .env:  EMBED_MODEL=BAAI/bge-m3  EMBED_DIMENSIONS=1024  EMBED_MAX_SEQ_LENGTH=1024  EMBED_BATCH_SIZE=8"
echo "  4) restart the backend, then: python scripts/eval_harness.py --compare baseline_minilm bge_m3"
