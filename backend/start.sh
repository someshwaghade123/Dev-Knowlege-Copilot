#!/bin/bash
# backend/start.sh
# ─────────────────────────────────────────────────────────────────────────────
# This script ensures models are downloaded BEFORE the server binds to a port.
# This prevents 502 errors on the first request.
# ─────────────────────────────────────────────────────────────────────────────

echo "[Startup] Downloading/Verifying models..."
python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); TextCrossEncoder(model_name='BAAI/bge-reranker-base')"

echo "[Startup] Starting Uvicorn server on port ${PORT:-8001}..."
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8001}
