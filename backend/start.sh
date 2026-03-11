#!/bin/bash
# backend/start.sh
# ─────────────────────────────────────────────────────────────────────────────
# This script ensures models are downloaded BEFORE the server binds to a port.
# This prevents 502 errors on the first request.
# ─────────────────────────────────────────────────────────────────────────────

echo "[Startup] System Info:"
python --version
pip show fastembed | grep Version

# Limit threads to save memory in onnxruntime
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[Startup] Downloading/Verifying Embedding model..."
python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

echo "[Startup] Downloading/Verifying Reranker model..."
python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

echo "[Startup] All models ready on disk."

echo "[Startup] Starting Uvicorn server on port ${PORT:-8001}..."
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8001}
