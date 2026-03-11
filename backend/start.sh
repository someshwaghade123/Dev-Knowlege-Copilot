#!/bin/bash
# backend/start.sh
# ─────────────────────────────────────────────────────────────────────────────
# This script ensures models are downloaded BEFORE the server binds to a port.
# This prevents 502 errors on the first request.
# ─────────────────────────────────────────────────────────────────────────────

echo "[Startup] System Info:"
python --version
pip show fastembed | grep Version

echo "[Startup] Downloading/Verifying models..."
# Robust check for TextCrossEncoder
python -c "
import fastembed
print(f'FastEmbed version: {fastembed.__version__ if hasattr(fastembed, \"__version__\") else \"unknown\"}')
try:
    from fastembed import TextEmbedding, TextCrossEncoder
    print('Import from top-level successful')
except ImportError:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    print('Import from sub-module successful')

TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
TextCrossEncoder(model_name='BAAI/bge-reranker-base')
"

echo "[Startup] Starting Uvicorn server on port ${PORT:-8001}..."
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8001}
