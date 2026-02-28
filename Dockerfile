# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile for the Developer Knowledge Copilot backend
#
# Build image:  docker build -t dev-copilot:latest .
# Run locally:  docker run -p 8000:8000 --env-file .env dev-copilot:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Base setup ───────────────────────────────────────────────────────
FROM python:3.11-slim as base

# Prevents Python from writing .pyc files (keeps image smaller)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: Dependency installation ─────────────────────────────────────────
# Install dependencies BEFORE copying app code.
# This way, the layer is cached as long as requirements.txt doesn't change.
# (One of the most important Docker optimisation patterns)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
# (avoids slow cold-start on first request)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ── Stage 3: Copy application ─────────────────────────────────────────────────
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create the data directory for FAISS index persistence
RUN mkdir -p /app/data

# ── Runtime configuration ────────────────────────────────────────────────────
EXPOSE 8000

# Health check — Docker will restart the container if this fails 3 times
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
