# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile for the Lumen backend
#
# Build image:  docker build -t dev-copilot:latest .
# Run locally:  docker run -p 8000:8000 --env-file .env dev-copilot:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Base setup ───────────────────────────────────────────────────────
# ── Stage 1: Base setup ───────────────────────────────────────────────────────
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Prevents Python from writing .pyc files (keeps image smaller)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: Dependency installation ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Copy application ─────────────────────────────────────────────────
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create the data directory for FAISS index persistence
RUN mkdir -p /app/data

# Make startup script executable and fix line endings for Linux
RUN sed -i 's/\r$//' /app/backend/start.sh && chmod +x /app/backend/start.sh

# ── Runtime configuration ────────────────────────────────────────────────────
EXPOSE 8001 10000

# Start the server using the optimized startup script
CMD ["/app/backend/start.sh"]
