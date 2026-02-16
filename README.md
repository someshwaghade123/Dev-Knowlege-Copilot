# 🧠 Developer Knowledge Copilot

A production-grade AI system that indexes technical documentation and answers developer queries with citations, confidence scores, and latency tracking.

**Stack**: FastAPI · FAISS · BGE-small embeddings · SQLite · React Native (Expo)

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env and add your free API key from https://openrouter.ai
```

### 3. Ingest sample documents

```bash
python scripts/ingest_docs.py --source data/sample_docs
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive API documentation.

### 5. Run the mobile app

```bash
cd mobile
npx create-expo-app@latest .    # First time only
npm start
# Scan QR with Expo Go on your phone
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Liveness check + indexed vector count |
| `POST` | `/api/v1/query` | Ask a question, get answer + citations |
| `GET` | `/api/v1/documents` | List all indexed documents |

### POST /api/v1/query

```json
// Request
{
  "query": "How do I configure CORS in FastAPI?",
  "top_k": 5
}

// Response
{
  "answer": "Use CORSMiddleware from fastapi.middleware.cors...",
  "citations": [
    {
      "title": "FastAPI Getting Started",
      "source_url": null,
      "text_preview": "from fastapi.middleware.cors import CORSMiddleware..."
    }
  ],
  "confidence": "high",
  "latency_ms": 482,
  "tokens_used": 347
}
```

---

## Run Tests

```bash
pytest backend/tests/test_week1.py -v
```

---

## Docker

```bash
# Build
docker build -t dev-copilot:latest .

# Run
docker run -p 8000:8000 --env-file .env dev-copilot:latest
```

---

## Project Structure

```
├── backend/
│   ├── api/routes.py         ← HTTP endpoints
│   ├── core/config.py        ← Settings (pydantic-settings)
│   ├── db/models.py          ← SQLite schema
│   ├── ingestion/
│   │   ├── chunker.py        ← Token-aware chunking with overlap
│   │   └── embedder.py       ← BGE-small-en-v1.5 embeddings
│   ├── retrieval/
│   │   └── vector_store.py   ← FAISS IndexFlatIP
│   ├── generation/
│   │   └── llm.py            ← Prompt builder + async LLM call
│   └── main.py               ← App entry point
├── mobile/
│   ├── app/search.tsx        ← Search screen UI
│   └── services/api.ts       ← Backend API calls
├── scripts/ingest_docs.py    ← CLI ingestion pipeline
└── data/sample_docs/         ← Sample .md documents
```

---

---

## Roadmap

| Week | Focus |
|------|-------|
| ✅ 1 | Basic RAG — chunking, embeddings, FAISS, citations |
| 2 | Hybrid search (BM25 + vector) |
| 3 | Latency metrics + request logging |
| 4 | Confidence scoring + hallucination heuristics |
| 5 | Redis caching + benchmarks |
| 6 | Load testing (50 concurrent users) |
| 7 | Deploy to Render + Expo APK build |
