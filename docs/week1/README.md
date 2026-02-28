# Week 1 — Developer Knowledge Copilot

## What We Built

A fully working **Retrieval-Augmented Generation (RAG)** backend with a React Native mobile app frontend.

```
User (mobile) → POST /api/v1/query → FastAPI → FAISS → SQLite → LLM → Response
```

---

## Project Structure

```
developer-knowledge-copilot/
├── backend/
│   ├── api/routes.py          ← HTTP endpoints (/query, /health, /documents)
│   ├── core/config.py         ← All config from .env (pydantic-settings)
│   ├── db/models.py           ← SQLite: documents + chunks tables
│   ├── ingestion/
│   │   ├── chunker.py         ← Split long docs into 384-token overlapping windows
│   │   └── embedder.py        ← BGE-small-en-v1.5 → 384-dim vectors
│   ├── retrieval/vector_store.py  ← FAISS IndexFlatIP: add, search, save, load
│   ├── generation/llm.py      ← Prompt builder + async LLM API call
│   └── main.py                ← App entry point, startup, CORS, route registration
├── mobile/
│   ├── services/api.ts        ← Typed fetch calls to backend
│   └── app/search.tsx         ← Search screen with citations + confidence badge
├── scripts/ingest_docs.py     ← CLI: read .md files → chunk → embed → index
├── data/sample_docs/          ← 3 seed documents (FastAPI, Docker, Vector DBs)
└── docs/week1/                ← This folder — interview prep docs
```

---

## How to Run Locally

### 1. Set up Python environment

```bash
cd c:\Users\hp\Documents\NITK\PROJECT
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Create .env file

```bash
copy .env.example .env
# Edit .env and add your LLM_API_KEY from openrouter.ai (free signup)
```

### 3. Ingest sample documents

```bash
python scripts/ingest_docs.py --source data/sample_docs
```

You should see:
```
📂 Loading documents from: data/sample_docs
  [Read] fastapi_getting_started.md
  [Read] docker_reference.md
  [Read] vector_databases_guide.md
✅ Loaded 3 document(s)
✂️  Chunking documents...
🔢 Embedding chunks...
💾 Saving vector index...
🎉 Ingestion complete!
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` — interactive API docs appear automatically.

### 5. Test a query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I add CORS to FastAPI?"}'
```

### 6. Run tests

```bash
pytest backend/tests/test_week1.py -v
```

### 7. Run mobile app

```bash
cd mobile
npx create-expo-app@latest .   # Only first time
npm start
# Scan QR code with Expo Go app on your phone
```

---

## Architecture Diagram (Week 1)

```
┌───────────────────────────────────────────────────────────────────┐
│  INGESTION (one-time)                                             │
│                                                                   │
│  .md files → Chunker → BGE Embedder → FAISS Index + SQLite       │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│  QUERY (real-time)                                                │
│                                                                   │
│  Mobile App                                                       │
│     │                                                             │
│     ▼                                                             │
│  POST /api/v1/query                                               │
│     │                                                             │
│     ├──► Embed query (BGE)                                        │
│     │         │                                                   │
│     ├──► FAISS search → top-5 chunk IDs                          │
│     │         │                                                   │
│     ├──► SQLite lookup → chunk text + metadata                   │
│     │         │                                                   │
│     ├──► Build prompt = system + context chunks + question        │
│     │         │                                                   │
│     ├──► LLM API → answer + tokens used                          │
│     │         │                                                   │
│     └──► Return JSON: answer + citations + confidence + latency   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Read These Docs (in order)

| File | Topic |
|------|-------|
| [`01_chunking.md`](./01_chunking.md) | Why we chunk, how overlap works |
| [`02_embeddings.md`](./02_embeddings.md) | What embeddings are, BGE model |
| [`03_faiss.md`](./03_faiss.md) | FAISS internals, index types |
| [`04_rag_pipeline.md`](./04_rag_pipeline.md) | Full RAG flow end-to-end |
| [`05_fastapi_structure.md`](./05_fastapi_structure.md) | API design decisions |
| [`06_interview_qa.md`](./06_interview_qa.md) | 15 interview questions + answers |

---

## Week 1 Commit History (Target)

```
feat: initialise project structure and requirements
feat: add pydantic-settings config and .env.example
feat: implement SQLite schema for documents and chunks
feat: add token-aware chunker with overlap
feat: add BGE-small embedding wrapper with query prefix
feat: add FAISS IndexFlatIP vector store with save/load
feat: add async LLM generation with structured prompt
feat: add FastAPI routes: /query, /health, /documents
feat: add ingest_docs.py CLI script
feat: add sample documents for local testing
test: add unit tests for chunker, embedder, vector store, and API
feat: add Dockerfile with pre-baked embedding model
feat: add mobile search screen with citation cards
docs: add week1 interview prep documentation
```
