# 04 — Full RAG Pipeline Explained

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of asking an LLM to answer from memory (which leads to hallucinations),
we first **retrieve** relevant context from our document index, then **generate**
an answer grounded in that context.

---

## Without RAG vs With RAG

```
WITHOUT RAG:
  User: "How do I configure CORS in FastAPI?"
  LLM:  "You can use the middleware..." (might be wrong or outdated)

WITH RAG:
  1. Retrieve: Find 5 chunks from our FastAPI docs most relevant to "CORS FastAPI"
  2. Generate: LLM reads those chunks as context, writes a grounded answer
  3. Cite: Tell user which documents the answer came from
```

---

## The Full Pipeline — Step by Step

```
┌─────────────────────────────────────────────────────────┐
│                  INGESTION (one-time)                    │
│                                                          │
│  Document Files                                          │
│       │                                                  │
│       ▼                                                  │
│  chunker.py ──► 384-token overlapping windows           │
│       │                                                  │
│       ▼                                                  │
│  embedder.py ──► BGE-small ──► float32 vectors (384D)  │
│       │                                                  │
│       ▼                                                  │
│  vector_store.py ──► FAISS IndexFlatIP (on disk)        │
│  db/models.py ──► SQLite chunks table (metadata)        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   QUERY (real-time)                      │
│                                                          │
│  User query: "How to add CORS to FastAPI?"               │
│       │                                                  │
│  [1] EMBED QUERY                                         │
│       │  embed_query("Represent this sentence...")       │
│       │  → vector of shape (384,)                        │
│       │                                                  │
│  [2] VECTOR SEARCH                                       │
│       │  FAISS.search(query_vector, k=5)                 │
│       │  → [(faiss_id=42, score=0.91), ...]              │
│       │                                                  │
│  [3] METADATA LOOKUP                                     │
│       │  SQLite: SELECT text, title, url WHERE faiss_id IN (42,...)
│       │  → [{text: "...", title: "FastAPI Docs", ...}]   │
│       │                                                  │
│  [4] PROMPT ASSEMBLY                                     │
│       │  system: "Answer only from context..."           │
│       │  user:   "[Context 1] FastAPI CORS setup..."     │
│       │           "QUESTION: How to add CORS?"           │
│       │                                                  │
│  [5] LLM GENERATION                                      │
│       │  POST https://openrouter.ai/api/v1/chat/completions
│       │  → "Use CORSMiddleware from fastapi.middleware..."│
│       │                                                  │
│  [6] RESPONSE ASSEMBLY                                   │
│       │  {answer, citations, confidence, latency_ms}     │
│       │                                                  │
│  Mobile App displays answer + citation cards             │
└─────────────────────────────────────────────────────────┘
```

---

## Latency Breakdown (Typical)

| Stage | Latency |
|-------|---------|
| Query embedding (BGE-small, CPU) | 40–80ms |
| FAISS search (20k vectors, exact) | 5–15ms |
| SQLite lookup (5 rows) | < 5ms |
| LLM API call (Mistral 7B via OpenRouter) | 300–600ms |
| JSON serialisation + network | 10–20ms |
| **Total end-to-end** | **~420–700ms** |

The LLM API call dominates. This is why caching (Week 6) matters.

---

## Prompt Engineering — Our Template

```
SYSTEM:
You are a precise technical documentation assistant.
Answer the user's question using ONLY the provided context blocks.
If the answer is not in the context, say: 'I could not find a reliable answer.'
Always cite which Context number(s) you used.

USER:
[Context 1] Source: FastAPI Getting Started
URL: https://fastapi.tiangolo.com

FastAPI is a modern web framework for building APIs with Python...
CORSMiddleware can be added with:
  app.add_middleware(CORSMiddleware, allow_origins=["*"])

---

[Context 2] Source: Docker Reference
...

QUESTION: How do I add CORS to FastAPI?

Provide a clear, accurate answer with citation numbers.
```

---

## The "Stuffing" Strategy

We fit **all retrieved chunks into one prompt** (stuffing).

**Alternative: Map-Reduce**
- For very long documents, summarise each chunk separately, then combine
- More expensive (N+1 LLM calls), but handles larger contexts
- We'll consider this if answers become incoherent (future week)

---

## Confidence Scoring

We compute confidence from FAISS similarity scores:

```python
top_score = max(scores)   # Highest cosine similarity among retrieved chunks

if top_score > 0.80:  → "high"    (query closely matched our docs)
elif top_score > 0.60: → "medium"
else:                  → "low"    (no strong match — risky answer)
```

**Why this matters**: A "low" confidence response is likely to be a hallucination or an out-of-scope question. We surface this to the user rather than hiding it.
