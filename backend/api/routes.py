"""
backend/api/routes.py
──────────────────────
FastAPI route definitions — the HTTP interface to the RAG system.

API DESIGN DECISIONS:
  - POST /query : Takes JSON body (not query params) — allows complex future fields
  - GET /health : Lightweight check — no DB/index required
  - GET /documents : Returns all indexed doc metadata

RESPONSE CONTRACT (locked from Week 1 — don't change this JSON shape later):
  POST /query response:
  {
    "answer": "...",
    "citations": [
      {"title": "...", "source_url": "...", "text_preview": "..."}
    ],
    "confidence": "high" | "medium" | "low",
    "latency_ms": 482,
    "tokens_used": 347
  }

INTERVIEW TIP:
  "I used Pydantic models for both request validation and response schema.
   This gives automatic OpenAPI docs at /docs and ensures the mobile app
   always receives well-typed JSON — no runtime surprises."
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.retrieval.vector_store import vector_store
from backend.generation.llm import generate_answer
from backend.db.models import get_chunks_by_faiss_ids, get_all_documents, insert_query_log
from backend.core.config import settings

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Schema for POST /query request body."""
    query: str
    top_k: int = settings.top_k         # Caller can override retrieval count
    min_score: float = 0.0              # Filter out very low relevance results


class Citation(BaseModel):
    """A single source reference included in the response."""
    title: str
    source_url: str | None
    text_preview: str       # First ~200 chars of the chunk shown to user


class QueryResponse(BaseModel):
    """Schema for POST /query response."""
    answer: str
    citations: list[Citation]
    confidence: str         # "high" | "medium" | "low"
    latency_ms: int
    tokens_used: int


class DocumentMetadata(BaseModel):
    """Schema for a single document's metadata."""
    id: int
    title: str
    source_url: str | None
    file_name: str
    ingested_at: str
    chunk_count: int
    total_tokens: int | None # SUM can be NULL if no chunks


class DocumentsListResponse(BaseModel):
    """Schema for GET /documents response."""
    count: int
    documents: list[DocumentMetadata]


# ── Confidence scoring helper ─────────────────────────────────────────────────

def compute_confidence(scores: list[float]) -> str:
    """
    Simple rule-based confidence from FAISS similarity scores.

    FAISS IndexFlatIP returns cosine similarity in [-1, 1] for normalised vectors.
      > 0.80 → high    (very relevant chunks found)
      > 0.60 → medium
      ≤ 0.60 → low     (no strong match in index)

    INTERVIEW TIP: In Week 5 we add an LLM self-check on top of this.
    """
    if not scores:
        return "low"
    top_score = max(scores)
    if top_score > 0.80:
        return "high"
    elif top_score > 0.60:
        return "medium"
    else:
        return "low"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """
    Lightweight liveness probe.
    Returns immediately — no DB or index access.
    Used by Docker health checks and load balancers.
    """
    return {
        "status": "ok",
        "indexed_vectors": vector_store._next_id,
        "env": settings.app_env,
    }


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Main RAG endpoint with robust error logging.
    """
    wall_start = time.perf_counter()

    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        if vector_store._next_id == 0:
            raise HTTPException(
                status_code=503,
                detail="No documents indexed yet. Run ingest_docs.py first."
            )

        # ── Step 1: Vector search ────────────────────────────────────────────────
        search_results = vector_store.search(request.query, top_k=request.top_k)

        # Filter by minimum similarity score
        filtered = [r for r in search_results if r["score"] >= request.min_score]
        if not filtered:
            return QueryResponse(
                answer="I could not find relevant information for your query.",
                citations=[],
                confidence="low",
                latency_ms=int((time.perf_counter() - wall_start) * 1000),
                tokens_used=0,
            )

        # ── Step 2: Fetch metadata from SQLite ───────────────────────────────────
        faiss_ids = [r["faiss_id"] for r in filtered]
        raw_chunks = get_chunks_by_faiss_ids(faiss_ids)

        if not raw_chunks:
             # This should not happen if search_results were found
             raise Exception(f"No metadata found for FAISS IDs: {faiss_ids}")

        # CRITICAL: SQL 'IN' doesn't guarantee order. 
        # We must re-sort raw_chunks to match the order of faiss_ids (relevance).
        chunks_map = {c["faiss_id"]: c for c in raw_chunks}
        chunks = [chunks_map[fid] for fid in faiss_ids if fid in chunks_map]

        # ── Step 3: Generate answer ──────────────────────────────────────────────
        llm_result = await generate_answer(request.query, chunks)

        # ── Step 4: Build response ───────────────────────────────────────────────
        citations = [
            Citation(
                title=c["title"],
                source_url=c.get("source_url"),
                text_preview=c.get("text_preview", c["text"][:200]),
            )
            for c in chunks
        ]

        total_latency_ms = int((time.perf_counter() - wall_start) * 1000)
        confidence = compute_confidence([r["score"] for r in filtered])

        # ── Step 5: Log query for analytics (Week 2) ─────────────────────────────
        insert_query_log(
            query=request.query,
            answer=llm_result["answer"],
            confidence=confidence,
            latency_ms=total_latency_ms,
            tokens_used=llm_result["tokens_used"],
        )

        return QueryResponse(
            answer=llm_result["answer"],
            citations=citations,
            confidence=confidence,
            latency_ms=total_latency_ms,
            tokens_used=llm_result["tokens_used"],
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API Error] /query failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents():
    """
    Return all indexed documents with statistics.
    Useful for the mobile app's "Browse" screen.
    """
    docs = get_all_documents()
    return DocumentsListResponse(count=len(docs), documents=docs)
