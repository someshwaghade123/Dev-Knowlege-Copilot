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
import asyncio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from backend.retrieval.vector_store import vector_store
from backend.retrieval.hybrid import hybrid_search
from backend.ingestion.embedder import embed_query
from backend.generation.llm import generate_answer, verify_factuality
from backend.db.models import (
    get_chunks_by_faiss_ids, 
    get_all_documents, 
    insert_query_log,
    get_latency_metrics
)

from backend.core.config import settings
from backend.retrieval.reranker import reranker
from backend.scoring.engine import compute_confidence
from backend.cache.cache_manager import cache_manager
from slowapi import Limiter
from slowapi.util import get_remote_address

# We'll initialize this in main.py but need the object here for decorators
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Schema for POST /query request body."""
    query: str
    top_k: int = settings.top_k         # Caller can override retrieval count
    min_score: float = 0.0              # Filter out very low relevance results
    search_mode: str = "hybrid"         # "hybrid" | "vector" | "bm25"
    min_confidence: str | None = None   # "high" | "medium" | "low"
    top_n_rerank: int = 5               # How many to keep after reranking
    bypass_llm: bool = False            # For benchmarking: skip LLM and only return retrieval results



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


class LatencyBreakdown(BaseModel):
    total: int
    embed: int
    retrieval: int
    llm: int
    rerank: int = 0
    fact: int = 0


class MetricsResponse(BaseModel):
    """Schema for GET /api/v1/metrics response."""
    total_queries: int
    total_tokens: int
    avg: LatencyBreakdown
    p50: LatencyBreakdown
    p95: LatencyBreakdown


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """
    Readiness probe.
    Verifies that the database and vector store are accessible.
    """
    health = {
        "status": "ok",
        "checks": {
            "database": "down",
            "vector_store": "down"
        }
    }
    
    # Check Database
    try:
        from backend.db.models import get_connection
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
        health["checks"]["database"] = "up"
    except Exception as e:
        health["status"] = "error"
        health["error_db"] = str(e)

    # Check Vector Store
    try:
        if vector_store._index is not None:
            health["checks"]["vector_store"] = "up"
            health["indexed_vectors"] = vector_store._index.ntotal
        else:
            health["status"] = "error"
            health["checks"]["vector_store"] = "not_loaded"
    except Exception as e:
        health["status"] = "error"
        health["error_vs"] = str(e)

    if health["status"] == "error":
        raise HTTPException(status_code=503, detail=health)
        
    return health


@router.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query_documents(body: QueryRequest, request: Request):
    """
    Main RAG endpoint with robust error logging and rate limiting.
    """
    wall_start = time.perf_counter()
    metrics = {}

    loop = asyncio.get_running_loop()

    try:
        if not body.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # ── Step 0: Embed Query Early for Semantic Cache ─────────────────────────
        # We need the vector to check for semantic similarity in the cache
        embed_start = time.perf_counter()
        query_vector = await loop.run_in_executor(None, embed_query, body.query)
        metrics["embed_ms"] = int((time.perf_counter() - embed_start) * 1000)

        # ── Step 1: Cache Lookup (Week 6/Refined) ─────────────────────────────────
        cached_data = cache_manager.get(query_vector)
        if cached_data:
            # FIX: Dynamically update latency for cache hits
            cached_data["latency_ms"] = int((time.perf_counter() - wall_start) * 1000)
            return QueryResponse(**cached_data)

        # ── Step 2: Hybrid search (Week 3/4) ─────────────────────────────────────
        # Note: We pass the pre-computed query_vector to avoid re-embedding
        search_data = await loop.run_in_executor(
            None, 
            hybrid_search, 
            body.query, 
            body.top_k, 
            body.search_mode,
            query_vector # New parameter
        )
        search_results = search_data["results"]
        metrics = search_data["metrics"]

        # Filter by minimum similarity score
        filtered = [r for r in search_results if r["score"] >= body.min_score]
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
        raw_chunks = await loop.run_in_executor(None, get_chunks_by_faiss_ids, faiss_ids)

        if not raw_chunks:
             # This should not happen if search_results were found
             raise Exception(f"No metadata found for FAISS IDs: {faiss_ids}")

        # CRITICAL: SQL 'IN' doesn't guarantee order. 
        # We must re-sort raw_chunks to match the order of faiss_ids (relevance).
        chunks_map = {c["faiss_id"]: c for c in raw_chunks}
        chunks = [chunks_map[fid] for fid in faiss_ids if fid in chunks_map]

        # ── Step 2.5: Re-ranking (Week 5) ─────────────────────────────────────────
        rerank_start = time.perf_counter()
        # We rerank all chunks from Step 2 to find the best top_n
        reranked = await loop.run_in_executor(
            None, 
            reranker.rerank, 
            body.query, 
            chunks, 
            body.top_n_rerank
        )
        metrics["rerank_ms"] = int((time.perf_counter() - rerank_start) * 1000)
        
        # New confidence based on Cross-Encoder scores
        confidence = compute_confidence(
            [c.get("rerank_score", -10.0) for c in reranked], 
            mode="rerank"
        )

        # ── Step 2.6: Confidence Guard ───────────────────────────────────────────
        # If user requested a minimum confidence and we are below it
        CONF_LEVELS = {"low": 0, "medium": 1, "high": 2}
        if body.min_confidence:
            target = CONF_LEVELS.get(body.min_confidence.lower(), 0)
            actual = CONF_LEVELS.get(confidence, 0)
            if actual < target:
                return QueryResponse(
                    answer=f"I don't have enough confidence ({confidence}) to answer this accurately based on your requirements.",
                    citations=[],
                    confidence=confidence,
                    latency_ms=int((time.perf_counter() - wall_start) * 1000),
                    tokens_used=0
                )

        chunks = reranked  # Use the re-ranked chunks for generation

        # ── Step 3: Generate answer (Optional) ──────────────────────────────────
        if body.bypass_llm:
            llm_result = {
                "answer": "[Bypass Mode] Retrieval only. No answer generated.",
                "tokens_used": 0
            }
            metrics["llm_ms"] = 0
            metrics["fact_ms"] = 0
            fact_check = {"is_grounded": True, "reasoning": "Bypass mode"}
        else:
            llm_start = time.perf_counter()
            llm_result = await generate_answer(body.query, chunks)
            metrics["llm_ms"] = int((time.perf_counter() - llm_start) * 1000)

            # ── Step 3.5: Factuality Check (Week 5) ──────────────────────────────
            # Perform a second-pass check to detect hallucinations
            fact_start = time.perf_counter()
            fact_check = await verify_factuality(body.query, llm_result["answer"], chunks)
            metrics["fact_ms"] = int((time.perf_counter() - fact_start) * 1000)
            
            # If the check fails, we downgrade confidence
            if not fact_check["is_grounded"]:
                confidence = "low"
                print(f"[Factuality Guard] Hallucination detected for query: {body.query}")
                print(f"Reasoning: {fact_check['reasoning']}")

        # ── Step 4: Build response ───────────────────────────────────────────────
        if body.bypass_llm:
            # In bypass mode, we return all top chunks as citations
            used_indices = list(range(1, len(chunks) + 1))
        else:
            from backend.generation.llm import extract_citation_indices
            used_indices = extract_citation_indices(llm_result["answer"])

        # If LLM didn't use any citations, or used invalid ones, we fallback?
        # Actually, if used_indices is empty, it means no [n] was found.
        # We only return citations that the LLM actually cited.
        
        all_citations = [
            Citation(
                title=c["title"],
                source_url=c.get("source_url"),
                text_preview=c.get("text_preview", c["text"][:200]),
            )
            for c in chunks
        ]
        
        # Filter: indices are 1-based from [1], [2]...
        citations = [all_citations[i-1] for i in used_indices if 0 < i <= len(all_citations)]

        total_latency_ms = int((time.perf_counter() - wall_start) * 1000)


        # ── Step 5: Log query for analytics (Week 2/4) ───────────────────────────
        try:
            insert_query_log(
                query=body.query,
                answer=llm_result.get("answer") or "N/A",
                confidence=confidence,
                latency_ms=total_latency_ms,
                tokens_used=llm_result.get("tokens_used", 0),
                embed_ms=metrics.get("embed_ms", 0),
                retrieval_ms=metrics.get("retrieval_ms", 0),
                rerank_ms=metrics.get("rerank_ms", 0),
                fact_ms=metrics.get("fact_ms", 0),
                llm_ms=metrics.get("llm_ms", 0),
            )
        except Exception as log_err:
            print(f"[Analytics] Non-critical: failed to log query: {log_err}")

        response = QueryResponse(
            answer=llm_result.get("answer") or "I could not generate an answer.",
            citations=citations,
            confidence=confidence,
            latency_ms=total_latency_ms,
            tokens_used=llm_result.get("tokens_used", 0),
        )

        # ── Step 6: Store in Cache (Week 6/Refined) ──────────────────────────────
        # We store the embedding alongside the response for semantic matching
        if not body.bypass_llm:
            cache_manager.set(body.query, query_vector, response.model_dump())

        return response

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


@router.get("/metrics", response_model=MetricsResponse)
async def get_performance_metrics():
    """
    Return aggregate performance statistics from query logs.
    """
    data = get_latency_metrics()
    summary = data["summary"]
    
    return MetricsResponse(
        total_queries=summary["total_queries"] or 0,
        total_tokens=summary["total_tokens"] or 0,
        avg=LatencyBreakdown(
            total=int(summary["avg_total"] or 0),
            embed=int(summary["avg_embed"] or 0),
            retrieval=int(summary["avg_retrieval"] or 0),
            llm=int(summary["avg_llm"] or 0),
            rerank=int(summary["avg_rerank"] or 0),
            fact=int(summary["avg_fact"] or 0)
        ),
        p50=LatencyBreakdown(**data["p50"]),
        p95=LatencyBreakdown(**data["p95"])
    )
