"""
backend/retrieval/hybrid.py
───────────────────────────
The heart of hybrid retrieval — merging semantic and keyword search.

ALGORITHM: Reciprocal Rank Fusion (RRF)
  - We take the Top-K results from Vector search and Top-K from BM25.
  - We give each document a score: 1 / (60 + rank).
  - Documents appearing at the top of BOTH lists get the highest final score.
  - This is mathematically robust and doesn't require normalizing scores 
    to the same scale (which is hard because FAISS is cosine, BM25 is unbounded).
"""

import time
from backend.retrieval.vector_store import vector_store
from backend.ingestion.embedder import embed_query
from backend.retrieval.bm25_store import bm25_store

def hybrid_search(query: str, top_k: int = 5, search_mode: str = "hybrid") -> dict:
    """
    Combined search interface with granular latency tracking.
    
    Returns:
        dict: {
            "results": list[dict],
            "embed_ms": int,
            "retrieval_ms": int
        }
    """
    metrics = {"embed_ms": 0, "retrieval_ms": 0}
    
    # ── Step 1: Embedding (Vector logic only) ────────────────────────────────
    query_vector = None
    if search_mode in ["vector", "hybrid"]:
        t0 = time.perf_counter()
        query_vector = embed_query(query)
        metrics["embed_ms"] = int((time.perf_counter() - t0) * 1000)

    # ── Step 2: Retrieval ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    
    if search_mode == "vector":
        final_results = vector_store.search_by_vector(query_vector, top_k=top_k)
    
    elif search_mode == "bm25":
        final_results = bm25_store.search(query, top_k=top_k)
    
    else:  # Hybrid Mode (RRF)
        # Fetch more than top_k from each to ensure good overlap
        v_results = vector_store.search_by_vector(query_vector, top_k=top_k * 3)
        b_results = bm25_store.search(query, top_k=top_k * 3)
        
        k = 60  # RRF constant
        scores = {}  # faiss_id -> rrf_score
        
        # Score Vector Results
        for rank, res in enumerate(v_results, start=1):
            fid = res["faiss_id"]
            scores[fid] = scores.get(fid, 0) + (1.0 / (k + rank))
            
        # Score BM25 Results
        for rank, res in enumerate(b_results, start=1):
            fid = res["faiss_id"]
            scores[fid] = scores.get(fid, 0) + (1.0 / (k + rank))
            
        # Sort by RRF score descending
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = [
            {"faiss_id": fid, "score": float(score)} 
            for fid, score in sorted_items
        ]
        
    metrics["retrieval_ms"] = int((time.perf_counter() - t0) * 1000)
    
    return {
        "results": final_results,
        "metrics": metrics
    }
