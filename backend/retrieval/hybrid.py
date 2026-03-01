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
import re
from backend.retrieval.vector_store import vector_store
from backend.ingestion.embedder import embed_query
from backend.retrieval.bm25_store import bm25_store
from backend.db.models import get_chunk_titles

def hybrid_search(query: str, top_k: int = 5, search_mode: str = "hybrid") -> dict:
    """
    Combined search interface with granular latency tracking and Title Boosting.
    """
    metrics = {"embed_ms": 0, "retrieval_ms": 0}
    
    # ── Step 1: Embedding ────────────────────────────────────────────────────
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
        # Fetch a wider pool (top_k * 10) to catch keyword matches that might
        # have lower initial semantic similarity.
        v_results = vector_store.search_by_vector(query_vector, top_k=top_k * 10)
        b_results = bm25_store.search(query, top_k=top_k * 10)
        
        k = 60
        scores = {}
        
        for rank, res in enumerate(v_results, start=1):
            fid = res["faiss_id"]
            scores[fid] = scores.get(fid, 0) + (1.0 / (k + rank))
            
        for rank, res in enumerate(b_results, start=1):
            fid = res["faiss_id"]
            scores[fid] = scores.get(fid, 0) + (1.0 / (k + rank))

        # ── Step 3: Title Boosting Heuristic (Week 4) ─────────────────────────
        if scores:
            keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
            faiss_ids = list(scores.keys())
            titles_map = get_chunk_titles(faiss_ids)
            
            for fid, score in scores.items():
                title = titles_map.get(fid, "").lower()
                if any(kw in title for kw in keywords):
                    # Multiplier: 3.0x (Significant priority for direct title matches)
                    scores[fid] = score * 3.0

        # Sort by boosted RRF score descending
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
