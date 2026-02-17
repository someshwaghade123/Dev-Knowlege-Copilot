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

from backend.retrieval.vector_store import vector_store
from backend.retrieval.bm25_store import bm25_store

def hybrid_search(query: str, top_k: int = 5, search_mode: str = "hybrid") -> list[dict]:
    """
    Combined search interface.
    
    Args:
        query: User's search string.
        top_k: Number of final results to return.
        search_mode: "vector", "bm25", or "hybrid".
    """
    
    if search_mode == "vector":
        return vector_store.search(query, top_k=top_k)
    
    if search_mode == "bm25":
        return bm25_store.search(query, top_k=top_k)

    # ── Hybrid Mode (RRF) ─────────────────────────────────────────────────────
    
    # Fetch more than top_k from each to ensure good overlap
    v_results = vector_store.search(query, top_k=top_k * 3)
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
    
    # Construct final result list
    final_results = []
    for fid, score in sorted_items:
        final_results.append({
            "faiss_id": fid,
            "score": float(score)  # This is the RRF score
        })
        
    return final_results
