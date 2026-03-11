"""
backend/retrieval/reranker.py
──────────────────────────────
Two-stage retrieval: Re-ranks top-K results using a Cross-Encoder.

Vector/Hybrid search is "Bi-Encoder" (fast but less accurate).
Cross-Encoders are slower but MUCH more accurate at judging 
the relationship between a specific query and a specific chunk.
"""

import threading
try:
    from fastembed import TextCrossEncoder
except ImportError:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError:
        # If both fail, we'll raise a clear error later during lazy loading
        TextCrossEncoder = None

from backend.core.config import settings

class Reranker:
    """
    Two-stage retrieval re-ranker using a Cross-Encoder via FastEmbed.
    Optimized with lazy loading to save memory at startup.
    """
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            if TextCrossEncoder is None:
                raise ImportError(
                    "Could not import TextCrossEncoder from fastembed. "
                    "Ensure fastembed >= 0.4.2 is installed."
                )
            with self._lock:
                if self._model is None:  # Double-check
                    print(f"[Reranker] Loading {self.model_name} via FastEmbed...")
                    self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
        """
        Re-score a list of chunks based on their semantic relevance to the query.
        """
        if not chunks:
            return []

        # Extract text from chunks for the reranker
        passages = [c["text"] for c in chunks]
        
        # Predict relevance scores (returns a generator of RerankResult)
        # Each result has .score and .index
        results = list(self.model.rerank(query, passages))
        
        # Re-attach scores to original chunks
        for result in results:
            idx = result.index
            chunks[idx]["rerank_score"] = float(result.score)
            
        # Sort by rerank_score descending
        ranked_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", -999), reverse=True)
        
        return ranked_chunks[:top_n]

# Singleton instance
reranker = Reranker()
