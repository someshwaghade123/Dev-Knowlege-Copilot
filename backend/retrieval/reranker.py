"""
backend/retrieval/reranker.py
──────────────────────────────
Two-stage retrieval: Re-ranks top-K results using a Cross-Encoder.

Vector/Hybrid search is "Bi-Encoder" (fast but less accurate).
Cross-Encoders are slower but MUCH more accurate at judging 
the relationship between a specific query and a specific chunk.
"""

import threading
import cohere
from backend.core.config import settings

class Reranker:
    """
    Two-stage retrieval re-ranker using Cohere's Rerank API.
    Offers superior accuracy with zero local memory footprint.
    """
    def __init__(self, model_name="rerank-english-v3.0"):
        self.model_name = model_name
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        """Lazy-load the Cohere client."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not settings.cohere_api_key or "your_" in settings.cohere_api_key:
                        # We return None so the system can gracefully degrade to hybrid-only
                        return None
                    self._client = cohere.Client(api_key=settings.cohere_api_key)
        return self._client

    @property
    def model(self):
        """Compatibility shim for main.py pre-loading."""
        if self.client:
            print("[Reranker] Cohere client initialised.")
        else:
            print("[Reranker] Warning: COHERE_API_KEY not found. Reranking will be bypassed.")
        return self.client

    def rerank(self, query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
        """
        Re-score chunks using Cohere.
        """
        if not chunks:
            return []

        if self.client is None:
            print("[Reranker] Bypassing rerank: Client not initialised.")
            return chunks[:top_n]

        # Extract text from chunks
        passages = [c["text"] for c in chunks]
        
        try:
            results = self.client.rerank(
                query=query,
                documents=passages,
                top_n=top_n,
                model=self.model_name
            )
            
            # Map results back to original chunks
            final_chunks = []
            for res in results.results:
                chunk = chunks[res.index]
                chunk["rerank_score"] = float(res.relevance_score)
                final_chunks.append(chunk)

            if final_chunks:
                print(f"[Reranker] Query: {query}")
                print(f"[Reranker] Top Cohere score: {final_chunks[0]['rerank_score']:.4f}")

            return final_chunks

        except Exception as e:
            print(f"[Reranker] Error during Cohere rerank: {e}")
            return chunks[:top_n]

# Singleton instance
reranker = Reranker()
