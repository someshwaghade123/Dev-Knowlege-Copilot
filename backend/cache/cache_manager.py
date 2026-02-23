import time
import threading
import numpy as np
import faiss
from typing import Optional, Any

class CacheManager:
    """
    Manages semantic response caching using vector similarity.
    Uses a lightweight FAISS index to find "near-matches" to previous queries.
    """
    def __init__(self, dimension: int = 384, ttl_seconds: int = 3600, threshold: float = 0.95):
        self.dimension = dimension
        self.ttl = ttl_seconds
        self.threshold = threshold
        
        self._lock = threading.Lock()
        self._index = faiss.IndexFlatIP(dimension)
        self._cache_data = []  # List of (expiry, data, original_query) matching index order

    def get(self, query_vector: np.ndarray) -> Optional[dict]:
        """
        Find a semantically similar cached response.
        Returns the data if similarity > threshold and hasn't expired.
        """
        if self._index.ntotal == 0:
            return None

        # Ensure correct shape (1, dim)
        v = query_vector.reshape(1, -1).astype("float32")
        
        with self._lock:
            # Search the cache index for the single closest match
            scores, indices = self._index.search(v, 1)
            
            idx = indices[0][0]
            score = scores[0][0]

            if idx != -1 and score >= self.threshold:
                expiry, data, original_query = self._cache_data[idx]
                if time.time() < expiry:
                    print(f"[Cache] Semantic Hit! Score: {score:.4f}")
                    print(f"  Match: '{original_query}'")
                    return data
                else:
                    # In a real system, we'd remove expired entries from FAISS.
                    # For this MVP, we just treat it as a miss.
                    print(f"[Cache] Semantic match found but expired.")
        return None

    def set(self, query: str, query_vector: np.ndarray, data: dict) -> None:
        """Store a response and its embedding in the semantic cache."""
        v = query_vector.reshape(1, -1).astype("float32")
        expiry = time.time() + self.ttl
        
        with self._lock:
            self._index.add(v)
            self._cache_data.append((expiry, data, query))
            print(f"[Cache] Stored semantic entry for: {query}")

# Global instance for app-wide use
cache_manager = CacheManager()
