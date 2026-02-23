import hashlib
import json
import time
from typing import Optional, Any

class CacheManager:
    """
    Manages response caching for queries to reduce LLM costs and latency.
    Supports in-memory caching by default, expandable to Redis for production.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._cache = {} # Key: md5_hash, Value: (expiry_timestamp, data)

    def _generate_key(self, query: str, top_k: int = 5) -> str:
        """Generate a unique key based on the query and retrieval parameters."""
        payload = f"{query.strip().lower()}:{top_k}"
        return hashlib.md5(payload.encode()).hexdigest()

    def get(self, query: str, top_k: int = 5) -> Optional[dict]:
        """Retrieve a cached response if it exists and hasn't expired."""
        key = self._generate_key(query, top_k)
        if key in self._cache:
            expiry, data = self._cache[key]
            if time.time() < expiry:
                print(f"[Cache] Hit for: {query}")
                return data
            else:
                print(f"[Cache] Expired for: {query}")
                del self._cache[key]
        return None

    def set(self, query: str, top_k: int = 5, data: dict = {}) -> None:
        """Store a response in the cache with a TTL."""
        key = self._generate_key(query, top_k)
        expiry = time.time() + self.ttl
        self._cache[key] = (expiry, data)
        print(f"[Cache] Stored response for: {query}")

# Global instance for app-wide use
cache_manager = CacheManager()
