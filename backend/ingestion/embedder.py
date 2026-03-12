"""
backend/ingestion/embedder.py
──────────────────────────────
Converts text chunks into dense vector embeddings using BGE-small-en-v1.5.

WHAT IS AN EMBEDDING?
  An embedding is a list of floating-point numbers (a vector) that captures
  the *meaning* of a piece of text. Similar meanings → similar vectors.

  "How do I install FastAPI?" → [0.12, -0.45, 0.78, ...]  (384 numbers)
  "FastAPI installation guide" → [0.11, -0.44, 0.80, ...]  (very close!)
  "The cat sat on the mat"    → [0.55, 0.23, -0.12, ...]  (far away)

WHY BGE-SMALL-EN-V1.5?
  - Open source, no API cost
  - 384 dimensions (small = fast, good enough for our use case)
  - Ranked top on MTEB benchmark for its size class
  - "bge" = Beijing Academy of AI (BAAI)
  - The instruction prefix "Represent this sentence for searching: " improves
    retrieval accuracy by telling the model the usage intent

INTERVIEW TIP:
  "Embedding dimension is a tradeoff. BGE-small gives 384 dims — smaller
   index, faster search, but slightly less semantic precision than
   BGE-large (1024 dims). For 20k documents at 384 dims, the FAISS index
   is ~30MB — trivially small."
"""

import threading
import numpy as np
import cohere
from backend.core.config import settings

# ── Singleton client ────────────────────────────────────────────────────────
_client: cohere.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> cohere.Client:
    """Lazy-load the Cohere client with thread safety."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not settings.cohere_api_key or "your_" in settings.cohere_api_key:
                    raise ValueError(
                        "COHERE_API_KEY is not set. Please set it in your .env file."
                    )
                _client = cohere.Client(api_key=settings.cohere_api_key)
    return _client


def _get_model():
    """Compatibility shim for main.py."""
    try:
        _get_client()
        print("[Embedder] Cohere client initialised.")
    except Exception as e:
        print(f"[Embedder] Warning: Cohere client init failed: {e}")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using Cohere.
    """
    client = _get_client()
    
    # Cohere v3 requires 'input_type' for better retrieval performance
    # 'search_document' for ingestion, 'search_query' for queries
    # For bulk ingestion, we use search_document
    response = client.embed(
        texts=texts,
        model=settings.embed_model,
        input_type="search_document",
        embedding_types=["float"]
    )
    
    # Extract vectors
    embeddings_list = response.embeddings.float
    
    # Convert to single 2D array and ensure float32 for FAISS
    embeddings = np.array(embeddings_list).astype(np.float32)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query using Cohere.
    """
    client = _get_client()
    
    response = client.embed(
        texts=[query],
        model=settings.embed_model,
        input_type="search_query",
        embedding_types=["float"]
    )
    
    embedding = np.array(response.embeddings.float[0]).astype(np.float32)
    return embedding   # Shape: (1024,)
