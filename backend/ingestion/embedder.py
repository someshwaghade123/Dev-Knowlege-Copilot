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
from openai import OpenAI
from backend.core.config import settings

# ── Singleton client ────────────────────────────────────────────────────────
_client: OpenAI | None = None
_client_lock = threading.Lock()


def _get_client() -> OpenAI:
    """Lazy-load the OpenAI client with thread safety."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not settings.openai_api_key or "your_" in settings.openai_api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is not set. Please set it in your .env file."
                    )
                _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _get_model():
    """Compatibility shim for main.py pre-loading."""
    try:
        _get_client()
        print("[Embedder] OpenAI client initialised.")
    except Exception as e:
        print(f"[Embedder] Warning: OpenAI client init failed: {e}")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using OpenAI and return a 2D numpy array.
    """
    client = _get_client()
    
    # OpenAI suggests cleaning newlines for best performance
    cleaned_texts = [t.replace("\n", " ") for t in texts]

    response = client.embeddings.create(
        input=cleaned_texts,
        model=settings.embed_model
    )
    
    # Extract vectors from response
    embeddings_list = [item.embedding for item in response.data]
    
    # Convert to single 2D array and ensure float32 for FAISS
    embeddings = np.array(embeddings_list).astype(np.float32)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query.
    Note: text-embedding-3-small does not strictly require the BGE prefix,
    but we keep the interface consistent.
    """
    embedding = embed_texts([query])
    return embedding[0]   # Shape: (1536,)
