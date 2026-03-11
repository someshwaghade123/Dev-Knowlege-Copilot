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
from fastembed import TextEmbedding
from backend.core.config import settings

# ── Singleton model ─────────────────────────────────────────────────────────
_model: TextEmbedding | None = None
_model_lock = threading.Lock()


def _get_model() -> TextEmbedding:
    """Lazy-load the embedding model with thread safety."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # Double-checked locking
                print(f"[Embedder] Loading model: {settings.embed_model}")
                # fastembed will download to a local cache if not present
                _model = TextEmbedding(model_name=settings.embed_model)
                print(f"[Embedder] Model loaded via FastEmbed.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts and return a 2D numpy array of shape (N, D).
    """
    model = _get_model()

    # fastembed returns a generator of numpy arrays
    # We convert to a single numpy array
    # Note: fastembed automatically handles normalization if requested in constructor,
    # but for BGE it usually returns unit vectors by default or we can manually normalize.
    # TextEmbedding.embed() returns an iterable of numpy arrays.
    embeddings_generator = model.embed(texts)
    embeddings_list = list(embeddings_generator)
    
    # Convert to single 2D array and ensure float32 for FAISS
    embeddings = np.array(embeddings_list).astype(np.float32)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query with the BGE query-side prefix.
    """
    # BGE instruction-tuned prefix
    prefixed_query = f"Represent this sentence for searching: {query}"
    embedding = embed_texts([prefixed_query])
    return embedding[0]   # Shape: (384,)
