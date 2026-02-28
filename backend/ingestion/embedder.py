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

import numpy as np
from sentence_transformers import SentenceTransformer
from backend.core.config import settings

# ── Singleton model ─────────────────────────────────────────────────────────
# Model is loaded once at module import time.
# Subsequent calls to embed() reuse the already-loaded model (no re-download).
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloaded once, cached locally)."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {settings.embed_model}")
        _model = SentenceTransformer(settings.embed_model)
        print(f"[Embedder] Model loaded. Output dimension: {settings.embed_dimension}")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts and return a 2D numpy array of shape (N, D).

    Args:
        texts: List of strings to embed (chunks or queries)

    Returns:
        float32 numpy array of shape (len(texts), embed_dimension)
        Each row is the embedding vector for the corresponding text.

    NOTE on BGE query prefix:
        BGE-small was instruction-tuned. For *documents* being indexed, no
        prefix is needed. For *queries* at search time, prefix with:
        "Represent this sentence for searching: "
        We handle this distinction in vector_store.py.
    """
    model = _get_model()

    # normalize_embeddings=True → L2-normalised vectors
    # This makes cosine similarity equivalent to dot product — faster with FAISS
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,        # Process 32 texts at a time (adjust for RAM)
        show_progress_bar=len(texts) > 50,   # Show bar only for large batches
    )
    return embeddings.astype(np.float32)   # FAISS requires float32


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query with the BGE query-side prefix.
    Returns a 1D float32 array of shape (embed_dimension,).
    """
    # The query prefix tells BGE to optimise for retrieval (asymmetric search)
    prefixed_query = f"Represent this sentence for searching: {query}"
    embedding = embed_texts([prefixed_query])
    return embedding[0]   # Shape: (384,)
