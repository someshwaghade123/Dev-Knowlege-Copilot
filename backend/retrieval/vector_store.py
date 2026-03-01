"""
backend/retrieval/vector_store.py
──────────────────────────────────
FAISS-based vector store: add embeddings, save to disk, load from disk, search.

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library for efficient nearest
  neighbour search in high-dimensional vector spaces.

  Given a query vector Q and a set of stored vectors [V1, V2, ..., Vn],
  FAISS finds the K vectors most similar to Q — extremely fast.

FAISS INDEX TYPES (We use IndexFlatIP for Week 1):
  ┌──────────────────┬──────────────┬──────────────┬────────────────────┐
  │ Index Type       │ Exact/Approx │ Speed        │ Use When           │
  ├──────────────────┼──────────────┼──────────────┼────────────────────┤
  │ IndexFlatL2      │ Exact        │ Slow (large) │ < 10k vectors      │
  │ IndexFlatIP      │ Exact        │ Slow (large) │ < 10k, normalised  │
  │ IndexIVFFlat     │ Approx       │ Fast         │ 10k–1M vectors     │
  │ IndexHNSWFlat    │ Approx       │ Very fast    │ Low-latency needed │
  └──────────────────┴──────────────┴──────────────┴────────────────────┘
  We use IndexFlatIP (Inner Product) because our embeddings are L2-normalised,
  making inner product == cosine similarity.

SAVE/LOAD STRATEGY:
  - FAISS index saved to disk as a .bin file (binary format)
  - FAISS IDs saved as a numpy array (.npy) alongside the index
  - On startup, load both files if they exist

INTERVIEW TIP:
  "In Week 1 I used IndexFlatIP for exact search — correctness over speed.
   Once we scale past 50k chunks I'd switch to IndexIVFFlat which uses
   quantised clusters for ~10x speedup with <5% recall loss."
"""

import numpy as np
import faiss
from pathlib import Path
from backend.core.config import settings
from backend.ingestion.embedder import embed_query


class VectorStore:
    """
    Wraps a FAISS index with save/load and search capabilities.

    The store assigns each embedding a sequential integer ID (faiss_id).
    This faiss_id is stored in SQLite so we can look up chunk metadata
    after a search returns IDs.
    """

    def __init__(self) -> None:
        self.dimension = settings.embed_dimension
        self.index_path = Path(settings.faiss_index_path)
        self._next_id: int = 0   # Auto-incrementing FAISS ID counter
        self._index: faiss.IndexFlatIP | None = None

    # ── Initialisation ──────────────────────────────────────────────────────

    def load_or_create(self) -> None:
        """
        Load an existing index from disk, or create a fresh empty one.
        Call this once at application startup.
        """
        if self.index_path.exists():
            print(f"[VectorStore] Loading existing index from {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            self._next_id = self._index.ntotal   # Resume ID counter
            print(f"[VectorStore] Loaded {self._next_id} vectors.")
        else:
            print("[VectorStore] Creating new empty index.")
            self._index = faiss.IndexFlatIP(self.dimension)
            self._next_id = 0

    # ── Adding vectors ───────────────────────────────────────────────────────

    def add_embeddings(self, embeddings: np.ndarray) -> list[int]:
        """
        Add a batch of embeddings to the index.

        Args:
            embeddings: float32 array of shape (N, dimension)

        Returns:
            List of assigned faiss_ids (sequential integers)
        """
        assert self._index is not None, "Call load_or_create() first."
        assert embeddings.dtype == np.float32

        n = len(embeddings)
        assigned_ids = list(range(self._next_id, self._next_id + n))

        self._index.add(embeddings)   # FAISS adds in order, no explicit IDs
        self._next_id += n

        return assigned_ids

    def save(self) -> None:
        """Persist the index to disk. Includes a safety guard to avoid wiping data."""
        if self._index is None or self._index.ntotal == 0:
            print("[VectorStore] Index is empty. Skipping save to protect existing data.")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        print(f"[VectorStore] Saved {self._next_id} vectors to {self.index_path}")

    # ── Searching ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Search the index for chunks closest to the query embedding.
        Default implementation that includes embedding step.
        """
        k = top_k or settings.top_k
        query_vector = embed_query(query)
        return self.search_by_vector(query_vector, k)

    def search_by_vector(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """
        Search using a pre-computed vector. 
        Useful for measuring embed vs search latency separately.
        """
        assert self._index is not None, "Call load_or_create() first."
        
        # Ensure correct shape (1, dim)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self._index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"faiss_id": int(idx), "score": float(score)})

        return results


# ── Singleton instance ───────────────────────────────────────────────────────
# Import this in routes.py and ingestion scripts
vector_store = VectorStore()
