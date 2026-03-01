"""
backend/retrieval/bm25_store.py
───────────────────────────────
Keyword-based retrieval using the Okapi BM25 algorithm.
BM25 is excellent for "exact match" queries (function names, error codes)
where vector search might be too fuzzy.

WHY BM25?
  Vector search is "semantic" — it knows "dog" is like "canine".
  BM25 is "lexical" — it knows "os.path.join" must match exactly "os.path.join".
  Hybrid RAG combines both for the best accuracy.
"""

import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from backend.core.config import settings

class BM25Store:
    """
    Wrapper around rank_bm25 to provide a persistent keyword index.
    Maps results to the same 'faiss_id' used in our vector index and SQLite.
    """
    def __init__(self):
        self.index = None
        self.faiss_ids = []  # Parallel list mapping BM25 doc index -> database faiss_id

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer for code and text.
        Lowercases and splits by whitespace/common punctuation.
        """
        # In a real production app, we'd use a more robust code tokenizer.
        # For Week 3, a clean split is a great start.
        return text.lower().replace("(", " ").replace(")", " ").replace(".", " ").split()

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build a fresh BM25 index from a list of chunks.
        chunks: List of dicts with 'text' and 'faiss_id'
        """
        if not chunks:
            print("[BM25] No chunks provided to build index.")
            return

        print(f"[BM25] Building index for {len(chunks)} chunks...")
        corpus = [self._tokenize(c["text"]) for c in chunks]
        self.faiss_ids = [c["faiss_id"] for c in chunks]
        self.index = BM25Okapi(corpus)
        self.save()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Perform keyword search and return scores.
        Returns: list of {"faiss_id": it, "score": float}
        """
        if not self.index:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # BM25 scores can be 0 if no keywords match — we filter those out
            if scores[idx] > 0:
                results.append({
                    "faiss_id": self.faiss_ids[idx],
                    "score": float(scores[idx])
                })
        return results

    def save(self) -> None:
        """Persist the index to disk using pickle."""
        os.makedirs(os.path.dirname(settings.bm25_index_path), exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "faiss_ids": self.faiss_ids
            }, f)
        print(f"[BM25] Index saved to {settings.bm25_index_path}")

    def load(self) -> None:
        """Load the index from disk if it exists."""
        if os.path.exists(settings.bm25_index_path):
            try:
                with open(settings.bm25_index_path, "rb") as f:
                    data = pickle.load(f)
                    self.index = data["index"]
                    self.faiss_ids = data["faiss_ids"]
                print(f"[BM25] Loaded index with {len(self.faiss_ids)} chunks.")
            except Exception as e:
                print(f"[BM25] Error loading index: {e}")
        else:
            print("[BM25] No index found on disk.")

# Singleton instance
bm25_store = BM25Store()

# Load at module import time
bm25_store.load()
