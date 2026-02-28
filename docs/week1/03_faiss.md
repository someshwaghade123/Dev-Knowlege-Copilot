# 03 — FAISS Explained

## What is FAISS?

FAISS (Facebook AI Similarity Search) is a library that answers one question efficiently:

> "Given a query vector Q, find the K stored vectors most similar to Q."

Without FAISS, you'd compare the query against every stored vector — O(N × D) where N = number of chunks and D = dimensions. At 20,000 chunks with 384 dims:

```
20,000 × 384 = 7,680,000 multiplication operations per query
```

FAISS makes this fast through clever indexing structures.

---

## Index Types — Visual Comparison

### IndexFlatIP (What We Use in Week 1)

```
All vectors stored in a flat array. Exact search — checks every vector.

Query Q → [compare to V1, V2, V3, ..., V20000] → return top-5

Pros: 100% accurate (no missed results)
Cons: O(N) per query — slow for N > 100k
Best for: < 50k vectors (perfect for Week 1)
```

### IndexIVFFlat (Week 6 upgrade)

```
Divides space into K clusters (Voronoi cells)

         ┌─── Cluster 1 ───┐     ┌─── Cluster 2 ───┐
         │  V1  V4  V7     │     │  V2  V5  V8     │
         └─────────────────┘     └─────────────────┘
              ┌─── Cluster 3 ───┐
              │  V3  V6  V9     │
              └─────────────────┘

Query Q → Find nearest 10 cluster centres → Search only those clusters
         (Not all 20,000 — maybe only 2,000)

Pros: 5–10x faster than FlatIP
Cons: ~5% recall loss (might miss some true neighbours)
```

### IndexHNSWFlat (Production-grade)

```
Multi-layer graph. Higher layers = fewer nodes, longer edges.

Layer 2:  A ─────────────────── B
Layer 1:  A ──────── C ──────── B
Layer 0:  A ─ D ─ E ─ C ─ F ─ G ─ B

Search enters at top layer, navigates down to find nearest neighbours.
Very fast, good recall, but uses more RAM.
```

---

## How We Use FAISS

### Step 1 — Create index

```python
import faiss
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine for normalised vecs
```

### Step 2 — Add vectors

```python
# embeddings: float32 array of shape (N, 384)
index.add(embeddings)   # Assigns IDs 0, 1, 2, ...
```

### Step 3 — Search

```python
query_vec = embed_query("How to install FastAPI?")   # Shape: (1, 384)
scores, indices = index.search(query_vec, k=5)

# scores[0] = [0.91, 0.87, 0.83, 0.79, 0.71]  (cosine similarities)
# indices[0] = [42, 17, 8, 103, 56]             (vector IDs = faiss_ids)
```

### Step 4 — Save / Load

```python
# Save (after ingestion)
faiss.write_index(index, "data/faiss_index.bin")

# Load (at app startup)
index = faiss.read_index("data/faiss_index.bin")
```

---

## FAISS ID → SQLite Chunk Metadata

FAISS only stores vectors and returns integer IDs. To get the actual text + title:

```
FAISS search → [42, 17, 8] → SQLite query → chunk text + source URL + title
                                    ▲
                                    faiss_id column in chunks table
```

This separation is intentional:
- FAISS is optimised for vector math (fast)
- SQLite is optimised for structured queries (flexible)

---

## Key Numbers to Know

| Metric | Value |
|--------|-------|
| Index type (Week 1) | IndexFlatIP |
| Embedding dimension | 384 |
| Storage per vector | 4 bytes × 384 = 1,536 bytes |
| 20k chunks index size | ~30MB |
| Typical search latency | 5–40ms (CPU) |
| Exact search? | Yes — no missed results |

---

## Interview Questions on FAISS

**Q: Why did you use FAISS over a managed vector DB like Pinecone?**
> A: For this scale (20k documents), FAISS is free, runs locally, and has zero latency overhead from network calls to an external service. It's also a great learning tool — I understand what's happening under the hood. Pinecone makes sense when you need multi-tenancy, managed scaling, or metadata filtering at the DB layer.

**Q: What FAISS index type did you use and why?**
> A: IndexFlatIP for Week 1 — exact search, no approximation. For N < 50k chunks this is perfectly fast. I'd upgrade to IndexIVFFlat once we reach 100k+ chunks. I've documented the tradeoff: IVFFlat gives ~5% recall loss but 10x speedup.

**Q: Why Inner Product instead of L2 distance?**
> A: Our embeddings are L2-normalised (magnitude = 1). For normalised vectors, inner product equals cosine similarity. Using IndexFlatIP with normalised vectors is equivalent to cosine similarity search — the preferred metric for text embeddings.
