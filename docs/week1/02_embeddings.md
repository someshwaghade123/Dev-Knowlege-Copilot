# 02 — Embeddings Explained

## What is an Embedding?

An embedding converts text into a list of numbers (a **vector**) that captures its *semantic meaning*.

```
"How do I install FastAPI?" → [0.12, -0.45, 0.78, 0.33, ... ] (384 numbers)
"FastAPI installation guide" → [0.11, -0.44, 0.80, 0.31, ... ] ← Very close!
"Chocolate cake recipe"      → [0.55,  0.23, -0.12, 0.67, ...] ← Far away
```

Similar meaning = similar numbers = vectors that are **close in 384-dimensional space**.

---

## The Math: Cosine Similarity

To measure how similar two embeddings are, we use **cosine similarity**:

```
similarity = (A · B) / (|A| × |B|)

Range: -1 (opposite) to 1 (identical)
```

Because we **normalise** our vectors to length 1 (`|A| = |B| = 1`), this simplifies to:

```
similarity = A · B   (just a dot product!)
```

This is why we use `IndexFlatIP` (Inner Product) in FAISS.

---

## Why BGE-small-en-v1.5?

| Model | Dims | Size | MTEB Score | Notes |
|-------|------|------|-----------|-------|
| BGE-small-en-v1.5 ✅ | 384 | 33M params | 62.2 | Best small, fast |
| E5-small-v2 | 384 | 33M params | 59.9 | Good alternative |
| all-MiniLM-L6-v2 | 384 | 22M params | 56.3 | Very fast, lower quality |
| BGE-large-en-v1.5 | 1024 | 335M params | 64.2 | More accurate, 10x heavier |

**MTEB** = Massive Text Embedding Benchmark — the standard evaluation suite.

---

## The Query/Document Asymmetry (Important!)

BGE-small was trained with **different instructions** for documents vs queries.

| Input Type | Prefix Needed | Why |
|-----------|--------------|-----|
| Document chunks | None | Already informative text |
| User query | `"Represent this sentence for searching: "` | Tells model to optimise for retrieval |

```python
# Correct — document embedding
doc_embedding = model.encode("FastAPI is a web framework...")

# Correct — query embedding
query_embedding = model.encode(
    "Represent this sentence for searching: How do I install FastAPI?"
)

# Wrong — query without prefix
# Results in lower retrieval accuracy
query_embedding = model.encode("How do I install FastAPI?")
```

This is **asymmetric retrieval** — the query and document spaces are related but not identical.

---

## Embedding Dimension = Vector Size

- BGE-small outputs 384 numbers per text input
- BGE-large outputs 1,024 numbers
- More dimensions = more expressive, but: larger index, slower search, more RAM

For 20,000 chunks at 384 dimensions:
```
20,000 chunks × 384 dims × 4 bytes (float32) = ~30MB FAISS index
```

Trivially small. Even 1M chunks = ~1.5GB — manageable on a single machine.

---

## Batching

Embedding one text at a time is wasteful. We batch:

```python
# Slow (N separate model calls)
for text in texts:
    embedding = model.encode(text)

# Fast (one batched call)
embeddings = model.encode(texts, batch_size=32)
```

Our `embed_texts()` function uses `batch_size=32`, processing 32 chunks per GPU/CPU pass.

---

## Interview Questions on Embeddings

**Q: What is the output dimension of your embedding model and why did you pick it?**
> A: BGE-small-en-v1.5 outputs 384 dimensions. I chose it because it tops the MTEB leaderboard for its size class, is completely free/open-source, and 384 dims keeps the FAISS index under 30MB for 20k chunks — fast to load and search.

**Q: What's the difference between how you embed queries vs documents?**
> A: BGE uses asymmetric retrieval. Documents are encoded directly. Queries use the prefix "Represent this sentence for searching: ..." which was part of the model's training signal. Without this prefix, retrieval accuracy drops measurably.

**Q: Why do you normalise embeddings before storing them?**
> A: Normalising to unit length makes cosine similarity equivalent to dot product. FAISS's IndexFlatIP (inner product) then gives cosine similarity directly — no extra computation needed. It also makes scores comparable across queries.
