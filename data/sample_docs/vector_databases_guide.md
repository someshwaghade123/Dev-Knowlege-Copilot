# Vector Databases and Similarity Search — Developer Guide

## What is a Vector Database?

A vector database stores data as high-dimensional vectors (arrays of numbers) and allows you to search for similar vectors efficiently. Instead of searching for exact keyword matches, vector databases find semantically similar content.

## Why Vector Databases?

Traditional databases answer: "Find rows where title = 'FastAPI'"
Vector databases answer: "Find documents that are semantically similar to 'how to build APIs in Python'"

This is the foundation of modern AI search, recommendation systems, and RAG (Retrieval-Augmented Generation).

## Core Concept: Embeddings

An embedding is a numerical representation of content:

```
"How to install Python?" → [0.23, -0.45, 0.12, 0.78, ...] (384 numbers)
"Python installation guide" → [0.24, -0.44, 0.13, 0.77, ...] (very similar!)
"Recipe for chocolate cake" → [-0.12, 0.67, -0.34, 0.02, ...] (very different!)
```

The distance between vectors represents semantic similarity.

## Similarity Metrics

### Cosine Similarity

Measures the angle between two vectors. Value between -1 and 1.
- 1.0 = identical meaning
- 0.0 = completely unrelated
- -1.0 = opposite meaning

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Dot Product

If vectors are L2-normalized (magnitude = 1), dot product == cosine similarity.
FAISS IndexFlatIP uses inner product (dot product).

### L2 Distance (Euclidean)

Measures straight-line distance. Smaller = more similar.
FAISS IndexFlatL2 uses this metric.

**Rule of thumb**: Use cosine similarity (or Inner Product on normalized vectors) for text embeddings.

## FAISS — Facebook AI Similarity Search

### Index Types

| Index | Type | Speed | Memory | Best Use |
|-------|------|-------|--------|----------|
| IndexFlatL2 | Exact | Slow | Low | < 10k vectors |
| IndexFlatIP | Exact | Slow | Low | < 10k, normalized |
| IndexIVFFlat | Approximate | Fast | Low | 10k–1M |
| IndexIVFPQ | Approximate | Very Fast | Very Low | Large scale |
| IndexHNSWFlat | Approximate | Very Fast | High | Low-latency |

### Basic FAISS Usage

```python
import faiss
import numpy as np

dimension = 384  # BGE-small output size
index = faiss.IndexFlatIP(dimension)

# Add vectors (must be float32)
vectors = np.random.random((1000, dimension)).astype(np.float32)
faiss.normalize_L2(vectors)  # Normalize for cosine similarity
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype(np.float32)
faiss.normalize_L2(query)
scores, indices = index.search(query, k=5)

print(f"Top 5 results: {indices[0]}")
print(f"Similarity scores: {scores[0]}")
```

### Saving and Loading FAISS Index

```python
# Save
faiss.write_index(index, "my_index.bin")

# Load
loaded_index = faiss.read_index("my_index.bin")
```

## ChromaDB — Alternative to FAISS

ChromaDB is a managed vector database with built-in metadata filtering.

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

# Add documents
collection.add(
    documents=["FastAPI is a web framework", "Docker containers are isolated"],
    metadatas=[{"source": "fastapi.md"}, {"source": "docker.md"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["how to build REST APIs"],
    n_results=2
)
```

### FAISS vs ChromaDB

| Feature | FAISS | ChromaDB |
|---------|-------|----------|
| Setup | Library (no server) | Can run in-memory or as server |
| Metadata filtering | No (use SQLite separately) | Built-in |
| Persistence | Manual save/load | Automatic |
| Scale | Billions of vectors | Millions of vectors |
| Language | Python, C++ | Python |

## Embedding Models

### Popular Open-Source Models

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| BGE-small-en-v1.5 | 384 | 33M params | Best small model |
| BGE-large-en-v1.5 | 1024 | 335M params | Higher accuracy |
| E5-small-v2 | 384 | 33M params | Good alternative |
| all-MiniLM-L6-v2 | 384 | 22M params | Very fast |
| text-embedding-ada-002 | 1536 | API only | OpenAI, costs money |

### Using sentence-transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Encode documents (no prefix needed)
doc_embeddings = model.encode(
    ["FastAPI tutorial", "Docker guide"],
    normalize_embeddings=True
)

# Encode queries (BGE needs prefix for queries)
query_embedding = model.encode(
    ["Represent this sentence for searching: how to use FastAPI"],
    normalize_embeddings=True
)
```

## Approximate Nearest Neighbor (ANN) Search

For large datasets, exact search is too slow. ANN trades a small accuracy loss for massive speed gains.

### HNSW (Hierarchical Navigable Small World)

Builds a multi-layer graph where each layer has fewer, longer-range connections:

```
Layer 2: A ←────────────────────────→ B
Layer 1: A ←──────→ C ←─────→ D ←──→ B
Layer 0: A → E → F → G → C → H → D → I → B  (full graph)
```

Search starts at top layer (few nodes, fast), narrows down to bottom layer.

### IVF (Inverted File Index)

Divides vector space into k clusters (Voronoi cells). At search time, only searches the nearest nprobe clusters.

```python
# IVF requires training
n_clusters = 100
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
index.train(training_vectors)  # Must train before adding
index.add(vectors)
index.nprobe = 10  # Search 10 nearest clusters
```

## Retrieval Evaluation Metrics

### Precision@K

Of the top-K retrieved documents, what fraction are actually relevant?

```python
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([r for r in retrieved_k if r in relevant_set]) / k
```

### Recall@K

Of all relevant documents, what fraction did we retrieve in top-K?

```python
def recall_at_k(retrieved, relevant, k):
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)
```

### MRR (Mean Reciprocal Rank)

Where does the first correct result appear?

```python
def mrr(queries_results, relevant_docs):
    rr_sum = 0
    for retrieved, relevant in zip(queries_results, relevant_docs):
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rr_sum += 1 / rank
                break
    return rr_sum / len(queries_results)
```
