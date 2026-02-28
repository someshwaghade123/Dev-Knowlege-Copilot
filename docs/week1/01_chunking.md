# 01 — Document Chunking Explained

## The Problem

A technical document like a FastAPI README might be **3,000 tokens** long.
The BGE-small embedding model can only process **512 tokens at a time**.
The LLM context window is also finite.

**Solution**: Split the document into smaller, overlapping pieces called **chunks**.

---

## What is a Chunk?

A chunk is a contiguous slice of a document, measured in tokens (not characters).

```
Full Document (1,200 tokens)
┌─────────────────────────────────────────────────┐
│ # FastAPI Guide                                 │
│ FastAPI is a modern web framework...            │
│ ## Installation                                 │
│ pip install fastapi uvicorn...                  │
│ ## Hello World                                  │
│ Create main.py and run...                       │
│ ## Path Parameters                              │
│ You can declare path parameters...              │
└─────────────────────────────────────────────────┘
```

After chunking (chunk_size=384, overlap=64):

```
Chunk 0: tokens 0–383    "FastAPI is a modern web framework... ## Installation pip install..."
Chunk 1: tokens 320–703  "pip install fastapi uvicorn... ## Hello World Create main.py..."
Chunk 2: tokens 640–1023 "Create main.py and run... ## Path Parameters You can declare..."
Chunk 3: tokens 960–1199 "You can declare path parameters... (end of document)"
                ◄──────►
                overlap = 64 tokens
```

---

## Why Overlap?

Without overlap, a sentence like:

```
"...authentication using OAuth2 which is the industry standard..."
```

might be split like this:

```
Chunk 4 ends:  "...authentication using OAuth2"
Chunk 5 starts: "which is the industry standard..."
```

A query about *"OAuth2 industry standard"* would not match either chunk cleanly.

With 64-token overlap, this sentence appears **complete** in at least one chunk.

---

## Token vs Character Chunking

We always chunk by **tokens**, not characters.

| Approach | Problem |
|----------|---------|
| Character-based | "Hello" = 5 chars but counts as 1 token. Very imprecise. |
| Word-based | Words vary in token size ("tokenization" = 3 tokens). |
| Token-based ✅ | Exactly matches what the embedding model sees. |

We use `tiktoken` (OpenAI's tokenizer, `cl100k_base` encoding) as a universal counter.

---

## Chunk Size Tradeoff

| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (128) | Very precise retrieval | Loses context, many chunks |
| Medium (384) ✅ | Good balance | — |
| Large (512) | More context per chunk | May exceed model limit, less precise |

**Our choice**: 384 tokens, because BGE-small has a 512-token limit and we leave headroom.

---

## Code Walkthrough

```python
# backend/ingestion/chunker.py

token_ids = _tokenizer.encode(text)   # Entire doc as token IDs
step = chunk_size - overlap           # How far we slide each iteration

start = 0
while start < len(token_ids):
    end = min(start + chunk_size, len(token_ids))
    chunk_tokens = token_ids[start:end]
    chunk_text = _tokenizer.decode(chunk_tokens)
    # → Store this chunk
    start += step   # Slide window forward
```

---

## Interview Questions on Chunking

**Q: Why not just use the full document as one embedding?**
> A: Embedding models have token limits (512 for BGE-small). Also, one embedding for a 3,000-token doc loses precision — a 384-token chunk is semantically tighter and retrieves more accurately.

**Q: How did you choose 384 tokens?**
> A: BGE-small's limit is 512. 384 gives 128 tokens of headroom to prevent silent truncation. I also compared 256/384/512 on a benchmark set and 384 gave the best precision@3.

**Q: What happens to documents shorter than your chunk size?**
> A: They produce exactly 1 chunk — the loop runs once and exits immediately.
