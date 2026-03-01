"""
backend/ingestion/chunker.py
─────────────────────────────
Splits a long document into smaller, overlapping chunks.

WHY DO WE CHUNK?
  LLMs and embedding models have a maximum token limit (e.g. 512 tokens for
  BGE-small). A long README might have 3,000 tokens. We can't embed all of it
  at once — so we split it into overlapping windows.

CHUNKING STRATEGY — Fixed-size with overlap:
  ┌──────────────────────────────────────────────────────┐
  │  Full Document (e.g. 1,200 tokens)                   │
  │  ┌─────────────┐                                     │
  │  │  Chunk 0    │  (0–384 tokens)                     │
  │  │        ┌────┴────────┐                            │
  │  │        │   Chunk 1   │  (320–704 tokens)          │
  │  │        │        ┌────┴────────┐                   │
  │  │        │        │   Chunk 2   │  (640–1024 tokens)│
  └──┴────────┴─────────────────────┴───────────────────┘
              ◄──────►
              overlap (64 tokens)

  Overlap ensures that a sentence split across two chunks is still
  represented in at least one complete chunk for retrieval.

INTERVIEW TIP:
  "I chose 384 tokens because BGE-small has a 512-token limit. Leaving head
   room avoids silent truncation. I tested 256/384/512 and found 384 gave
   the best precision@3 on my benchmark set."
"""

import re
import tiktoken
from dataclasses import dataclass
from pathlib import Path
from backend.core.config import settings


# We use the cl100k_base tokenizer (same as GPT-4) as a universal counter.
# This lets us measure tokens accurately regardless of which LLM we use.
_tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single piece of a document, ready for embedding."""
    text: str           # The actual text content
    token_count: int    # How many tokens this chunk uses
    chunk_index: int    # 0-based position within the source document
    doc_title: str      # Propagated from parent doc for easy citation


def _clean_text(text: str) -> str:
    """
    Basic text normalisation before chunking.
      - Collapse multiple blank lines into one
      - Strip leading/trailing whitespace
    We keep markdown syntax (##, `, etc.) because it can help the LLM
    understand structure during generation.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ newlines → 2 newlines
    return text.strip()


def chunk_document(text: str, doc_title: str = "") -> list[Chunk]:
    """
    Split a document into overlapping token-level chunks.

    Args:
        text:       Raw document text (markdown, plain text, etc.)
        doc_title:  Used to label each chunk for citation purposes

    Returns:
        List of Chunk objects ready for embedding
    """
    text = _clean_text(text)

    # Tokenise once — work at the token level for precise sizing
    token_ids = _tokenizer.encode(text)
    total_tokens = len(token_ids)

    chunk_size = settings.chunk_size       # e.g. 384
    overlap = settings.chunk_overlap       # e.g. 64
    step = chunk_size - overlap            # Slide window by this amount each step

    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)

        # Decode token IDs back to text for this window
        chunk_tokens = token_ids[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)

        chunks.append(Chunk(
            text=chunk_text,
            token_count=len(chunk_tokens),
            chunk_index=chunk_index,
            doc_title=doc_title,
        ))

        if end == total_tokens:
            break   # Reached end of document

        start += step
        chunk_index += 1

    return chunks


def _get_code_blocks(text: str, language: str) -> list[str]:
    """
    Experimental: Identify logical blocks (classes, functions) in code.
    This helps keep related logic together in a single chunk.
    """
    blocks = []
    lines = text.split("\n")
    current_block = []

    # Simple regex-based markers for block starts
    if language == "python":
        # Python: starts with 'def ' or 'class ' at the beginning of a line
        pattern = re.compile(r"^(def|class)\s+\w+")
    else:
        # JS/TS: 'function', 'class', 'const/let ... = (...) =>'
        pattern = re.compile(r"^(export\s+)?(class|function|const|let)\s+\w+")

    for line in lines:
        if pattern.match(line) and current_block:
            # New block started, save previous
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def chunk_code(text: str, file_name: str, doc_title: str = "") -> list[Chunk]:
    """
    Intelligent code chunking that respects structural boundaries.
    Falls back to token-based chunking if blocks are too large or small.
    """
    ext = Path(file_name).suffix.lower()
    language = "python" if ext == ".py" else "javascript"

    # 1. Identify raw structural blocks
    raw_blocks = _get_code_blocks(text, language)

    # 2. Refine blocks (merge small ones, split huge ones)
    final_chunks: list[Chunk] = []
    chunk_index = 0

    current_text = ""
    current_tokens = 0

    for block in raw_blocks:
        block_tokens = len(_tokenizer.encode(block))

        # If block is huge (> max chunk size), split it using standard chunker
        if block_tokens > settings.chunk_size:
            # If we had pending text, save it first
            if current_text:
                final_chunks.append(Chunk(current_text, current_tokens, chunk_index, doc_title))
                chunk_index += 1
                current_text = ""
                current_tokens = 0

            # Split huge block
            sub_chunks = chunk_document(block, doc_title)
            for sc in sub_chunks:
                sc.chunk_index = chunk_index
                final_chunks.append(sc)
                chunk_index += 1
            continue

        # If adding this block exceeds limit, save current and start new
        if current_tokens + block_tokens > settings.chunk_size:
            final_chunks.append(Chunk(current_text, current_tokens, chunk_index, doc_title))
            chunk_index += 1
            current_text = block
            current_tokens = block_tokens
        else:
            # Merge block
            current_text = (current_text + "\n\n" + block).strip() if current_text else block
            current_tokens += block_tokens

    # Final wrap up
    if current_text:
        final_chunks.append(Chunk(current_text, current_tokens, chunk_index, doc_title))

    return final_chunks


def chunk_documents(docs: list[dict]) -> list[tuple[dict, list[Chunk]]]:
    """
    Chunk a list of documents. Detects if a document is code and uses
    specialised structural chunking if so.
    """
    result = []
    code_extensions = {".py", ".js", ".ts", ".tsx"}

    for doc in docs:
        file_name = doc.get("file_name", "")
        ext = Path(file_name).suffix.lower()

        if ext in code_extensions:
            chunks = chunk_code(doc["text"], file_name, doc_title=doc.get("title", ""))
            print(f"  [Chunker] (CODE) '{doc['title']}' -> {len(chunks)} chunks")
        else:
            chunks = chunk_document(doc["text"], doc_title=doc.get("title", ""))
            print(f"  [Chunker] (TEXT) '{doc['title']}' -> {len(chunks)} chunks")

        result.append((doc, chunks))
    return result
