"""
scripts/ingest_docs.py
───────────────────────
CLI script to ingest markdown/text documents into the vector index.

USAGE:
  # Ingest all .md files from a folder
  python scripts/ingest_docs.py --source data/sample_docs

  # Ingest a single file
  python scripts/ingest_docs.py --source data/sample_docs/fastapi_readme.md

WHAT THIS SCRIPT DOES:
  For each document:
    1. Read the file text
    2. Split into overlapping chunks (chunker.py)
    3. Embed all chunks in a batch (embedder.py)
    4. Add embeddings to FAISS index (vector_store.py)
    5. Save chunk metadata (text, title, URL) to SQLite (db/models.py)
    6. Save the updated FAISS index to disk

  This is a one-time (or periodic) operation. Once indexed, the app
  serves queries instantly from the persisted index.
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root: python scripts/ingest_docs.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.models import init_db, insert_document, insert_chunk
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.embedder import embed_texts
from backend.retrieval.vector_store import vector_store
from backend.core.config import settings


def load_documents_from_folder(folder: Path) -> list[dict]:
    """
    Read all .md and .txt files from a folder.
    Returns list of {"title", "text", "source_url", "file_name"} dicts.
    """
    docs = []
    extensions = {".md", ".txt", ".rst", ".py", ".js", ".ts", ".tsx"}

    for file_path in sorted(folder.rglob("*")):
        if file_path.suffix.lower() in extensions and file_path.is_file():
            # Skip hidden files/folders (like .git, .expo)
            if any(part.startswith(".") for part in file_path.parts):
                continue

            # Skip node_modules and venv
            if "node_modules" in file_path.parts or "venv" in file_path.parts:
                continue

            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if len(text.strip()) < 50:  # Code files can be shorter than MD
                continue

            # Title: Include parent folder to differentiate (e.g., "api/routes.py")
            title = f"{file_path.parent.name}/{file_path.name}"
            
            docs.append({
                "title": title,
                "text": text,
                "source_url": None,
                "file_name": file_path.name,
            })
            print(f"  [Read] {title} ({len(text):,} chars)")

    return docs


def ingest(source: str) -> None:
    """Main ingestion pipeline."""
    source_path = Path(source)

    if not source_path.exists():
        print(f"[Error] Path does not exist: {source_path}")
        sys.exit(1)

    # ── 1. Load files ────────────────────────────────────────────────────────
    print(f"\n[Ingest] Loading documents from: {source_path}")
    if source_path.is_dir():
        docs = load_documents_from_folder(source_path)
    else:
        # Single file
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        docs = [{
            "title": source_path.stem.replace("-", " ").title(),
            "text": text,
            "source_url": None,
            "file_name": source_path.name,
        }]

    if not docs:
        print("[Error] No documents found.")
        sys.exit(1)

    print(f"\n[Ingest] Loaded {len(docs)} document(s)")

    # ── 2. Init DB and FAISS ─────────────────────────────────────────────────
    print("\n[Ingest] Initialising database and vector store...")
    init_db()
    vector_store.load_or_create()

    # ── 3. Chunk all documents ───────────────────────────────────────────────
    print("\n[Ingest] Chunking documents...")
    doc_chunk_pairs = chunk_documents(docs)

    total_chunks = sum(len(chunks) for _, chunks in doc_chunk_pairs)
    print(f"   -> {total_chunks} total chunks across {len(docs)} docs")

    # ── 4. Embed and index ────────────────────────────────────────────────────
    print(f"\n[Ingest] Embedding {total_chunks} chunks with {settings.embed_model}...")

    for doc, chunks in doc_chunk_pairs:
        # Insert document record into SQLite
        doc_id = insert_document(
            title=doc["title"],
            source_url=doc.get("source_url"),
            file_name=doc["file_name"],
        )

        # Embed all chunks for this doc in one batch
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)   # Shape: (N, 384)

        # Add to FAISS and get assigned IDs
        faiss_ids = vector_store.add_embeddings(embeddings)

        # Store chunk metadata in SQLite
        for chunk, faiss_id in zip(chunks, faiss_ids):
            insert_chunk(
                doc_id=doc_id,
                faiss_id=faiss_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
            )

        print(f"   OK: '{doc['title']}' - {len(chunks)} chunks indexed")

    # ── 5. Save FAISS index to disk ──────────────────────────────────────────
    print(f"\n[Ingest] Saving vector index...")
    vector_store.save()

    print(f"\n[Ingest] Ingestion complete!")
    print(f"   Documents: {len(docs)}")
    print(f"   Chunks:    {total_chunks}")
    print(f"   Index:     {settings.faiss_index_path}")
    print(f"   DB:        {settings.faiss_metadata_db}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the knowledge base.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a folder or single file to ingest",
    )
    args = parser.parse_args()
    ingest(args.source)
