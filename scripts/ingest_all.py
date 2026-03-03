"""
scripts/ingest_all.py
───────────────────────
Master ingestion script.
Wipes the database and indexes clean, then re-ingests everything in one pass.

USAGE:
  python scripts/ingest_all.py

It will index:
  - backend/   (all .py source files)
  - data/sample_docs/   (starter .md documentation)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.models import init_db, get_all_chunks, get_connection
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.embedder import embed_texts
from backend.retrieval.vector_store import vector_store
from backend.retrieval.bm25_store import bm25_store
from backend.core.config import settings


def load_files_from_folder(folder: Path) -> list[dict]:
    """Read all supported files from a directory."""
    docs = []
    extensions = {".md", ".txt", ".rst", ".py", ".js", ".ts"}
    
    # ── Folder & File Exclusions ──────────────────────────────────────────────
    # User requested to mention and exclude all project folders except sample data
    EXCLUDE_FOLDERS = {
        "backend", "mobile", "scripts", "docs", "venv", 
        ".git", ".github", ".pytest_cache", "node_modules", "__pycache__",
        "bench" # Exclude other data subfolders
    }
    EXCLUDE_FILES = {"main.py", "__init__.py"}

    # ── URL Mapping ────────────────────────────────────────────────────────
    # Maps specific filenames to their official documentation websites
    FILE_URL_MAPPING = {
        "fastapi_getting_started.md": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "docker_reference.md": "https://docs.docker.com/get-started/",
        "vector_databases_guide.md": "https://www.pinecone.io/learn/vector-database/"
    }
    
    # Fallback: Maps local folder prefixes to base documentation URLs (e.g. GitHub)
    SOURCE_URL_MAPPING = {
        "data/sample_docs": "https://github.com/someshwaghade123/Dev-Knowlege-Copilot/blob/main/data/sample_docs"
    }

    for file_path in sorted(folder.rglob("*")):
        if file_path.suffix.lower() not in extensions or not file_path.is_file():
            continue
            
        # Check if any part of the path is in the excluded folders list
        if any(p in EXCLUDE_FOLDERS for p in file_path.parts):
            continue
            
        # Ensure we are ONLY indexing things inside sample_docs if starting from data
        if "sample_docs" not in file_path.parts:
            continue

        # Skip specific infrastructure files
        if file_path.name in EXCLUDE_FILES:
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")

        if len(text.strip()) < 50:
            continue

        # Derive source URL
        source_url = None
        
        # Priority 1: Specific File Mapping (Official Docs)
        if file_path.name in FILE_URL_MAPPING:
            source_url = FILE_URL_MAPPING[file_path.name]
        
        # Priority 2: Folder Mapping (e.g. GitHub)
        if not source_url:
            rel_path = file_path.as_posix() # e.g. "data/sample_docs/file.md"
            for prefix, base_url in SOURCE_URL_MAPPING.items():
                if rel_path.startswith(prefix):
                    source_url = f"{base_url}/{file_path.name}"
                    break
        
        # Priority 3: Local file URL
        if not source_url:
            source_url = f"file:///{file_path.absolute().as_posix()}"

        title = f"{file_path.parent.name}/{file_path.name}"
        docs.append({
            "title": title,
            "text": text,
            "source_url": source_url,
            "file_name": file_path.name,
        })
        print(f"  [Read] {title} -> {source_url}")



    return docs


def insert_document(conn, title, source_url, file_name):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (title, source_url, file_name) VALUES (?, ?, ?)",
        (title, source_url, file_name)
    )
    conn.commit()
    return cursor.lastrowid


def insert_chunk(conn, doc_id, faiss_id, chunk_index, text, token_count):
    preview = text[:300]
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (doc_id, faiss_id, chunk_index, text, text_preview, token_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, faiss_id, chunk_index, text, preview, token_count)
    )
    conn.commit()


def main():
    # Strictly focusing on data/sample_docs as requested
    source_folders = [Path("data")]


    # 1. Wipe old data
    print("\n[Reset] Wiping old index files...")
    data_dir = Path("data")
    for fname in ["metadata.db", "faiss_index.bin", "bm25_index.pkl"]:
        p = data_dir / fname
        if p.exists():
            p.unlink()
            print(f"  Deleted: {p}")

    # 2. Init fresh DB and FAISS
    print("\n[Init] Creating fresh database and vector store...")
    init_db()
    vector_store.load_or_create()

    # 3. Load all docs
    all_docs = []
    for folder in source_folders:
        print(f"\n[Load] Reading from '{folder}'...")
        all_docs.extend(load_files_from_folder(folder))

    print(f"\n  => Total: {len(all_docs)} documents loaded")

    # 4. Chunk
    print("\n[Chunk] Splitting documents...")
    doc_chunk_pairs = chunk_documents(all_docs)
    total_chunks = sum(len(c) for _, c in doc_chunk_pairs)
    print(f"  => {total_chunks} chunks across {len(all_docs)} docs")

    # 5. Embed & Store
    print("\n[Index] Embedding and indexing all chunks...")
    conn = get_connection()
    for doc, chunks in doc_chunk_pairs:
        doc_id = insert_document(conn, doc["title"], doc.get("source_url"), doc["file_name"])

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        faiss_ids = vector_store.add_embeddings(embeddings)

        for chunk, fid in zip(chunks, faiss_ids):
            insert_chunk(conn, doc_id, fid, chunk.chunk_index, chunk.text, chunk.token_count)

        print(f"  OK: '{doc['title']}' => {len(chunks)} chunks")

    conn.close()

    # 6. Save FAISS
    print("\n[Save] Writing FAISS index to disk...")
    vector_store.save()

    # 7. Rebuild BM25
    print("\n[BM25] Building keyword index...")
    all_chunks = get_all_chunks()
    bm25_store.build_index(all_chunks)
    bm25_store.save()

    # 8. Final Summary
    print(f"\n{'='*50}")
    print(f"Ingestion Complete!")
    print(f"  Documents : {len(all_docs)}")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Vectors   : {vector_store._index.ntotal}")
    print(f"  Index     : {settings.faiss_index_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
