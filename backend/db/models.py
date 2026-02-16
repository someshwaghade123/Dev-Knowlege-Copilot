"""
backend/db/models.py
─────────────────────
SQLite database setup for storing chunk metadata.

WHY SQLite?
  - Zero setup: no server needed, just a .db file
  - Perfect for Week 1 — swap to Postgres later with one connection string change
  - We store metadata (not vectors) here: title, URL, chunk text preview

SCHEMA:
  documents  — one row per ingested document
  chunks     — one row per chunk, linked to a document
               The faiss_id column maps chunk → FAISS index position

INTERVIEW TIP:
  "I separated vector storage (FAISS) from metadata storage (SQLite).
   FAISS gives fast ANN search; SQLite gives structured queries on metadata
   like filtering by source URL or document title."
"""

import sqlite3
from pathlib import Path
from backend.core.config import settings


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite metadata database."""
    Path(settings.faiss_metadata_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.faiss_metadata_db)
    conn.row_factory = sqlite3.Row   # Rows behave like dicts — easier to work with
    return conn


def init_db() -> None:
    """
    Create tables if they don't exist yet.
    Called once at application startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── documents table ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            source_url  TEXT,               -- GitHub URL, file path, etc.
            file_name   TEXT NOT NULL,
            ingested_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── chunks table ─────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER NOT NULL REFERENCES documents(id),
            faiss_id    INTEGER NOT NULL UNIQUE,  -- Position in FAISS index
            chunk_index INTEGER NOT NULL,         -- 0-based position within doc
            text        TEXT NOT NULL,            -- Full chunk text
            text_preview TEXT,                    -- First 200 chars for citations
            token_count INTEGER,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── query_logs table (Week 2) ──────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            answer      TEXT NOT NULL,
            confidence  TEXT,
            latency_ms  INTEGER,
            tokens_used INTEGER,
            timestamp   TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Tables initialised.")


def insert_document(title: str, source_url: str, file_name: str) -> int:
    """Insert a document record and return its auto-generated ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (title, source_url, file_name) VALUES (?, ?, ?)",
        (title, source_url, file_name),
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def insert_chunk(doc_id: int, faiss_id: int, chunk_index: int,
                 text: str, token_count: int) -> None:
    """Insert a single chunk record linked to a document."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chunks
           (doc_id, faiss_id, chunk_index, text, text_preview, token_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, faiss_id, chunk_index, text, text[:200], token_count),
    )
    conn.commit()
    conn.close()


def get_chunks_by_faiss_ids(faiss_ids: list[int]) -> list[dict]:
    """
    Given a list of FAISS result IDs, return full chunk metadata from SQLite.
    This is how we turn a vector search result into a citation.
    """
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(faiss_ids))
    cursor.execute(
        f"""SELECT c.id, c.faiss_id, c.text, c.text_preview, c.token_count,
                   d.title, d.source_url, d.file_name
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.faiss_id IN ({placeholders})""",
        faiss_ids,
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_all_documents() -> list[dict]:
    """Return all indexed documents — used by the /documents endpoint."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def insert_query_log(query: str, answer: str, confidence: str,
                     latency_ms: int, tokens_used: int) -> None:
    """Store a record of a RAG query and its performance metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO query_logs
           (query, answer, confidence, latency_ms, tokens_used)
           VALUES (?, ?, ?, ?, ?)""",
        (query, answer, confidence, latency_ms, tokens_used),
    )
    conn.commit()
    conn.close()
