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
            embed_ms    INTEGER,
            retrieval_ms INTEGER,
            llm_ms      INTEGER,
            tokens_used INTEGER,
            timestamp   TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Migration for existing Week 2 databases ──
    try:
        cursor.execute("ALTER TABLE query_logs ADD COLUMN embed_ms INTEGER")
        cursor.execute("ALTER TABLE query_logs ADD COLUMN retrieval_ms INTEGER")
        cursor.execute("ALTER TABLE query_logs ADD COLUMN llm_ms INTEGER")
    except sqlite3.OperationalError:
        # Columns likely already exist
        pass

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


def get_chunk_titles(faiss_ids: list[int]) -> dict[int, str]:
    """
    Efficiently fetch only the document titles for a set of faiss_ids.
    Used for the title-boosting heuristic during hybrid search.
    """
    if not faiss_ids:
        return {}
        
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(faiss_ids))
    cursor.execute(
        f"""SELECT c.faiss_id, d.title
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.faiss_id IN ({placeholders})""",
        faiss_ids,
    )
    mapping = {row["faiss_id"]: row["title"] for row in cursor.fetchall()}
    conn.close()
    return mapping


def get_all_documents() -> list[dict]:
    """
    Return all indexed documents with statistics (chunk count, total tokens).
    Uses a LEFT JOIN to ensure documents with 0 chunks (if any) are still shown.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.*, 
               COUNT(c.id) as chunk_count,
               SUM(c.token_count) as total_tokens
        FROM documents d
        LEFT JOIN chunks c ON d.id = c.doc_id
        GROUP BY d.id
        ORDER BY d.ingested_at DESC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def insert_query_log(query: str, answer: str, confidence: str,
                      latency_ms: int, tokens_used: int,
                      embed_ms: int = 0, retrieval_ms: int = 0, llm_ms: int = 0) -> None:
    """Store a record of a RAG query and its performance metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO query_logs
           (query, answer, confidence, latency_ms, tokens_used, 
            embed_ms, retrieval_ms, llm_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (query, answer, confidence, latency_ms, tokens_used, 
         embed_ms, retrieval_ms, llm_ms),
    )
    conn.commit()
    conn.close()


def get_all_chunks() -> list[dict]:
    """Retrieve all chunks with their faiss_id and text — used to build BM25."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT faiss_id, text FROM chunks")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_latency_metrics() -> dict:
    """
    Calculate aggregate performance stats from query_logs.
    Includes Average, P50 (Median), and P95 latency.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # ── 1. Basic Counts & Averages ───────────────────────────────────────────
    cursor.execute("""
        SELECT 
            COUNT(*) as total_queries,
            SUM(tokens_used) as total_tokens,
            AVG(latency_ms) as avg_total,
            AVG(embed_ms) as avg_embed,
            AVG(retrieval_ms) as avg_retrieval,
            AVG(llm_ms) as avg_llm
        FROM query_logs
    """)
    summary = dict(cursor.fetchone())
    
    # ── 2. Percentile Calculation (P50, P95) ──────────────────────────────────
    # SQLite doesn't have NTILE/PERCENTILE by default, so we sort in Python
    def get_percentile(column: str, p: float) -> int:
        cursor.execute(f"SELECT {column} FROM query_logs WHERE {column} IS NOT NULL ORDER BY {column} ASC")
        values = [r[0] for r in cursor.fetchall()]
        if not values:
            return 0
        idx = int(p * len(values))
        return values[min(idx, len(values) - 1)]

    metrics = {
        "summary": summary,
        "p50": {
            "total": get_percentile("latency_ms", 0.50),
            "embed": get_percentile("embed_ms", 0.50),
            "retrieval": get_percentile("retrieval_ms", 0.50),
            "llm": get_percentile("llm_ms", 0.50),
        },
        "p95": {
            "total": get_percentile("latency_ms", 0.95),
            "embed": get_percentile("embed_ms", 0.95),
            "retrieval": get_percentile("retrieval_ms", 0.95),
            "llm": get_percentile("llm_ms", 0.95),
        }
    }
    
    conn.close()
    return metrics
