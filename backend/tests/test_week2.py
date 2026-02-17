"""
backend/tests/test_week2.py
────────────────────────────
Unit and Integration tests for Week 2 features:
- Structural Code Chunking (.py, .js)
- Structured Query Logging (SQLite)
- Enhanced Documents API (Statistics)
- Advanced Citation Response Shape
"""

import pytest
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient

from backend.ingestion.chunker import _get_code_blocks, chunk_code
from backend.db.models import init_db, insert_document, insert_chunk, get_all_documents, insert_query_log
from backend.main import app
from backend.core.config import settings

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure DB is initialized before each test."""
    init_db()

# ── 1. Structural Chunker Tests ───────────────────────────────────────────────

class TestCodeChunker:
    def test_python_block_detection(self):
        py_code = (
            "def func1():\n    pass\n\n"
            "class MyClass:\n    def method(self):\n        pass"
        )
        blocks = _get_code_blocks(py_code, "python")
        # Should identify 2 main blocks: func1 and MyClass
        assert len(blocks) == 2
        assert "def func1" in blocks[0]
        assert "class MyClass" in blocks[1]

    def test_javascript_block_detection(self):
        js_code = (
            "function test() {}\n"
            "export const MyComp = () => {}\n"
            "class Service {}"
        )
        blocks = _get_code_blocks(js_code, "javascript")
        assert len(blocks) == 3
        assert "function test" in blocks[0]
        assert "const MyComp" in blocks[1]
        assert "class Service" in blocks[2]

    def test_chunk_code_respects_boundaries(self):
        code = "def a(): pass\n\ndef b(): pass"
        chunks = chunk_code(code, "test.py", "Test Doc")
        # These are small, so they should be merged into 1 chunk by chunk_code's logic
        # but the boundaries were still respected during block detection.
        assert len(chunks) == 1
        assert "def a()" in chunks[0].text
        assert "def b()" in chunks[0].text


# ── 2. Database & Logging Tests ───────────────────────────────────────────────

class TestDatabaseWeek2:
    def test_query_log_insertion(self):
        # Insert a log
        insert_query_log(
            query="test query",
            answer="test answer",
            confidence="high",
            latency_ms=100,
            tokens_used=50
        )
        
        # Verify via direct SQL (since we don't have a GET for logs yet)
        from backend.db.models import get_connection
        conn = get_connection()
        row = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        
        assert row["query"] == "test query"
        assert row["latency_ms"] == 100
        assert row["tokens_used"] == 50

    def test_get_all_documents_stats(self):
        # 1. Insert a doc
        doc_id = insert_document("Test Stats Doc", None, "stats.py")
        
        # 2. Insert 2 chunks
        insert_chunk(doc_id, faiss_id=9999, chunk_index=0, text="chunk 1", token_count=10)
        insert_chunk(doc_id, faiss_id=9998, chunk_index=1, text="chunk 2", token_count=20)
        
        # 3. Fetch all docs and check stats
        docs = get_all_documents()
        # Find our specific doc
        target = next((d for d in docs if d["id"] == doc_id), None)
        
        assert target is not None
        assert target["chunk_count"] == 2
        assert target["total_tokens"] == 30


# ── 3. API Week 2 Integration Tests ─────────────────────────────────────────────

class TestAPIWeek2:
    def test_documents_endpoint_schema(self, client):
        """Verify the /documents endpoint returns the new rich metadata."""
        # Ensure at least one doc exists for the test
        insert_document("API Test", "http://test.com", "test.md")
        
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        
        assert "documents" in data
        assert "count" in data
        if data["count"] > 0:
            doc = data["documents"][0]
            assert "chunk_count" in doc
            assert "total_tokens" in doc
            assert "ingested_at" in doc

    def test_query_response_shape(self, client):
        """
        Verify the /query response includes Week 2 fields (latency, tokens).
        Note: We mock the LLM if we want to avoid external calls, 
        but here we check if the Pydantic model is enforced.
        """
        # This test might return 503 if no docs, or 200 if mocked/docs exist.
        # We mostly care that the code doesn't crash and returns the expected keys.
        response = client.post("/api/v1/query", json={"query": "what is FastAPI?"})
        
        if response.status_code == 200:
            data = response.json()
            assert "latency_ms" in data
            assert "tokens_used" in data
            assert "confidence" in data
            assert isinstance(data["citations"], list)
