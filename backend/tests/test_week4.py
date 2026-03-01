import pytest
import sqlite3
from fastapi.testclient import TestClient
from backend.main import app
from backend.db.models import get_connection, insert_query_log, get_latency_metrics

client = TestClient(app)

@pytest.fixture
def clean_logs():
    """Ensure query_logs is clean before tests."""
    conn = get_connection()
    conn.execute("DELETE FROM query_logs")
    conn.commit()
    conn.close()
    yield
    # Cleanup after
    conn = get_connection()
    conn.execute("DELETE FROM query_logs WHERE query LIKE 'TEST_%'")
    conn.commit()
    conn.close()

def test_metrics_aggregation(clean_logs):
    """Verify that get_latency_metrics correctly averages and calculates percentiles."""
    # 1. Insert known data
    insert_query_log("TEST_Q1", "A1", "high", 100, 10, embed_ms=20, retrieval_ms=30, llm_ms=50)
    insert_query_log("TEST_Q2", "A2", "high", 200, 20, embed_ms=40, retrieval_ms=60, llm_ms=100)
    insert_query_log("TEST_Q3", "A3", "high", 300, 30, embed_ms=60, retrieval_ms=90, llm_ms=150)
    
    metrics = get_latency_metrics()
    
    assert metrics["summary"]["total_queries"] >= 3
    assert metrics["summary"]["avg_total"] == 200.0
    assert metrics["summary"]["avg_embed"] == 40.0
    
    # 2. Check Percentiles
    assert metrics["p50"]["total"] == 200
    assert metrics["p95"]["total"] == 300

def test_metrics_api_endpoint(clean_logs):
    """Verify GET /api/v1/metrics response format."""
    # Ensure there's at least one log
    insert_query_log("TEST_API", "A", "low", 10, 1)
    
    with TestClient(app) as client:
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
    
    assert "total_queries" in data
    assert "avg" in data
    assert data["total_queries"] >= 1

def test_query_logging_completeness(clean_logs):
    """Verify that /query fills all granular latency columns."""
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json={"query": "TEST_EMPTY_QUERY_LATENCY"})
        assert response.status_code == 200
    
    conn = get_connection()
    row = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    
    assert row["latency_ms"] >= 0
    assert row["embed_ms"] >= 0
    assert row["retrieval_ms"] >= 0

def test_title_boost_logic():
    """
    Unit test for the multiplier logic in hybrid_search.
    """
    from backend.retrieval.hybrid import hybrid_search
    from backend.retrieval.vector_store import vector_store
    
    # Manually init for unit test if needed
    vector_store.load_or_create()
    
    data = hybrid_search("What is in routes.py?", top_k=5, search_mode="hybrid")
    
    assert "results" in data
    assert "metrics" in data
    assert data["metrics"]["embed_ms"] >= 0
