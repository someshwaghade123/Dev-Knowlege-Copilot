import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.bm25_store import bm25_store
from backend.retrieval.vector_store import vector_store

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_bm25_search_functionality():
    """Verify BM25 can find exact keywords that Vector might miss."""
    # Build a small dummy index
    chunks = [
        {"faiss_id": 101, "text": "The quick brown fox jumps over the lazy dog."},
        {"faiss_id": 102, "text": "Python is a versatile programming language."},
        {"faiss_id": 103, "text": "BM25 handles exact keyword matching very well."}
    ]
    bm25_store.build_index(chunks)
    
    # Search for a keyword
    results = bm25_store.search("versatile", top_k=1)
    assert len(results) > 0
    assert results[0]["faiss_id"] == 102
    
    results = bm25_store.search("BM25", top_k=1)
    assert results[0]["faiss_id"] == 103

def test_hybrid_search_logic():
    """Verify RRF merger logic."""
    # This assumes vector_store is loaded with at least some vectors 
    # from the previous ingestion.
    vector_store.load_or_create()
    
    query = "hybrid search"
    results = hybrid_search(query, top_k=5, search_mode="hybrid")
    
    assert len(results) > 0
    # Every result should have a score and a faiss_id
    for res in results:
        assert "faiss_id" in res
        assert "score" in res
        assert isinstance(res["score"], float)

def test_api_search_mode_parameter(client):
    """Verify the /query API respects the search_mode parameter."""
    query = "test query"
    
    # Test Vector mode
    resp_v = client.post("/api/v1/query", json={"query": query, "search_mode": "vector"})
    assert resp_v.status_code == 200
    
    # Test BM25 mode
    resp_b = client.post("/api/v1/query", json={"query": query, "search_mode": "bm25"})
    assert resp_b.status_code == 200
    
    # Test Hybrid mode
    resp_h = client.post("/api/v1/query", json={"query": query, "search_mode": "hybrid"})
    assert resp_h.status_code == 200

def test_invalid_search_mode(client):
    """Verify API handles (or ignores) invalid search modes gracefully."""
    # The API currently uses a string but we should check if it defaults to hybrid
    resp = client.post("/api/v1/query", json={"query": "test", "search_mode": "invalid_mode"})
    assert resp.status_code == 200  # Should default or handle gracefully
    # Based on hybrid_search implementation, it defaults to RRF if not vector/bm25
