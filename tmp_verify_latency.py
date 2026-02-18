import time
from fastapi.testclient import TestClient
from backend.main import app
from backend.db.models import get_connection

def verify_latency_logging():
    with TestClient(app) as client:
        print("Running test query...")
        response = client.post("/api/v1/query", json={"query": "What is Hybrid Search?"})
        assert response.status_code == 200
        
        print("Checking database for latency metrics...")
        conn = get_connection()
        row = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        
        print(f"Log ID: {row['id']}")
        print(f"Query: {row['query']}")
        print(f"Total Latency: {row['latency_ms']}ms")
        print(f"  - Embed: {row['embed_ms']}ms")
        print(f"  - Retrieval: {row['retrieval_ms']}ms")
        print(f"  - LLM: {row['llm_ms']}ms")
        
        assert row['embed_ms'] >= 0
        assert row['retrieval_ms'] >= 0
        assert row['llm_ms'] >= 0
        print("\nVerification successful! All metrics captured.")

if __name__ == "__main__":
    verify_latency_logging()
