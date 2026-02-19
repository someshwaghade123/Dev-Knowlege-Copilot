from fastapi.testclient import TestClient
from backend.main import app
import json

def verify_metrics_api():
    with TestClient(app) as client:
        print("Calling /api/v1/metrics...")
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Total Queries: {data['total_queries']}")
        print(f"Total Tokens:  {data['total_tokens']}")
        
        print("\nLATENCY (Average):")
        print(f"  Total:     {data['avg']['total']}ms")
        print(f"  Embed:     {data['avg']['embed']}ms")
        print(f"  Retrieval: {data['avg']['retrieval']}ms")
        print(f"  LLM:       {data['avg']['llm']}ms")
        
        print("\nLATENCY (P95):")
        print(f"  Total:     {data['p95']['total']}ms")
        print(f"  Embed:     {data['p95']['embed']}ms")
        print(f"  Retrieval: {data['p95']['retrieval']}ms")
        print(f"  LLM:       {data['p95']['llm']}ms")
        
        # Basic integrity checks
        assert data['total_queries'] > 0
        assert data['p95']['total'] >= data['p50']['total']
        print("\nMetrics API verification successful!")

if __name__ == "__main__":
    verify_metrics_api()
