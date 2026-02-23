import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_cache():
    query = "How do I configure CORS in FastAPI?"
    payload = {"query": query, "top_k": 5}
    
    print(f"--- Running Query 1 (Cache Miss) ---")
    start = time.perf_counter()
    resp1 = requests.post(f"{BASE_URL}/query", json=payload)
    latency1 = (time.perf_counter() - start) * 1000
    
    if resp1.status_code == 200:
        print(f"Success! Latency: {latency1:.2f}ms")
        # print(f"Answer: {resp1.json()['answer'][:100]}...")
    else:
        print(f"Error: {resp1.status_code} - {resp1.text}")
        return

    print(f"\n--- Running Query 2 (Cache Hit) ---")
    start = time.perf_counter()
    resp2 = requests.post(f"{BASE_URL}/query", json=payload)
    latency2 = (time.perf_counter() - start) * 1000
    
    if resp2.status_code == 200:
        print(f"Success! Latency: {latency2:.2f}ms")
        print(f"Speedup: {latency1/latency2:.1f}x")
        
        # Verify answers are identical
        if resp1.json()["answer"] == resp2.json()["answer"]:
            print("Verified: Answer is consistent.")
    else:
        print(f"Error: {resp2.status_code}")

if __name__ == "__main__":
    test_cache()
