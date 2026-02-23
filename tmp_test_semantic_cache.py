import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_semantic_cache():
    print("--- Testing Semantic Cache & Latency FIX ---")
    
    # Query 1: Original
    q1 = "How do I install FastAPI?"
    print(f"\n1. Sending original query: '{q1}'")
    r1 = requests.post(f"{BASE_URL}/query", json={"query": q1, "bypass_llm": True})
    res1 = r1.json()
    print(f"   Latency: {res1['latency_ms']}ms")
    
    # Query 2: Paraphrased (Minor changes)
    q2 = "fastapi installation guide"
    print(f"\n2. Sending paraphrased query: '{q2}'")
    t0 = time.perf_counter()
    r2 = requests.post(f"{BASE_URL}/query", json={"query": q2, "bypass_llm": True})
    res2 = r2.json()
    actual_wall_time = int((time.perf_counter() - t0) * 1000)
    
    print(f"   Response Answer: {res2['answer']}")
    print(f"   Reported Latency: {res2['latency_ms']}ms")
    print(f"   Actual Wall Time: {actual_wall_time}ms")
    
    if res2['latency_ms'] < 50 and "Bypass Mode" in res2['answer']:
        print("\n✅ SUCCESS: Semantic cache hit and latency reporting fixed!")
    else:
        print("\n❌ FAILURE: Cache miss or stagnant latency detected.")

if __name__ == "__main__":
    try:
        test_semantic_cache()
    except Exception as e:
        print(f"Error: {e}")
