import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def ping_query():
    payload = {"query": "What is FastAPI?", "top_k": 5}
    try:
        r = requests.post(f"{BASE_URL}/query", json=payload)
        print(f"Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Response: {r.text}")
        else:
            data = r.json()
            print(f"Success! Answer length: {len(data.get('answer', ''))}")
            print(f"Citations: {len(data.get('citations', []))}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    ping_query()
