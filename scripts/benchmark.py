import os
import sys
import json
import time
import requests
import numpy as np
from datetime import datetime

# Force UTF-8 for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000/api/v1"
DATASET_PATH = "data/golden_dataset.json"
REPORT_PATH = "docs/week6/benchmarks.md"

def run_benchmarks():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    results = []
    latencies = []
    precision_k = []
    recall_k = []
    k = 3

    print(f"--- Starting Benchmark on {len(dataset)} queries ---")

    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item["expected_docs"]
        
        start = time.perf_counter()
        try:
            # We use bypass_llm=True to measure pure retrieval quality without LLM costs/limits
            response = requests.post(f"{BASE_URL}/query", json={
                "query": query, 
                "top_k": 5, 
                "bypass_llm": True
            })
            duration = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                retrieved_titles = [c["title"].lower() for c in data["citations"]]
                found = [exp for exp in expected if any(exp.lower() in rt for rt in retrieved_titles)]
                
                prec = len(found) / min(k, len(retrieved_titles)) if retrieved_titles else 0
                rec = len(found) / len(expected) if expected else 0
                
                precision_k.append(prec)
                recall_k.append(rec)
                latencies.append(duration)
                
                results.append({
                    "query": query,
                    "latency_ms": duration,
                    "precision": prec,
                    "recall": rec,
                    "success": True
                })
                print(f"[{i+1}/{len(dataset)}] [OK] {query[:30]}... ({duration:.0f}ms)")
            else:
                error_msg = response.text
                print(f"[{i+1}/{len(dataset)}] [FAIL] {query[:30]}... (Error {response.status_code}: {error_msg})")
                results.append({"query": query, "success": False, "error": error_msg})

        except Exception as e:
            print(f"[{i+1}/{len(dataset)}] [ERROR] {query[:30]}... ({str(e)})")
            results.append({"query": query, "success": False})

        # Short delay between queries
        time.sleep(1)

    # Summary Statistics
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    avg_precision = np.mean(precision_k) if precision_k else 0
    avg_recall = np.mean(recall_k) if recall_k else 0

    generate_report(dataset, results, avg_latency, p95_latency, avg_precision, avg_recall)

def generate_report(dataset, results, avg_lat, p95_lat, avg_prec, avg_rec):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    
    with open(REPORT_PATH, "w", encoding='utf-8') as f:
        f.write("# Retrieval Benchmark Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Queries**: {len(dataset)}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Metric | Result |\n")
        f.write("|--------|--------|\n")
        f.write(f"| Avg Latency | {avg_lat:.2f} ms |\n")
        f.write(f"| P95 Latency | {p95_lat:.2f} ms |\n")
        f.write(f"| Avg Precision@3 | {avg_prec:.2%}| \n")
        f.write(f"| Avg Recall@3 | {avg_rec:.2%} |\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| Query | Latency | Prec | Rec | Status |\n")
        f.write("|-------|---------|------|-----|--------|\n")
        for res in results:
            status = "✅" if res["success"] else "❌"
            latency = f"{res['latency_ms']:.0f}ms" if res["success"] else "-"
            prec = f"{res['precision']:.1f}" if res["success"] else "-"
            rec = f"{res['recall']:.1f}" if res["success"] else "-"
            f.write(f"| {res['query']} | {latency} | {prec} | {rec} | {status} |\n")
                
    print(f"\nBenchmark complete! Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_benchmarks()
