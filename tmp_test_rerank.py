import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastembed import TextCrossEncoder
import numpy as np

def test_rerank():
    model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    query = "How to install FastAPI?"
    passages = [
        "To install FastAPI, run: pip install fastapi",
        "The weather is nice today.",
        "Python is a programming language."
    ]
    
    print(f"Query: {query}")
    print(f"Passages: {passages}")
    
    results = list(model.rerank(query, passages))
    for res in results:
        print(f"Score: {res.score}, Index: {res.index}, Document: {res.document}")

if __name__ == "__main__":
    test_rerank()
