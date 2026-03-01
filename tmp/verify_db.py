import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.models import get_all_documents

def test():
    print("Testing get_all_documents with aggregation...")
    docs = get_all_documents()
    print(f"Total documents found: {len(docs)}")
    for doc in docs[:3]:
        print(f"\n- Title: {doc['title']}")
        print(f"  Chunk Count: {doc['chunk_count']}")
        print(f"  Total Tokens: {doc['total_tokens']}")
        print(f"  Ingested At: {doc['ingested_at']}")

if __name__ == "__main__":
    test()
