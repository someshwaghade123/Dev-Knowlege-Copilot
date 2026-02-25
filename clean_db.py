import sys
import sqlite3
from pathlib import Path

db_path = Path("C:/Users/hp/Documents/NITK/PROJECT/data/metadata.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find the max valid faiss_id (should be 13, since we have 14 base vectors)
cursor.execute("SELECT COUNT(*) FROM chunks")
total_chunks = cursor.fetchone()[0]
print(f"Total chunks in DB before cleanup: {total_chunks}")

# Delete any chunks that didn't make it into FAISS
# We know the base system had 14 chunks. Let's delete anything >= 14
cursor.execute("DELETE FROM chunks WHERE faiss_id >= 14")
deleted = cursor.rowcount
print(f"Deleted {deleted} orphaned chunks.")

# Also delete any documents that have no chunks
cursor.execute("DELETE FROM documents WHERE id NOT IN (SELECT doc_id FROM chunks)")
docs_deleted = cursor.rowcount
print(f"Deleted {docs_deleted} orphaned documents.")

conn.commit()
conn.close()
