import asyncio
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.db.models import get_all_documents, insert_document, insert_chunk, get_all_chunks
from backend.api.v1.schemas import DocumentsListResponse
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import embed_texts
from backend.retrieval.vector_store import vector_store
from backend.retrieval.bm25_store import bm25_store

router = APIRouter()

@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents():
    """
    Return all indexed documents with statistics.
    Useful for the mobile app's 'Browse' screen.
    """
    docs = get_all_documents()
    return DocumentsListResponse(count=len(docs), documents=docs)

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Dynamically upload, chunk, embed, and index a new text/markdown file.
    """
    if not file.filename.endswith((".md", ".txt")):
        raise HTTPException(status_code=400, detail="Only .md and .txt files are supported for now.")
    
    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode file. Ensure it is valid UTF-8 text. Error: {e}")

    loop = asyncio.get_running_loop()

    try:
        # 1. Chunking
        chunks = chunk_document(content, file.filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text chunks from the file.")
            
        chunk_texts = [c.text for c in chunks]

        # 2. Embedding
        embed_start = time.perf_counter()
        embeddings = await loop.run_in_executor(None, embed_texts, chunk_texts)
        print(f"[Upload] Embedded {len(chunks)} chunks in {time.perf_counter() - embed_start:.2f}s")

        # 3. Database & FAISS Insertion
        doc_id = await loop.run_in_executor(None, insert_document, file.filename, None, file.filename)
        
        # We need the current FAISS ID offset to map chunks correctly
        start_faiss_id = vector_store._index.ntotal if vector_store._index else 0
        
        for i, chunk in enumerate(chunks):
            faiss_id = start_faiss_id + i
            await loop.run_in_executor(
                None, 
                insert_chunk, 
                doc_id, 
                faiss_id, 
                i, 
                chunk.text, 
                chunk.token_count
            )

        # Update FAISS
        vector_store.add_embeddings(embeddings)
        vector_store.save()

        # 4. Update BM25
        # The easiest way for now is to fully rebuild the BM25 index from the DB,
        # ensuring it stays perfectly in sync.
        all_chunks = await loop.run_in_executor(None, get_all_chunks)
        await loop.run_in_executor(None, bm25_store.build_index, all_chunks)

        return {
            "message": "File successfully uploaded and indexed.",
            "file_name": file.filename,
            "chunks_processed": len(chunks)
        }

    except Exception as e:
        print(f"[Upload Error] {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")
