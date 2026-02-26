import asyncio
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.db.models import get_all_documents, insert_document, insert_chunk, get_all_chunks, delete_document
from backend.api.v1.schemas import DocumentsListResponse
from backend.ingestion.chunker import chunk_document, chunk_code
from backend.ingestion.parsers import extract_text, RICH_FORMATS, CODE_FORMATS, ALL_SUPPORTED
from backend.ingestion.embedder import embed_texts
from backend.retrieval.vector_store import vector_store
from backend.retrieval.bm25_store import bm25_store
from backend.cache.cache_manager import cache_manager

from pydantic import BaseModel

router = APIRouter()

class TextUploadRequest(BaseModel):
    text: str
    name: str = "pasted_document.txt"

async def _index_content(content: str, doc_name: str, file_ext: str, loop) -> dict:
    """Shared indexing logic for both file upload and text upload endpoints."""
    if file_ext in CODE_FORMATS:
        chunks = chunk_code(content, doc_name, doc_title=doc_name)
    else:
        chunks = chunk_document(content, doc_title=doc_name)

    if not chunks:
        raise ValueError("Could not extract any text chunks from the content.")

    chunk_texts = [c.text for c in chunks]

    # 1. Embed and Add to Vector Store FIRST to get assigned IDs
    embed_start = time.perf_counter()
    embeddings = await loop.run_in_executor(None, embed_texts, chunk_texts)
    print(f"[Upload] Embedded {len(chunks)} chunks in {time.perf_counter() - embed_start:.2f}s")

    assigned_ids = vector_store.add_embeddings(embeddings)
    vector_store.save()

    # 2. Record in Database using the ACTUAL assigned IDs
    doc_id = await loop.run_in_executor(None, insert_document, doc_name, None, doc_name)

    for i, (chunk, faiss_id) in enumerate(zip(chunks, assigned_ids)):
        await loop.run_in_executor(None, insert_chunk, doc_id, faiss_id, i, chunk.text, chunk.token_count)

    # 3. Refresh BM25 index
    all_chunks = await loop.run_in_executor(None, get_all_chunks)
    await loop.run_in_executor(None, bm25_store.build_index, all_chunks)

    # 4. Invalidate Semantic Cache
    cache_manager.clear()

    return {"message": "Indexed successfully.", "file_name": doc_name, "chunks_processed": len(chunks)}


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
    Dynamically upload, chunk, embed, and index a document.
    Supports rich formats (PDF, DOCX, PPTX) and plain text/code files.
    """
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if file_ext not in ALL_SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Supported: {', '.join(sorted(ALL_SUPPORTED))}"
        )

    content_bytes = await file.read()

    try:
        # 1a. Text extraction — rich formats use parsers, others decode as UTF-8
        if file_ext in RICH_FORMATS:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, extract_text, content_bytes, file_ext)
        else:
            try:
                content = content_bytes.decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode file: {e}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))

    loop = asyncio.get_running_loop()
    try:
        return await _index_content(content, file.filename, file_ext, loop)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[Upload Error] {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")


@router.post("/documents/upload-text")
async def upload_text(body: TextUploadRequest):
    """
    Index raw text pasted directly from the mobile app.
    Accepts a JSON body with 'text' and optional 'name'.
    This avoids the need to create a fake file URI in React Native.
    """
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text body cannot be empty.")

    safe_name = body.name.strip() or "pasted_document.txt"
    if not any(safe_name.endswith(ext) for ext in [".txt", ".md", ".py", ".js"]):
        safe_name = safe_name + ".txt"

    file_ext = "." + safe_name.rsplit(".", 1)[-1].lower()
    loop = asyncio.get_running_loop()
    try:
        return await _index_content(body.text, safe_name, file_ext, loop)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[UploadText Error] {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index text: {e}")


@router.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: int):
    """
    Delete a document from the knowledge base.
    This removes records from SQLite, deletes vectors from FAISS, and refreshes BM25.
    """
    loop = asyncio.get_running_loop()
    
    try:
        # 1. Delete from database and get FAISS IDs
        faiss_ids = await loop.run_in_executor(None, delete_document, doc_id)
        
        if not faiss_ids:
            # Document might have 0 chunks or not exist
            return {"message": "Document record removed (0 chunks)."}

        # 2. Remove from FAISS index
        await loop.run_in_executor(None, vector_store.remove_ids, faiss_ids)
        await loop.run_in_executor(None, vector_store.save, True)  # Force save even if smaller

        # 3. Refresh BM25 index
        all_chunks = await loop.run_in_executor(None, get_all_chunks)
        await loop.run_in_executor(None, bm25_store.build_index, all_chunks)

        # 4. Invalidate Semantic Cache
        cache_manager.clear()

        return {"message": f"Document {doc_id} and {len(faiss_ids)} chunks deleted successfully."}

    except Exception as e:
        print(f"[Delete Error] {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")
