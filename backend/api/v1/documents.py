from fastapi import APIRouter
from backend.db.models import get_all_documents
from backend.api.v1.schemas import DocumentsListResponse

router = APIRouter()

@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents():
    """
    Return all indexed documents with statistics.
    Useful for the mobile app's 'Browse' screen.
    """
    docs = get_all_documents()
    return DocumentsListResponse(count=len(docs), documents=docs)
