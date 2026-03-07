from fastapi import APIRouter
from backend.api.v1.health import router as health_router
from backend.api.v1.query import router as query_router
from backend.api.v1.documents import router as documents_router
from backend.api.v1.metrics import router as metrics_router

router = APIRouter()

router.include_router(health_router, tags=["Health"])
router.include_router(query_router, tags=["Search & RAG"])
router.include_router(documents_router, tags=["Documents"])
router.include_router(metrics_router, tags=["Metrics"])
