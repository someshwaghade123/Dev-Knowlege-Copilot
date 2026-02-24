from fastapi import APIRouter, HTTPException
from backend.retrieval.vector_store import vector_store

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Readiness probe.
    Verifies that the database and vector store are accessible.
    """
    health = {
        "status": "ok",
        "checks": {
            "database": "down",
            "vector_store": "down"
        }
    }
    
    try:
        from backend.db.models import get_connection
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
        health["checks"]["database"] = "up"
    except Exception as e:
        health["status"] = "error"
        health["error_db"] = str(e)

    try:
        if vector_store._index is not None:
            health["checks"]["vector_store"] = "up"
            health["indexed_vectors"] = vector_store._index.ntotal
        else:
            health["status"] = "error"
            health["checks"]["vector_store"] = "not_loaded"
    except Exception as e:
        health["status"] = "error"
        health["error_vs"] = str(e)

    if health["status"] == "error":
        raise HTTPException(status_code=503, detail=health)
        
    return health
