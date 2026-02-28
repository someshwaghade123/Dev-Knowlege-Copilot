"""
backend/main.py
────────────────
FastAPI application entry point.

This file:
  1. Creates the FastAPI app instance
  2. Runs startup logic (DB init, FAISS index load)
  3. Registers all API routes
  4. Configures CORS so the mobile app can call the backend

STARTUP SEQUENCE:
  App boots → init_db() → vector_store.load_or_create() → Ready to serve

CORS (Cross-Origin Resource Sharing):
  The mobile app (Expo) runs on a different origin than the backend.
  Without CORS headers, the browser/mobile runtime will block responses.
  We allow all origins for development; lock this down in production.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.db.models import init_db
from backend.retrieval.vector_store import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup code before the app begins serving requests.
    Using lifespan (not deprecated @app.on_event) — modern FastAPI style.
    """
    print("[Startup] Initialising database...")
    init_db()

    print("[Startup] Loading FAISS vector index...")
    vector_store.load_or_create()

    print("[Startup] ✅ Application ready.")
    yield   # App serves requests between yield and shutdown

    print("[Shutdown] Saving FAISS index to disk...")
    vector_store.save()
    print("[Shutdown] Done.")


app = FastAPI(
    title="Developer Knowledge Copilot",
    description=(
        "A production-grade RAG system that indexes technical documentation "
        "and answers developer queries with citations and confidence scores."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← Restrict to your app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")
# All endpoints are now under /api/v1 prefix:
#   GET  /api/v1/health
#   POST /api/v1/query
#   GET  /api/v1/documents


# ── Run directly ──────────────────────────────────────────────────────────────
# Usage: python backend/main.py
# Or:    uvicorn backend.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
