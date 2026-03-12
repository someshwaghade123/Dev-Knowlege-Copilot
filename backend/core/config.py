"""
backend/core/config.py
─────────────────────
Centralised settings loaded from the .env file.

WHY pydantic-settings?
  - Type-safe: every value is validated at startup (e.g. CHUNK_SIZE must be int)
  - One place to change config — nothing is hardcoded anywhere else
  - Works perfectly with Docker / Render env vars (no code change needed)

INTERVIEW TIP:
  "I used pydantic-settings so the app fails fast at startup if a required
   env var is missing, rather than crashing mid-request."
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_api_key: str
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemini-2.0-flash-lite-001"

    # ── External APIs ────────────────────────────────────────────────────────
    openai_api_key: str | None = None
    cohere_api_key: str | None = None

    # ── Embeddings ───────────────────────────────────────────────────────────
    embed_model: str = "embed-english-v3.0"
    embed_dimension: int = 1024   # Must match Cohere embed-english-v3.0

    # ── Storage ──────────────────────────────────────────────────────────────
    # We resolve paths as absolute to avoid issues with running from different dirs
    base_path: Path = Path(__file__).parent.parent.parent
    faiss_index_path: str = str(base_path / "data" / "faiss_index.bin")
    faiss_metadata_db: str = str(base_path / "data" / "metadata.db")
    bm25_index_path: str = str(base_path / "data" / "bm25_index.pkl")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 384       # Tokens per chunk
    chunk_overlap: int = 64     # Overlapping tokens between chunks

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5              # Chunks to retrieve per query

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    server_port: int = 8001     # Moved to 8001 due to frequent WinError 10013
    rate_limit: str = "20/minute"

    # ── Caching ──────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.95

    # Tells pydantic-settings to read from .env file automatically
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Singleton — import this everywhere instead of constructing Settings() again
settings = Settings()
