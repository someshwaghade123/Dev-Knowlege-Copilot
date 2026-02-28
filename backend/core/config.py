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
    llm_model: str = "mistralai/mistral-7b-instruct"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dimension: int = 384   # Must match the chosen model's output size

    # ── Storage ──────────────────────────────────────────────────────────────
    # We resolve paths as absolute to avoid issues with running from different dirs
    base_path: Path = Path(__file__).parent.parent.parent
    faiss_index_path: str = str(base_path / "data" / "faiss_index.bin")
    faiss_metadata_db: str = str(base_path / "data" / "metadata.db")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 384       # Tokens per chunk
    chunk_overlap: int = 64     # Overlapping tokens between chunks

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5              # Chunks to retrieve per query

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # Tells pydantic-settings to read from .env file automatically
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


# Singleton — import this everywhere instead of constructing Settings() again
settings = Settings()
