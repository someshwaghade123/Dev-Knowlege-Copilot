from pydantic import BaseModel
from backend.core.config import settings

class QueryRequest(BaseModel):
    query: str
    top_k: int = settings.top_k
    min_score: float = 0.0
    search_mode: str = "hybrid"
    min_confidence: str | None = None
    top_n_rerank: int = 5
    bypass_llm: bool = False

class Citation(BaseModel):
    title: str
    source_url: str | None
    text_preview: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    latency_ms: int
    tokens_used: int

class DocumentMetadata(BaseModel):
    id: int
    title: str
    source_url: str | None
    file_name: str
    ingested_at: str
    chunk_count: int
    total_tokens: int | None

class DocumentsListResponse(BaseModel):
    count: int
    documents: list[DocumentMetadata]

class LatencyBreakdown(BaseModel):
    total: int
    embed: int
    retrieval: int
    llm: int
    rerank: int = 0
    fact: int = 0

class MetricsResponse(BaseModel):
    total_queries: int
    total_tokens: int
    avg: LatencyBreakdown
    p50: LatencyBreakdown
    p95: LatencyBreakdown
