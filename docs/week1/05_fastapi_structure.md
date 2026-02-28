# 05 — FastAPI Project Structure Explained

## Why FastAPI?

| Feature | Why It Matters |
|---------|---------------|
| Async support | Handles concurrent requests without blocking (crucial for LLM calls) |
| Pydantic validation | Request/response schemas enforced automatically |
| Auto-generated /docs | Swagger UI appears immediately — no setup |
| Type hints | Python type hints = self-documenting code |
| Performance | On par with NodeJS — one of fastest Python frameworks |

---

## Our Router Structure

```python
# main.py registers all routes under /api/v1 prefix
app.include_router(router, prefix="/api/v1")

# Results in:
GET  /api/v1/health
POST /api/v1/query
GET  /api/v1/documents
```

The `/api/v1` prefix is a versioning strategy — when we change the API in Week 4+,
we can add `/api/v2` without breaking the mobile app still using v1.

---

## Pydantic Request/Response Models

```python
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    latency_ms: int
    tokens_used: int
```

**Benefits**:
- Invalid requests are rejected with a useful 422 error (not a crash)
- Response shape is guaranteed — mobile app never gets unexpected null fields
- `/docs` endpoint shows the full schema automatically

---

## Lifespan Events (Startup/Shutdown)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                       # Create tables if missing
    vector_store.load_or_create()   # Load FAISS index from disk
    yield                           # App serves requests here
    vector_store.save()             # Save index on graceful shutdown
```

**Why lifespan instead of `@app.on_event("startup")`?**
The `on_event` decorator is deprecated in FastAPI 0.95+. Lifespan is the modern
pattern and ensures shutdown code also runs (important for saving the FAISS index).

---

## CORS Middleware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow any origin in development
    ...
)
```

Without this, the mobile app (running on a different origin) would receive a CORS
error and the response would be blocked. In production, replace `"*"` with your
actual app domain.

---

## Error Handling Pattern

```python
if not request.query.strip():
    raise HTTPException(status_code=400, detail="Query cannot be empty.")

if vector_store._next_id == 0:
    raise HTTPException(status_code=503, detail="No documents indexed yet.")
```

**HTTP status codes we use**:
| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Successful query |
| 400 | Bad Request | Empty query |
| 422 | Unprocessable Entity | Pydantic validation failure (auto) |
| 503 | Service Unavailable | No documents indexed |
| 500 | Internal Server Error | Unexpected crash |

---

## Async vs Sync

Our `/query` route is `async` because it `await`s the LLM API call:

```python
@router.post("/query")
async def query_documents(request: QueryRequest):
    ...
    llm_result = await generate_answer(...)   # Async HTTP call
```

While waiting for the LLM response (~400ms), FastAPI can handle other incoming
requests. If we used `def` (sync), that 400ms would block the entire server.

---

## Interview Questions on FastAPI Structure

**Q: Why did you use Pydantic models instead of plain dicts?**
> A: Pydantic validates input at the edge — before it touches our business logic. A malformed request triggers a 422 with a clear error message, not a cryptic KeyError 3 layers deep. It also auto-generates the /docs Swagger UI, which I use for testing during development.

**Q: How does your API handle concurrent requests?**
> A: The LLM call is async (`await httpx.AsyncClient.post()`), so while waiting for the LLM response, uvicorn's event loop serves other requests. I also load FAISS at startup into memory — searches are in-memory and fast. For heavier concurrency I'd add a process pool for CPU-bound embedding tasks.
