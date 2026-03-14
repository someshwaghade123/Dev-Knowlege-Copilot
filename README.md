# Lumen

A production-grade Retrieval-Augmented Generation (RAG) system built to index technical documentation and provide precise, grounded answers with citations and confidence scores.

## Features

*   **Hybrid Search**: Combines Dense Vector Search (Cohere/FAISS) and Sparse Keyword Search (BM25) with Reciprocal Rank Fusion (RRF) for optimal document retrieval.
*   **Semantic Caching**: Re-uses prior LLM answers for semantically similar queries (>95% similarity) to reduce latency from ~500ms down to <10ms and save API costs.
*   **Confidence Scoring & Factuality Guards**: Uses Cohere API reranking and secondary LLM checks to verify hallucination limits. 
*   **Concurrency & Rate Limiting**: Built with FastAPI, leveraging asyncio.run_in_executor for non-blocking ML operations, protecting endpoints with slowapi (20 req/min).
*   **Premium Mobile UI**: Built in React Native (Expo) featuring query skeletons, dynamic metric dashboard pills (Latency, Tokens, Confidence), and robust network retry logic. 

## Tech Stack

*   **Backend**: Python, FastAPI, Pydantic
*   **AI/ML**: Cohere (Embeddings & Reranking), rank_bm25, FAISS (CPU)
*   **Database**: SQLite (Metadata & Query Logs)
*   **Mobile**: React Native, Expo, React-Native-Markdown-Display
*   **DevOps**: Docker, Docker Compose, Locust (Load Testing)

## Quickstart

### Prerequisites
*   Docker & Docker Compose installed.
*   An OpenRouter API Key and a Cohere API Key.

### 1. Clone & Configure
```bash
git clone https://github.com/someshwaghade123/Dev-Knowlege-Copilot.git
cd Dev-Knowlege-Copilot
```
Create a .env file in the root directory:
```env
LLM_API_KEY=sk-or-v1-...
COHERE_API_KEY=...
APP_ENV=production
LOG_LEVEL=INFO
SERVER_PORT=8001
RATE_LIMIT=20/minute
```

### 2. Run the Backend
```bash
docker-compose up --build -d
```
The FastAPI server will be available at http://localhost:8001.
You can view the auto-generated Swagger docs at http://localhost:8001/docs.

### 3. Run the Mobile App
```bash
cd mobile
npm install
npx expo start
```
Scan the QR code with the Expo Go app on your phone, or press a or i to run on a local Android/iOS emulator.

## API Reference

### POST /api/v1/query
Main RAG querying endpoint. Includes rate limiting.
**Request Body**:
```json
{
  "query": "How do I configure CORS in FastAPI?",
  "top_k": 5,
  "search_mode": "hybrid",
  "min_confidence": "medium"
}
```
**Response**:
```json
{
  "answer": "To configure CORS in FastAPI...",
  "citations": [
    {
      "title": "FastAPI Deployment",
      "source_url": "https://fastapi.tiangolo.com/tutorial/cors/",
      "text_preview": "CORS or Cross-Origin Resource Sharing..."
    }
  ],
  "confidence": "high",
  "latency_ms": 482,
  "tokens_used": 150
}
```

### GET /api/v1/health
Readiness probe returning the status of the DB and Vector Store.

### GET /api/v1/metrics
Returns aggregate performance statistics across all logged queries (p50/p95 latency, total tokens used, and specific pipeline timings).

## Benchmarking & Load Testing
The system was load-tested using Locust with 50 concurrent users, achieving 0% failure rate and a median latency of 4ms (powered by the semantic cache).

## License
MIT License
