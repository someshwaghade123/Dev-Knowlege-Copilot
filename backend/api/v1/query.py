import time
import asyncio
from fastapi import APIRouter, HTTPException, Request
from backend.ingestion.embedder import embed_query
from backend.retrieval.hybrid import hybrid_search
from backend.generation.llm import generate_answer, verify_factuality
from backend.db.models import get_chunks_by_faiss_ids, insert_query_log
from backend.retrieval.reranker import reranker
from backend.scoring.engine import compute_confidence
from backend.cache.cache_manager import cache_manager
from backend.api.dependencies import limiter
from backend.core.config import settings
from backend.api.v1.schemas import QueryRequest, QueryResponse, Citation

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit)
async def query_documents(body: QueryRequest, request: Request):
    """
    Main RAG endpoint with robust error logging and rate limiting.
    """
    wall_start = time.perf_counter()
    metrics = {}
    loop = asyncio.get_running_loop()

    try:
        if not body.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        embed_start = time.perf_counter()
        query_vector = await loop.run_in_executor(None, embed_query, body.query)
        metrics["embed_ms"] = int((time.perf_counter() - embed_start) * 1000)

        cached_data = cache_manager.get(query_vector)
        if cached_data:
            cached_data["latency_ms"] = int((time.perf_counter() - wall_start) * 1000)
            return QueryResponse(**cached_data)

        search_data = await loop.run_in_executor(
            None, 
            hybrid_search, 
            body.query, 
            body.top_k, 
            body.search_mode,
            query_vector
        )
        search_results = search_data["results"]
        metrics = search_data["metrics"]

        filtered = [r for r in search_results if r["score"] >= body.min_score]
        if not filtered:
            return QueryResponse(
                answer="I could not find relevant information for your query.",
                citations=[],
                confidence="low",
                latency_ms=int((time.perf_counter() - wall_start) * 1000),
                tokens_used=0,
            )

        faiss_ids = [r["faiss_id"] for r in filtered]
        raw_chunks = await loop.run_in_executor(None, get_chunks_by_faiss_ids, faiss_ids)

        if not raw_chunks:
             raise Exception(f"No metadata found for FAISS IDs: {faiss_ids}")

        chunks_map = {c["faiss_id"]: c for c in raw_chunks}
        chunks = [chunks_map[fid] for fid in faiss_ids if fid in chunks_map]

        rerank_start = time.perf_counter()
        reranked = await loop.run_in_executor(
            None, 
            reranker.rerank, 
            body.query, 
            chunks, 
            body.top_n_rerank
        )
        metrics["rerank_ms"] = int((time.perf_counter() - rerank_start) * 1000)
        
        confidence = compute_confidence(
            [c.get("rerank_score", -10.0) for c in reranked], 
            mode="rerank"
        )

        CONF_LEVELS = {"low": 0, "medium": 1, "high": 2}
        if body.min_confidence:
            target = CONF_LEVELS.get(body.min_confidence.lower(), 0)
            actual = CONF_LEVELS.get(confidence, 0)
            if actual < target:
                return QueryResponse(
                    answer=f"I don't have enough confidence ({confidence}) to answer this accurately based on your requirements.",
                    citations=[],
                    confidence=confidence,
                    latency_ms=int((time.perf_counter() - wall_start) * 1000),
                    tokens_used=0
                )

        chunks = reranked

        if body.bypass_llm:
            llm_result = {
                "answer": "[Bypass Mode] Retrieval only. No answer generated.",
                "tokens_used": 0
            }
            metrics["llm_ms"] = 0
            metrics["fact_ms"] = 0
            fact_check = {"is_grounded": True, "reasoning": "Bypass mode"}
        else:
            llm_start = time.perf_counter()
            llm_result = await generate_answer(body.query, chunks)
            metrics["llm_ms"] = int((time.perf_counter() - llm_start) * 1000)

            fact_start = time.perf_counter()
            fact_check = await verify_factuality(body.query, llm_result["answer"], chunks)
            metrics["fact_ms"] = int((time.perf_counter() - fact_start) * 1000)
            
            if not fact_check["is_grounded"]:
                confidence = "low"
                print(f"[Factuality Guard] Hallucination detected for query: {body.query}")
                print(f"Reasoning: {fact_check['reasoning']}")

        if body.bypass_llm:
            used_indices = list(range(1, len(chunks) + 1))
        else:
            from backend.generation.llm import extract_citation_indices
            used_indices = extract_citation_indices(llm_result["answer"])

        all_citations = [
            Citation(
                title=c["title"],
                source_url=c.get("source_url"),
                text_preview=c.get("text_preview", c["text"][:200]),
            )
            for c in chunks
        ]
        
        citations = [all_citations[i-1] for i in used_indices if 0 < i <= len(all_citations)]

        total_latency_ms = int((time.perf_counter() - wall_start) * 1000)

        try:
            insert_query_log(
                query=body.query,
                answer=llm_result.get("answer") or "N/A",
                confidence=confidence,
                latency_ms=total_latency_ms,
                tokens_used=llm_result.get("tokens_used", 0),
                embed_ms=metrics.get("embed_ms", 0),
                retrieval_ms=metrics.get("retrieval_ms", 0),
                rerank_ms=metrics.get("rerank_ms", 0),
                fact_ms=metrics.get("fact_ms", 0),
                llm_ms=metrics.get("llm_ms", 0),
            )
        except Exception as log_err:
            print(f"[Analytics] Non-critical: failed to log query: {log_err}")

        response = QueryResponse(
            answer=llm_result.get("answer") or "I could not generate an answer.",
            citations=citations,
            confidence=confidence,
            latency_ms=total_latency_ms,
            tokens_used=llm_result.get("tokens_used", 0),
        )

        if not body.bypass_llm:
            cache_manager.set(body.query, query_vector, response.model_dump())

        return response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e) or repr(e)
        print(f"[API Error] /query failure: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
