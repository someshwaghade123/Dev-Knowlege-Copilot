from fastapi import APIRouter
from backend.db.models import get_latency_metrics
from backend.api.v1.schemas import MetricsResponse, LatencyBreakdown

router = APIRouter()

@router.get("/metrics", response_model=MetricsResponse)
async def get_performance_metrics():
    """
    Return aggregate performance statistics from query logs.
    """
    data = get_latency_metrics()
    summary = data["summary"]
    
    return MetricsResponse(
        total_queries=summary["total_queries"] or 0,
        total_tokens=summary["total_tokens"] or 0,
        avg=LatencyBreakdown(
            total=int(summary["avg_total"] or 0),
            embed=int(summary["avg_embed"] or 0),
            retrieval=int(summary["avg_retrieval"] or 0),
            llm=int(summary["avg_llm"] or 0),
            rerank=int(summary["avg_rerank"] or 0),
            fact=int(summary["avg_fact"] or 0)
        ),
        p50=LatencyBreakdown(**data["p50"]),
        p95=LatencyBreakdown(**data["p95"])
    )
