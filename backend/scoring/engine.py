"""
backend/scoring/engine.py
───────────────────────────
Centralized logic for search scoring and confidence estimation.
"""

def compute_confidence(scores: list[float], mode: str = "hybrid") -> str:
    """
    Calculate confidence level based on retrieval scores.
    
    Confidence Levels:
      - HIGH:   Very strong match in top results
      - MEDIUM: Likely relevant but less certain
      - LOW:    Weak match or no results
    """
    if not scores:
        return "low"
        
    top_score = max(scores)
    
    # Thresholds for Cross-Encoder (rerank_score)
    # Calibrated for Cohere Rerank v3 (0-1 range).
    if mode == "rerank":
        if top_score > 0.7: return "high"
        if top_score > 0.3: return "medium"
        return "low"
        
    # Thresholds for Hybrid (RRF)
    if mode == "hybrid":
        if top_score > 0.03: return "high"
        if top_score > 0.01: return "medium"
        return "low"

    # Thresholds for Vector only
    if mode == "vector":
        if top_score > 0.80: return "high"
        if top_score > 0.60: return "medium"
        return "low"

    return "low"
