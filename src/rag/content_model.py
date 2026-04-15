"""
Backward-compatibility shim for moved hybrid logic.

The active implementation now lives in `src.hybrid_model`.
Keep this module so older imports from `src.rag.content_model` continue to work.
"""

from src.hybrid_model import compute_content_scores, compute_to_read_boosts, rerank_recommendations_hybrid

__all__ = [
    "compute_content_scores",
    "compute_to_read_boosts",
    "rerank_recommendations_hybrid",
]

