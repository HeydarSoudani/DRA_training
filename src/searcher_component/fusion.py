"""Compatibility shim — function-based fusion API backed by the new fusion classes.

The fusion logic lives in the agentic_retrieval_research.fusion package (class-based).
This module re-exports the same function-level interface so that existing callers
(searcher_component/__init__.py, parallel_dm_agent.py, retrieval_evaluator.py, etc.)
continue to work without modification.
"""

from typing import Any, Dict, List, Optional

from agentic_retrieval_research.fusion.interleaving import InterleavingFusion, NestedInterleavingFusion
from agentic_retrieval_research.fusion.rrf import ReciprocalRankFusion
from agentic_retrieval_research.fusion.concatenation import SimpleConcatenation


# ---------------------------------------------------------------------------
# Function-level API (mirrors the original fusion.py)
# ---------------------------------------------------------------------------

def interleaving_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    window: int = 1,
) -> List[Dict[str, Any]]:
    """Block-interleaved fusion across multiple ranked lists."""
    return InterleavingFusion(window=window).fuse(ranked_lists)


def nested_interleaving_fusion(
    nested_ranked_lists: List[List[List[Dict[str, Any]]]],
    window: int = 1,
) -> List[Dict[str, Any]]:
    """Two-level interleaving fusion (within + across Search calls)."""
    return NestedInterleavingFusion(window=window).fuse(nested_ranked_lists)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    return ReciprocalRankFusion(k=k).fuse(ranked_lists)


def simple_concatenation(
    ranked_lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Deduplicative concatenation of multiple ranked lists."""
    return SimpleConcatenation().fuse(ranked_lists)


FUSION_METHODS: Dict[str, Any] = {
    "interleaving": interleaving_fusion,
    "rrf": reciprocal_rank_fusion,
    "concatenation": simple_concatenation,
}


def fuse_results(
    ranked_lists: List[List[Dict[str, Any]]],
    fusion_method: str = "interleaving",
    rrf_k: int = 60,
    interleaving_window: int = 3,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists with the named method."""
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0]
    if fusion_method == "interleaving":
        return interleaving_fusion(ranked_lists, window=interleaving_window)
    if fusion_method == "rrf":
        return reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    if fusion_method == "concatenation":
        return simple_concatenation(ranked_lists)
    raise ValueError(f"Unknown fusion method: {fusion_method!r}")


def fuse_retrieval_results(
    iterations: List[List[Dict[str, Any]]],
    fusion_method: str = "interleaving",
    rrf_k: int = 60,
    interleaving_window: Optional[int] = 3,
) -> List[Dict[str, Any]]:
    """High-level wrapper used by agents and evaluators."""
    window = interleaving_window if interleaving_window is not None else 3
    return fuse_results(
        iterations,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
        interleaving_window=window,
    )
