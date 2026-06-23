"""Config-driven reranker factory.

Maps a pipeline reranker *name* (e.g. ``"rankllama"``, ``"qwen3_reranker"``) plus a
config ``kwargs`` dict to a constructed reranker, dispatching to the matching
``setup_*`` helper in this package.  Kept here, next to the rerankers it builds,
so callers only depend on :mod:`searcher_component.rerankers`.
"""

from typing import Any, Dict, Optional


def build_reranker_from_config(
    post_retrieval_reranker: str,
    kwargs: Dict[str, Any],
    device: Optional[str] = None,
):
    """Instantiate and return a reranker, or None if reranking is disabled."""
    if post_retrieval_reranker == "null":
        return None

    if post_retrieval_reranker == "batched_reranker":
        cfg = kwargs.get("batched_reranker_config")
        if cfg is None:
            return None
        from .batched_reranker import setup_batched_reranker
        # The score-threshold selector is an optional component of the batched
        # reranker (enabled via cfg["enable_selector"]), not a separate reranker.
        return setup_batched_reranker(
            reranker_model=cfg["reranker_model"],
            max_chars_per_document=cfg["max_chars_per_document"],
            enable_selector=cfg.get("enable_selector", False),
            selector_score_threshold=cfg.get("selector_score_threshold", 3.0),
            selector_min_keep=cfg.get("selector_min_keep", 0),
        )

    if post_retrieval_reranker == "listwise":
        cfg = kwargs.get("listwise_reranker_config")
        if cfg is None:
            return None
        from .listwise_vllm_reranker import setup_listwise_vllm_reranker
        return setup_listwise_vllm_reranker(**cfg)

    if post_retrieval_reranker == "rankllama":
        cfg = kwargs.get("rankllama_reranker_config")
        if cfg is None:
            return None
        if device:
            cfg = dict(cfg)
            cfg["device"] = device
        from .rankllama_reranker import setup_rankllama_reranker
        return setup_rankllama_reranker(**cfg)

    if post_retrieval_reranker == "rank1":
        cfg = kwargs.get("rank1_reranker_config")
        if cfg is None:
            return None
        if device:
            cfg = dict(cfg)
            cfg["device"] = device
        from .rank1_reranker import setup_rank1_reranker
        return setup_rank1_reranker(**cfg)

    if post_retrieval_reranker == "rank_r1":
        cfg = kwargs.get("rank_r1_reranker_config")
        if cfg is None:
            return None
        from .rank_r1_reranker import setup_rank_r1_reranker
        return setup_rank_r1_reranker(**cfg)

    if post_retrieval_reranker == "qwen3_reranker":
        cfg = kwargs.get("qwen3_reranker_config")
        if cfg is None:
            return None
        from .qwen3_reranker import setup_qwen3_reranker
        return setup_qwen3_reranker(**cfg)

    if post_retrieval_reranker == "monot5":
        cfg = kwargs.get("monot5_reranker_config")
        if cfg is None:
            return None
        if device:
            cfg = dict(cfg)
            cfg["device"] = device
        from .monot5_reranker import setup_monot5_reranker
        return setup_monot5_reranker(**cfg)

    return None
