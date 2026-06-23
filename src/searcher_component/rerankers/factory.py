"""Config-driven reranker factory.

Maps a pipeline reranker *name* (e.g. ``"rankllama"``, ``"report_aware"``) plus a
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

    if post_retrieval_reranker in ("batched_reranker", "reranker_selector"):
        cfg = kwargs.get("batched_reranker_config")
        if cfg is None:
            return None
        from .batched_reranker import setup_batched_reranker
        return setup_batched_reranker(
            reranker_model=cfg["reranker_model"],
            max_chars_per_document=cfg["max_chars_per_document"],
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

    if post_retrieval_reranker in ("report_aware", "report_aware_iter"):
        cfg = kwargs.get("report_aware_reranker_config")
        if cfg is None:
            return None
        from .report_aware_reranker import setup_report_aware_reranker
        return setup_report_aware_reranker(**cfg)

    if post_retrieval_reranker == "query_aware_doc_rewriting":
        cfg = kwargs.get("query_aware_doc_rewriting_config")
        if cfg is None:
            return None
        # Build the inner reranker first
        inner_reranker_name = cfg.get("rewrite_rerank_model", "rankllama")
        inner_reranker = build_reranker_from_config(inner_reranker_name, kwargs, device=device)
        if inner_reranker is None:
            return None
        from .query_aware_doc_rewriting_reranker import setup_query_aware_doc_rewriting_reranker
        return setup_query_aware_doc_rewriting_reranker(
            rewrite_model=cfg.get("rewrite_model", "gpt-4.1-mini"),
            inner_reranker=inner_reranker,
            max_chars=cfg.get("max_chars", 1500),
            max_words=cfg.get("max_words", 150),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 500),
            verbose=cfg.get("verbose", False),
        )

    return None
