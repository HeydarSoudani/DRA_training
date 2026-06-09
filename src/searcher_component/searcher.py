"""Retrieval search tool — unified search primitive for all deep research agents.

Wraps a retriever with optional post-retrieval reranking, fusion, and
post-fusion reranking so that agents get a single call that handles
multi-query retrieval pipelines transparently.

Simple mode (single query, no reranker):
    tool.execute("climate change effects")
    → equivalent to retriever.retrieve(query) + normalize

Full mode (multiple queries, with rerankers):
    tool.execute(["climate change effects", "global warming impact"],
                 original_query="climate change")
    → retrieve each → (post-retrieval rerank per query) → fuse → (post-fusion rerank) → single ranked list

Configurable inputs (retrieval_input / post_fusion_reranker_input):
    Control what text is fed to the retriever and post-fusion reranker by
    composing trajectory context (original query, subqueries, reasoning)
    at each stage.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from agentic_retrieval_research.searcher_component.fusion import fuse_results
from agentic_retrieval_research.searcher_component import normalize_retrieval_response

logger = logging.getLogger(__name__)


class RetrievalSearchTool:
    """Unified search tool: retrieve → (post-retrieval rerank) → fuse → (post-fusion rerank).

    Parameters
    ----------
    retriever:
        Any retriever exposing ``.retrieve(query) -> list | dict``.
        A ``BaseRetriever`` subclass (local BM25/dense/SPLADE).
    post_retrieval_reranker:
        Optional reranker applied per sub-query immediately after retrieval.
        Reranks each sub-query's doc list individually before fusion.
    post_fusion_reranker:
        Optional reranker applied after fusion on the merged ranked list.
        Reranks the fused list against the original query.
    reranker:
        Legacy alias for ``post_fusion_reranker``. If both are provided,
        ``post_fusion_reranker`` takes precedence.
    top_k:
        Number of documents to retrieve per query (passed to retriever).
    rerank_top_k:
        Number of top documents to pass to each reranker.
        Documents beyond this position are appended unchanged.
    fusion_method:
        Method for fusing multi-query results.
        One of ``"interleaving"``, ``"rrf"``, ``"concatenation"``.
    fusion_window:
        Block size for interleaving fusion (ignored for other methods).
    rrf_k:
        K parameter for reciprocal rank fusion (ignored for other methods).
    retrieval_input:
        Controls what text is sent to the retriever for each sub-query.
        One of ``"subquery"`` (default — current behaviour),
        ``"original_query+subquery"``, ``"reasoning+subquery"``.
    post_fusion_reranker_input:
        Controls what text is sent to the post-fusion reranker.
        One of ``"original_query"`` (default — current behaviour),
        ``"original_query+subqueries"``, ``"original_query+reasoning"``,
        ``"reasoning+subqueries"``.
    """

    # Valid option sets (also used for CLI choices)
    RETRIEVAL_INPUT_CHOICES = ("subquery", "original_query+subquery", "reasoning+subquery")
    POST_FUSION_RERANKER_INPUT_CHOICES = ("original_query", "original_query+subqueries", "original_query+reasoning", "reasoning+subqueries")

    def __init__(
        self,
        retriever,
        *,
        reranker=None,
        post_retrieval_reranker=None,
        post_fusion_reranker=None,
        top_k: int = 100,
        rerank_top_k: int = 100,
        fusion_method: str = "interleaving",
        fusion_window: int = 1,
        rrf_k: int = 60,
        retrieval_input: str = "subquery",
        post_fusion_reranker_input: str = "original_query",
        ensure_novel_seen_docs: bool = False,
        seen_top_k: int = 5,
    ):
        self.retriever = retriever
        self.post_retrieval_reranker = post_retrieval_reranker
        # post_fusion_reranker takes precedence; fall back to legacy `reranker`
        self.post_fusion_reranker = post_fusion_reranker or reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.fusion_method = fusion_method
        self.fusion_window = fusion_window
        self.rrf_k = rrf_k
        self.retrieval_input = retrieval_input
        self.post_fusion_reranker_input = post_fusion_reranker_input
        self.ensure_novel_seen_docs = ensure_novel_seen_docs
        self.seen_top_k = seen_top_k
        self._global_seen_ids: set = set()

    def reset(self) -> None:
        """Clear the cross-iteration memory bank of seen doc_ids.

        Call this between queries so that the novelty filter is per-query.
        """
        self._global_seen_ids.clear()

    # ------------------------------------------------------------------
    # Query composition helpers
    # ------------------------------------------------------------------

    def _compose_retrieval_query(
        self,
        subquery: str,
        original_query: Optional[str],
        reasoning: Optional[str],
    ) -> str:
        """Build the retrieval query for a single sub-query based on ``self.retrieval_input``."""
        if self.retrieval_input == "original_query+subquery" and original_query:
            return f"{original_query} {subquery}"
        if self.retrieval_input == "reasoning+subquery" and reasoning:
            return f"{reasoning} {subquery}"
        # default / fallback: plain subquery
        return subquery

    def _compose_reranker_query(
        self,
        original_query: Optional[str],
        query_list: List[str],
        reasoning: Optional[str],
        raw_queries: Union[str, List[str]],
    ) -> str:
        """Build the reranking query for post-fusion reranker based on ``self.post_fusion_reranker_input``."""
        base = original_query or (raw_queries if isinstance(raw_queries, str) else query_list[0])
        if self.post_fusion_reranker_input == "original_query+subqueries":
            subqueries_text = " ".join(query_list)
            return f"{base} {subqueries_text}"
        if self.post_fusion_reranker_input == "original_query+reasoning" and reasoning:
            return f"{base} {reasoning}"
        if self.post_fusion_reranker_input == "reasoning+subqueries" and reasoning:
            subqueries_text = " ".join(query_list)
            return f"{reasoning} {subqueries_text}"
        # default / fallback: original_query only
        return base

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(
        self,
        queries: Union[str, List[str]],
        *,
        original_query: Optional[str] = None,
        reasoning: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full retrieval pipeline and return a single ranked list.

        Parameters
        ----------
        queries:
            A single query string or a list of queries.  When multiple
            queries are provided, each is retrieved independently and the
            results are fused into one list.
        original_query:
            The original user query, used as the reranking query when the
            search queries are sub-queries/keywords that differ from it.
            Falls back to ``queries`` (or ``queries[0]``) if not provided.
        reasoning:
            Optional trajectory context (e.g. current report, section plans,
            accumulated reasoning) that can be mixed into retrieval or
            reranking queries depending on ``retrieval_input`` /
            ``post_fusion_reranker_input`` configuration.
        top_k:
            Override the default ``self.top_k`` for this call.

        Returns
        -------
        List[Dict]
            Deduplicated ranked list of documents, each with at least
            ``doc_id`` and ``text``/``relevant_text`` keys.
        """
        # 1. Normalize input
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        if not query_list:
            return []

        # 2. Retrieve per query (compose retrieval query based on config)
        per_query_docs: List[List[Dict[str, Any]]] = []
        for q in query_list:
            retrieval_q = self._compose_retrieval_query(q, original_query, reasoning)
            try:
                raw = self.retriever.retrieve(retrieval_q)
                docs = normalize_retrieval_response(raw)
                per_query_docs.append(docs)
            except Exception as e:
                logger.warning(f"Retrieval failed for '{retrieval_q[:60]}': {e}")
                per_query_docs.append([])

        # 3. Optional post-retrieval reranking (per sub-query)
        if self.post_retrieval_reranker is not None:
            for i, (q, docs) in enumerate(zip(query_list, per_query_docs)):
                if not docs:
                    continue
                rerank_top_k = min(self.rerank_top_k, len(docs))
                rerank_input = docs[:rerank_top_k]
                rerank_tail = docs[rerank_top_k:]
                try:
                    reranked = self.post_retrieval_reranker.rerank(q, rerank_input)
                    per_query_docs[i] = reranked + rerank_tail
                except Exception as e:
                    logger.warning(f"Post-retrieval reranking failed for '{q[:60]}': {e}")

        # 4. Fuse (no-op for single query via fuse_results short-circuit)
        fused_docs = fuse_results(
            per_query_docs,
            fusion_method=self.fusion_method,
            rrf_k=self.rrf_k,
            interleaving_window=self.fusion_window,
        )

        # 5. Deduplicate (keep first occurrence)
        seen_ids = set()
        deduped: List[Dict[str, Any]] = []
        for doc in fused_docs:
            doc_id = doc.get("doc_id") or doc.get("id")
            if doc_id and doc_id in seen_ids:
                continue
            if doc_id:
                seen_ids.add(doc_id)
            deduped.append(doc)

        # 6. Optional post-fusion reranking (compose query based on config)
        if self.post_fusion_reranker is not None and deduped:
            rerank_query = self._compose_reranker_query(
                original_query, query_list, reasoning, queries,
            )
            rerank_top_k = min(self.rerank_top_k, len(deduped))
            rerank_input = deduped[:rerank_top_k]
            rerank_tail = deduped[rerank_top_k:]
            try:
                reranked = self.post_fusion_reranker.rerank(rerank_query, rerank_input)
                deduped = reranked + rerank_tail
            except Exception as e:
                logger.warning(f"Post-fusion reranking failed, returning fused order: {e}")

        # 7. Optional cross-iteration novelty filter
        #    When enabled, remove docs already seen in previous iterations
        #    and register the top seen_top_k novel docs in the memory bank
        #    (those are the ones the agent will consume via [:seen_top_k]).
        if self.ensure_novel_seen_docs:
            novel_docs: List[Dict[str, Any]] = []
            for doc in deduped:
                doc_id = doc.get("doc_id") or doc.get("id")
                if doc_id and doc_id in self._global_seen_ids:
                    continue
                novel_docs.append(doc)
            # Pre-register the first seen_top_k novel docs
            registered = 0
            for doc in novel_docs:
                if registered >= self.seen_top_k:
                    break
                doc_id = doc.get("doc_id") or doc.get("id")
                if doc_id:
                    self._global_seen_ids.add(doc_id)
                    registered += 1
            return novel_docs

        return deduped
