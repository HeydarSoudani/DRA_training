"""Fusion construction: turn per-section ranked lists into a fused ranking.

Defines the catalogue of aggregation fusion methods and the helpers that build
a :class:`RankingResults` by fusing the per-section ``final_ranked_list`` arrays
produced by the parallel pipeline.  Evaluation/scoring of the resulting rankings
lives in :mod:`evaluation.retrieval.fusion.evaluate`.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from utils.ranking_results import RankingResults, RankingResult
from searcher_component.fusion_methods.interleaving import InterleavingFusion
from searcher_component.fusion_methods.rrf import ReciprocalRankFusion
from searcher_component.fusion_methods.concatenation import SimpleConcatenation
from searcher_component.fusion_methods.random_fusion import RandomFusion
from searcher_component.fusion_methods.epsilon_greedy import EpsilonGreedyFusion
from searcher_component.fusion_methods.thompson_bernoulli import ThompsonBernoulliFusion
from searcher_component.fusion_methods.thompson_gaussian import ThompsonGaussianFusion

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


DIVERSITY_FUSION_METHODS = {
    "diversity_linear",
    "diversity_concave",
    "bernoulli_topk_ucb_diversity_linear",
    "bernoulli_topk_ucb_diversity_concave",
}

#: All aggregation fusion methods tried by the parallel pipeline.
ALL_AGGREGATION_FUSION_METHODS: List[str] = [
    # --- Deterministic baselines ---
    "interleaving",
    "rrf",
    "concatenation",
    # --- Simple stochastic ---
    "random",
    "epsilon_greedy",
    # --- Bernoulli Thompson sampling variants ---
    "thompson_bernoulli",
    "bernoulli_rank_aware",
    "bernoulli_ucb",
    "bernoulli_topk",
    # --- Diversity-aware ---
    "diversity_linear",
    "diversity_concave",
    "bernoulli_topk_ucb_diversity_linear",
    "bernoulli_topk_ucb_diversity_concave",
    # --- Gaussian Thompson sampling ---
    "thompson_gaussian",
    "thompson_gaussian_nig",
]


def build_embeddings_dictionary(ranked_lists: List[List[Dict[str, Any]]], encoder, batch_size: int = 32, position: int = 0) -> Dict[str, Any]:
    """Build a ``{doc_id: embedding}`` mapping for all unique documents in *ranked_lists*.

    Uses the already-loaded *encoder* (an ``Encoder`` instance from the dense
    retriever) to embed document text in batches.  Each embedding is shaped
    ``(1, d)`` so it is directly compatible with
    ``sklearn.metrics.pairwise.cosine_similarity``.

    Documents are encoded in batches of *batch_size* to avoid GPU OOM.
    The encoder is reused as-is on its existing device — no extra model
    copies are loaded.

    Args:
        ranked_lists: Flat list of ranked result lists (one per subquery / section).
        encoder:      An ``Encoder`` instance from the dense retriever.
        batch_size:   Number of documents per encoding batch.

    Returns:
        Dict mapping doc_id → numpy array of shape ``(1, d)``.
    """
    # Collect unique docs preserving first-occurrence text
    doc_texts: Dict[str, str] = {}
    for rl in ranked_lists:
        for doc in rl:
            doc_id = doc.get("doc_id") or doc.get("id")
            if doc_id and doc_id not in doc_texts:
                text = doc.get("contents") or doc.get("text") or ""
                doc_texts[doc_id] = text

    if not doc_texts:
        return {}

    doc_ids = list(doc_texts.keys())
    texts = [doc_texts[d] for d in doc_ids]

    print(f"Encoding {len(doc_ids)} documents on {encoder.device} (batch_size={batch_size}) …")

    all_embeddings = []
    iter_range = range(0, len(texts), batch_size)
    if _HAS_TQDM:
        iter_range = _tqdm(iter_range, desc=f"[worker {position}] Encoding docs", unit="batch", position=position, leave=True)
    for i in iter_range:
        batch = texts[i : i + batch_size]
        embs = encoder.encode(batch, is_query=False)  # shape (b, d)
        all_embeddings.append(embs)

    embeddings = np.concatenate(all_embeddings, axis=0)  # shape (n, d)
    return {doc_id: emb.reshape(1, -1) for doc_id, emb in zip(doc_ids, embeddings)}


def build_fusion_obj(method: str, rrf_k: int = 60, interleaving_window: int = 3, embeddings_dictionary: Optional[Dict[str, Any]] = None):
    """Instantiate the fusion object for *method*.

    Args:
        method:               Fusion method name.
        rrf_k:                Smoothing constant for RRF.
        interleaving_window:  Window size for interleaving fusion.
        embeddings_dictionary: Pre-built ``{doc_id: embedding}`` mapping required
                              by diversity fusion methods.  If ``None`` or empty
                              when a diversity method is requested, a warning is
                              emitted and the method falls back to plain Bernoulli TS.
    """
    if method == "interleaving":
        return InterleavingFusion(window=interleaving_window)
    if method == "rrf":
        return ReciprocalRankFusion(k=rrf_k)
    if method == "concatenation":
        return SimpleConcatenation()
    if method == "random":
        return RandomFusion()
    if method == "epsilon_greedy":
        return EpsilonGreedyFusion()
    if method == "thompson_bernoulli":
        return ThompsonBernoulliFusion()
    if method == "thompson_gaussian":
        return ThompsonGaussianFusion()
    if method == "thompson_gaussian_nig":
        return ThompsonGaussianFusion(nig=True)
    if method == "bernoulli_rank_aware":
        return ThompsonBernoulliFusion(rank_aware=True)
    if method == "bernoulli_ucb":
        return ThompsonBernoulliFusion(ucb=True)
    if method == "bernoulli_topk":
        return ThompsonBernoulliFusion(k=5)
    if method in DIVERSITY_FUSION_METHODS:
        emb_dict = embeddings_dictionary or {}
        if not emb_dict:
            raise ValueError(
                f"[{method}] embeddings_dictionary is required for diversity fusion but was not provided. "
                "Pass an encoder via build_ranking_with_fusion(..., encoder=...) to build it automatically."
            )
        if method == "diversity_linear":
            return ThompsonBernoulliFusion(diversity="linear", embeddings_dictionary=emb_dict)
        if method == "diversity_concave":
            return ThompsonBernoulliFusion(diversity="concave", embeddings_dictionary=emb_dict)
        if method == "bernoulli_topk_ucb_diversity_linear":
            return ThompsonBernoulliFusion(k=5, ucb=True, diversity="linear",
                                           embeddings_dictionary=emb_dict)
        if method == "bernoulli_topk_ucb_diversity_concave":
            return ThompsonBernoulliFusion(k=5, ucb=True, diversity="concave",
                                           embeddings_dictionary=emb_dict)
    raise ValueError(f"Unknown fusion method: {method!r}")


def build_ranking_with_fusion(results: Dict[str, Any], fusion_method: str, rrf_k: int = 60, interleaving_window: int = 3, fusion_k: Optional[int] = None, encoder=None, embeddings_dictionary: Optional[Dict[str, Any]] = None, desc: str = "", position: int = 0, leave: bool = False, reranker=None, rerank_top_k: int = 100) -> RankingResults:
    """Build a :class:`RankingResults` by fusing per-section lists with *fusion_method*.

    Each entry in ``results`` must have a ``sections`` key where every section
    carries a ``final_ranked_list`` (populated by :class:`ParallelDMAgent`).
    Queries without any section results are silently skipped.

    Args:
        results:             Unified results dict keyed by query_id.
        fusion_method:       Aggregation fusion method (``"interleaving"``, ``"rrf"``,
                             ``"concatenation"``, etc.).
        rrf_k:               K constant for RRF (ignored for other methods).
        interleaving_window: Window size for interleaving (ignored for other methods).
        fusion_k:            If set, only the first *fusion_k* documents from each
                             section list are passed to the fusion method. Documents
                             beyond position *fusion_k* are appended via cheap
                             interleaving. ``None`` (default) fuses all documents.
        encoder:             Optional ``Encoder`` instance (the retriever backbone).
                             Required for diversity fusion methods; ignored otherwise.
                             If ``None`` and a diversity method is requested, an
                             ``embeddings_dictionary`` warning is emitted.

    Returns:
        :class:`RankingResults` ready for TREC saving and evaluation.
    """
    # Build embeddings dictionary once across all queries for diversity fusion methods.
    # A pre-built dict takes priority (e.g. when called from a worker process).
    if fusion_method in DIVERSITY_FUSION_METHODS and not embeddings_dictionary and encoder is not None:
        all_ranked_lists = [
            s["final_ranked_list"]
            for result in results.values()
            for s in result.get("sections", [])
            if s.get("final_ranked_list")
        ]
        embeddings_dictionary = build_embeddings_dictionary(all_ranked_lists, encoder, position=position)

    fuser = build_fusion_obj(
        fusion_method,
        rrf_k=rrf_k,
        interleaving_window=interleaving_window,
        embeddings_dictionary=embeddings_dictionary,
    )
    tail_fuser = InterleavingFusion(window=interleaving_window)
    ranking_results = RankingResults(results=[])

    query_items = results.items()
    if _HAS_TQDM:
        query_items = _tqdm(query_items, total=len(results), desc=desc or fusion_method, unit="query", leave=leave, position=position)
    for query_id, result in query_items:
        sections = result.get("sections", [])
        per_section_lists = [
            s["final_ranked_list"]
            for s in sections
            if s.get("final_ranked_list")
        ]

        if not per_section_lists:
            # Fallback: use final_ranked_list directly (e.g. single-section)
            final_rl = result.get("final_ranked_list", [])
            if final_rl:
                per_section_lists = [final_rl]

        if not per_section_lists:
            # Fallback: extract per-iteration lists from trajectory
            # (deep_research_agents format: trajectory steps with "docs" or "all_docs")
            for step in result.get("trajectory", []):
                if "all_docs" in step and step["all_docs"]:
                    all_docs = step["all_docs"]
                    if all_docs and isinstance(all_docs[0], dict):
                        turn_docs = all_docs
                    else:
                        turn_docs = [doc for retrieve_docs in all_docs for doc in retrieve_docs]
                    if turn_docs:
                        per_section_lists.append(turn_docs)
                elif "docs" in step and step["docs"]:
                    per_section_lists.append(step["docs"])

        if not per_section_lists:
            continue

        # Track which iteration/section each doc_id first appeared in (1-based)
        doc_first_iter: Dict[str, int] = {}
        for iter_idx, section_docs in enumerate(per_section_lists, 1):
            for doc in section_docs:
                did = doc.get("doc_id") or doc.get("id") or ""
                if did and did not in doc_first_iter:
                    doc_first_iter[did] = iter_idx

        if len(per_section_lists) == 1:
            fused = per_section_lists[0]
        elif fusion_k is not None:
            # Split each section list into top-k (careful fusion) and tail (cheap interleave)
            top_lists = [lst[:fusion_k] for lst in per_section_lists]
            tail_lists = [lst[fusion_k:] for lst in per_section_lists]

            fused_top = fuser.fuse(top_lists)

            non_empty_tails = [t for t in tail_lists if t]
            if non_empty_tails:
                fused_tail = (
                    tail_fuser.fuse(non_empty_tails)
                    if len(non_empty_tails) > 1
                    else non_empty_tails[0]
                )
            else:
                fused_tail = []

            fused = fused_top + fused_tail
        else:
            fused = fuser.fuse(per_section_lists)

        # Deduplicate: keep first occurrence per doc_id
        seen: set = set()
        deduped = []
        for doc in fused:
            did = doc.get("doc_id") or doc.get("id") or ""
            if did and did not in seen:
                seen.add(did)
                deduped.append(doc)
            elif not did:
                deduped.append(doc)

        # Optional post-fusion reranking
        if reranker is not None and deduped:
            query_text = result.get("query", "")
            if query_text:
                deduped = reranker.rerank(query_text, deduped[:rerank_top_k]) + deduped[rerank_top_k:]

        for rank, doc in enumerate(deduped, 1):
            did = doc.get("doc_id") or doc.get("id") or ""
            if did:
                score = doc.get("score", doc.get("rank_score", 0.0))
                iter_idx = doc_first_iter.get(did, 1)
                meta = {"run_tag": f"iter_{iter_idx}"}
                reranker_score = doc.get("reranker_score")
                if reranker_score is not None:
                    meta["reranker_score"] = reranker_score
                ranking_results.add_result(RankingResult(
                    query_id=query_id,
                    doc_id=did,
                    rank=rank,
                    rank_score=score,
                    metadata=meta,
                ))

    # Scores are normalized centrally by normalize_scores_by_rank() at
    # evaluation / TREC-write time, so we keep original scores here.
    return ranking_results
