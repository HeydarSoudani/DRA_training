"""utils.eval_utils — pipeline orchestration helpers.

Contains the evaluation / fusion / reranker symbols used by the inference
pipeline (``experiments/run_dra_inference.py`` and ``utils/pipeline.py``):

    build_evaluators                Instantiate Retrieval/Generation/Trajectory/
                                    CitedDoc/SeenDoc/Accuracy evaluators.
    load_processed_results          Reload saved files for resumed queries.
    evaluate_and_save               Run all evaluations and write summary.json.
    ALL_AGGREGATION_FUSION_METHODS  All aggregation fusion methods.
    run_fusion_eval                 Evaluate all fusion methods, update summary.json.
    build_reranker_from_config      Instantiate a reranker from pipeline kwargs.

Modules internalised into this repo (corpus_dataset → indexing_corpus_dataset,
evaluation) are imported from their new homes.
"""

import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from utils.io_utils import (
    load_result_from_trec,
    load_result_from_saved_files,
)

from indexing_corpus_dataset.ranking_results import (
    RankingResults, RankingResult, save_ranking_results,
)
from evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    TrajectoryEvaluator,
    CitedDocRetrievalEvaluator,
    SeenDocRetrievalEvaluator,
    AccuracyEvaluator,
)
from evaluation.retrieval_evaluation import evaluate_results
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


# ===========================================================================
# Reranker construction
# ===========================================================================

LISTWISE_TEMPLATE_TO_MODEL = {
    "rankzephyr": "castorini/rank_zephyr_7b_v1_full",
    "rankgpt": "azure/gpt-4.1",
    "qwen3": "Qwen/Qwen3-8B",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
}


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
        from searcher_component.rerankers import setup_batched_reranker
        return setup_batched_reranker(
            reranker_model=cfg["reranker_model"],
            max_chars_per_document=cfg["max_chars_per_document"],
        )

    if post_retrieval_reranker == "listwise":
        cfg = kwargs.get("listwise_reranker_config")
        if cfg is None:
            return None
        from searcher_component.rerankers import setup_listwise_vllm_reranker
        return setup_listwise_vllm_reranker(**cfg)

    if post_retrieval_reranker == "rankllama":
        cfg = kwargs.get("rankllama_reranker_config")
        if cfg is None:
            return None
        if device:
            cfg = dict(cfg)
            cfg["device"] = device
        from searcher_component.rerankers import setup_rankllama_reranker
        return setup_rankllama_reranker(**cfg)

    if post_retrieval_reranker == "rank1":
        cfg = kwargs.get("rank1_reranker_config")
        if cfg is None:
            return None
        if device:
            cfg = dict(cfg)
            cfg["device"] = device
        from searcher_component.rerankers import setup_rank1_reranker
        return setup_rank1_reranker(**cfg)

    if post_retrieval_reranker == "rank_r1":
        cfg = kwargs.get("rank_r1_reranker_config")
        if cfg is None:
            return None
        from searcher_component.rerankers import setup_rank_r1_reranker
        return setup_rank_r1_reranker(**cfg)

    if post_retrieval_reranker == "qwen3_reranker":
        cfg = kwargs.get("qwen3_reranker_config")
        if cfg is None:
            return None
        from searcher_component.rerankers import setup_qwen3_reranker
        return setup_qwen3_reranker(**cfg)

    if post_retrieval_reranker in ("report_aware", "report_aware_iter"):
        cfg = kwargs.get("report_aware_reranker_config")
        if cfg is None:
            return None
        from searcher_component.rerankers import setup_report_aware_reranker
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
        from searcher_component.rerankers import setup_query_aware_doc_rewriting_reranker
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


# ===========================================================================
# Fusion utilities
# ===========================================================================

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


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _evaluate_one_fusion_worker(method: str, results: Dict[str, Any], qrels: Dict[str, Dict[str, int]], k_values: List[int], rrf_k: int, interleaving_window: int, fusion_k: Optional[int], run_dir_str: Optional[str], embeddings_dictionary: Optional[Dict[str, Any]] = None, worker_idx: int = 0) -> Dict[str, Any]:
    """Run one fusion method end-to-end: build ranking, evaluate, save TREC.

    Returns a dict with keys ``method``, ``metrics`` (or ``None`` on skip),
    ``trec_path`` (or ``None``), and ``error`` (or ``None``).
    """
    try:
        ranking_results = build_ranking_with_fusion(
            results,
            fusion_method=method,
            rrf_k=rrf_k,
            interleaving_window=interleaving_window,
            fusion_k=fusion_k,
            embeddings_dictionary=embeddings_dictionary,
            desc=f"[worker {worker_idx}] {method}",
            position=worker_idx,
            leave=True,
        )

        if len(ranking_results.results) == 0 or len(qrels) == 0:
            return {"method": method, "metrics": None, "trec_path": None, "error": None, "skipped": True}

        metrics = evaluate_results(
            results=ranking_results,
            qrels=qrels,
            k_values=k_values,
        )
        num_queries = len(ranking_results.get_unique_queries())
        metrics["num_queries"] = num_queries
        metrics["avg_docs_per_query"] = (
            len(ranking_results.results) / num_queries if num_queries > 0 else 0.0
        )

        trec_path = None
        if run_dir_str:
            trec_path = f"{run_dir_str.rstrip('/')}/ranking_results_{method}.trec"
            save_ranking_results(ranking_results, trec_path, format_type="trec")

        return {"method": method, "metrics": metrics, "trec_path": trec_path, "error": None, "skipped": False}

    except Exception as exc:
        return {"method": method, "metrics": None, "trec_path": None, "error": str(exc), "skipped": False}


def _evaluate_gpu_fusion_worker(method: str, results: Dict[str, Any], qrels: Dict[str, Dict[str, int]], k_values: List[int], rrf_k: int, interleaving_window: int, fusion_k: Optional[int], run_dir_str: Optional[str], encoder_kwargs: Dict[str, Any], device: str, worker_idx: int = 0) -> Dict[str, Any]:
    """Run one GPU-based (diversity) fusion method on a dedicated GPU device.

    Creates a fresh :class:`Encoder` on *device*, builds the embeddings
    dictionary there, then delegates to the standard fusion + evaluation path.
    This lets multiple diversity methods run in parallel, each owning one GPU.
    """
    try:
        # Re-import here because this runs in a separate process
        import sys
        from pathlib import Path as _Path
        _here = _Path(__file__).resolve()
        sys.path.append(str(_here.parents[1] / "src"))
        from searcher_component.retriever import Encoder

        encoder = Encoder(device=device, **encoder_kwargs)

        all_ranked_lists = [
            s["final_ranked_list"]
            for result in results.values()
            for s in result.get("sections", [])
            if s.get("final_ranked_list")
        ]
        embeddings_dictionary = build_embeddings_dictionary(all_ranked_lists, encoder, position=worker_idx)

        return _evaluate_one_fusion_worker(
            method, results, qrels, k_values, rrf_k, interleaving_window,
            fusion_k, run_dir_str, embeddings_dictionary=embeddings_dictionary,
            worker_idx=worker_idx,
        )
    except Exception as exc:
        return {"method": method, "metrics": None, "trec_path": None, "error": str(exc), "skipped": False}


def evaluate_all_fusions_and_save(results: Dict[str, Any], qrels: Dict[str, Dict[str, int]], k_values: List[int], run_dir: Optional[Path], fusion_methods: Optional[List[str]] = None, rrf_k: int = 60, interleaving_window: int = 3, fusion_k: Optional[int] = None, max_workers: Optional[int] = None, encoder=None, num_gpus: int = 0, reranker=None, rerank_top_k: int = 100) -> Dict[str, Any]:
    """Evaluate retrieval with every aggregation fusion method and save TREC + summary.

    For each method in *fusion_methods* this function:
    1. Builds a ranking by fusing per-section ``final_ranked_list`` arrays.
    2. Saves ``ranking_results_{method}.trec`` under *run_dir*.
    3. Computes recall/NDCG/… metrics at *k_values*.

    CPU methods run sequentially, each showing a per-query ``tqdm`` bar.
    GPU diversity methods run in parallel (one process per GPU), showing a
    per-method completion bar — subprocess tqdm cannot propagate to the main
    terminal, so per-query progress is not available for these.

    Args:
        results:             Unified results dict keyed by query_id.
        qrels:               Ground-truth relevance judgements.
        k_values:            Evaluation cut-offs.
        run_dir:             Output directory; if None, saving is skipped.
        fusion_methods:      Methods to evaluate; defaults to
                             :data:`ALL_AGGREGATION_FUSION_METHODS`.
        rrf_k:               K parameter for RRF.
        interleaving_window: Window size for interleaving fusion.
        fusion_k:            If set, only the first *fusion_k* documents from
                             each section list are passed to the fusion method;
                             the remainder are appended via cheap interleaving.
                             ``None`` fuses all documents.
        max_workers:         Unused; kept for API compatibility.
        encoder:             Optional ``Encoder`` instance (the retriever backbone).
                             Required for diversity fusion methods.  The embeddings
                             dictionary is built once and reused across all
                             diversity methods.

    Returns:
        Dict ``{method_name: metrics_dict}`` for every evaluated method.
    """
    if fusion_methods is None:
        fusion_methods = ALL_AGGREGATION_FUSION_METHODS

    run_dir_str = str(run_dir) if run_dir else None

    # Split methods into CPU-only and GPU-based (diversity) groups.
    cpu_methods = [m for m in fusion_methods if m not in DIVERSITY_FUSION_METHODS]
    diversity_methods = [m for m in fusion_methods if m in DIVERSITY_FUSION_METHODS]

    # Determine whether we have the encoder info needed to spawn per-GPU processes.
    can_use_gpu_workers = (
        bool(diversity_methods)
        and encoder is not None
        and num_gpus > 0
        and hasattr(encoder, "model_name")
        and hasattr(encoder, "model_path")
    )

    # Build embeddings once in the main process when GPU workers are not available.
    embeddings_dictionary: Optional[Dict[str, Any]] = None
    encoder_kwargs: Optional[Dict[str, Any]] = None
    if diversity_methods:
        if can_use_gpu_workers:
            encoder_kwargs = {
                "model_name": encoder.model_name,
                "model_path": encoder.model_path,
                "pooling_method": encoder.pooling_method,
                "max_length": encoder.max_length,
                "use_fp16": encoder.use_fp16,
            }
        elif encoder is not None:
            all_ranked_lists = [
                s["final_ranked_list"]
                for result in results.values()
                for s in result.get("sections", [])
                if s.get("final_ranked_list")
            ]
            embeddings_dictionary = build_embeddings_dictionary(all_ranked_lists, encoder)
        else:
            print(f"  ⚠  No encoder available — skipping {len(diversity_methods)} diversity fusion method(s): {diversity_methods}")
            diversity_methods = []

    all_metrics: Dict[str, Any] = {}

    # ── CPU methods: sequential, each with a per-query tqdm bar ──────────────
    for method in cpu_methods:
        try:
            ranking_results = build_ranking_with_fusion(
                results,
                fusion_method=method,
                rrf_k=rrf_k,
                interleaving_window=interleaving_window,
                fusion_k=fusion_k,
                desc=method,
                reranker=reranker,
                rerank_top_k=rerank_top_k,
            )

            if len(ranking_results.results) == 0 or len(qrels) == 0:
                print(f"  ⚠  [{method}] No results or qrels — skipping evaluation")
                continue

            metrics = evaluate_results(results=ranking_results, qrels=qrels, k_values=k_values)
            num_queries = len(ranking_results.get_unique_queries())
            metrics["num_queries"] = num_queries
            metrics["avg_docs_per_query"] = (
                len(ranking_results.results) / num_queries if num_queries > 0 else 0.0
            )

            trec_path = None
            if run_dir_str:
                trec_path = f"{run_dir_str.rstrip('/')}/ranking_results_{method}.trec"
                save_ranking_results(ranking_results, trec_path, format_type="trec")

            recall_at_100 = metrics.get("Recall", {}).get("Recall@100", None)
            ndcg_at_10 = metrics.get("NDCG", {}).get("NDCG@10", None)
            recall_str = f"  Recall@100={recall_at_100:.4f}" if recall_at_100 is not None else ""
            ndcg_str = f"  NDCG@10={ndcg_at_10:.4f}" if ndcg_at_10 is not None else ""
            saved_str = "  ✓" if trec_path else ""
            print(f"  [{method}]{recall_str}{ndcg_str}  ({num_queries} queries){saved_str}")

            all_metrics[method] = metrics

        except Exception as exc:
            print(f"  ✗ [{method}] error: {exc}")

    # ── Diversity methods: parallel processes, one tqdm bar per worker ───────
    # Each worker uses tqdm(position=worker_idx) so all bars display simultaneously:
    #   [worker 0] method_a  9%|███ | 7/74 [01:50<17:20, 15.53s/query]
    #   [worker 1] method_b 45%|███████████| 33/74 [00:42<00:51, 1.02s/query]
    #   ...
    if diversity_methods:
        futures: Dict = {}
        _spawn_ctx = multiprocessing.get_context("spawn")
        n_workers = min(len(diversity_methods), max(1, num_gpus) if num_gpus > 0 else len(diversity_methods))
        gpu_tag = f" on {n_workers} GPU(s)" if can_use_gpu_workers else f" ({n_workers} parallel workers)"
        print(f"  [diversity] Launching {len(diversity_methods)} methods in parallel{gpu_tag}: {diversity_methods}")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_spawn_ctx) as pool:
            for i, method in enumerate(diversity_methods):
                worker_idx = i % n_workers
                if can_use_gpu_workers:
                    device = f"cuda:{i % num_gpus}"
                    f = pool.submit(
                        _evaluate_gpu_fusion_worker,
                        method, results, qrels, k_values,
                        rrf_k, interleaving_window, fusion_k, run_dir_str,
                        encoder_kwargs, device, worker_idx,
                    )
                else:
                    emb_dict = embeddings_dictionary
                    f = pool.submit(
                        _evaluate_one_fusion_worker,
                        method, results, qrels, k_values,
                        rrf_k, interleaving_window, fusion_k, run_dir_str,
                        emb_dict, worker_idx,
                    )
                futures[f] = method

            for future in as_completed(futures):
                method = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    print(f"  ✗ [{method}] unexpected error: {exc}")
                    continue

                if out.get("error"):
                    print(f"  ✗ [{method}] error: {out['error']}")
                    continue
                if out.get("skipped"):
                    print(f"  ⚠  [{method}] No results or qrels — skipping evaluation")
                    continue

                metrics = out["metrics"]
                trec_path = out["trec_path"]
                num_queries = metrics.get("num_queries", 0)
                recall_at_100 = metrics.get("Recall", {}).get("Recall@100", None)
                ndcg_at_10 = metrics.get("NDCG", {}).get("NDCG@10", None)
                recall_str = f"  Recall@100={recall_at_100:.4f}" if recall_at_100 is not None else ""
                ndcg_str = f"  NDCG@10={ndcg_at_10:.4f}" if ndcg_at_10 is not None else ""
                gpu_tag = " [GPU]" if can_use_gpu_workers else ""
                saved_str = "  ✓" if trec_path else ""
                print(f"  [{method}]{gpu_tag}{recall_str}{ndcg_str}  ({num_queries} queries){saved_str}")

                all_metrics[method] = metrics

    return all_metrics


def run_fusion_eval(
    results: Dict[str, Any],
    qrels_dict: Dict[str, Dict[str, int]],
    kwargs: Dict[str, Any],
    run_dir: Optional[Path],
    num_gpus: int,
) -> Dict[str, Any]:
    """Evaluate all aggregation fusion methods and update summary.json.

    Thin wrapper around :func:`evaluate_all_fusions_and_save` that extracts
    the relevant parameters from *kwargs*, prints a summary header, and
    persists the resulting metrics into ``{run_dir}/summary.json``.

    Args:
        results:    Unified results dict keyed by query_id.
        qrels_dict: Ground-truth relevance judgements.
        kwargs:     Pipeline kwargs dict (read: k_values, aggregation_rrf_k,
                    aggregation_interleaving_window, fusion_k, fusion_methods,
                    encoder).
        run_dir:    Output directory for TREC files and summary.json.
        num_gpus:   Number of GPU workers for diversity fusion methods.

    Returns:
        ``{method_name: metrics_dict}`` for every evaluated method.
    """
    k_values = kwargs.get("k_values", [1, 3, 5, 10, 25, 100, 500, 1000])
    rrf_k = kwargs.get("aggregation_rrf_k", 60)
    interleaving_window = kwargs.get("aggregation_interleaving_window", 3)
    fusion_k = kwargs.get("fusion_k", None)
    fusion_methods = kwargs.get("fusion_methods", None) or ALL_AGGREGATION_FUSION_METHODS
    print(f"\n{'=' * 80}")
    print("MULTI-FUSION AGGREGATION EVALUATION")
    print(f"  run_dir : {run_dir}")
    print(f"  queries : {len(results)}")
    print(f"  methods : {fusion_methods}")
    if fusion_k is not None:
        print(f"  fusion_k: {fusion_k}")
    print(f"{'=' * 80}")
    fusion_metrics = evaluate_all_fusions_and_save(
        results=results,
        qrels=qrels_dict,
        k_values=k_values,
        run_dir=run_dir,
        fusion_methods=fusion_methods,
        rrf_k=rrf_k,
        interleaving_window=interleaving_window,
        fusion_k=fusion_k,
        encoder=kwargs.get("encoder", None),
        num_gpus=num_gpus,
        reranker=kwargs.get("reranker", None),
        rerank_top_k=kwargs.get("rerank_top_k", 100),
    )
    if fusion_metrics and run_dir:
        summary_path = Path(str(run_dir)) / "summary.json"
        summary = json.load(open(summary_path)) if summary_path.exists() else {}
        summary["retrieval"] = fusion_metrics
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Updated summary with fusion metrics: {summary_path}")
    return fusion_metrics or {}


# ===========================================================================
# Evaluator construction + results loading / evaluation
# ===========================================================================

def build_evaluators(qrels: Dict, kwargs: Dict[str, Any], answers: Optional[Dict[str, str]] = None, questions: Optional[Dict[str, str]] = None) -> Tuple[RetrievalEvaluator, GenerationEvaluator, TrajectoryEvaluator, CitedDocRetrievalEvaluator, SeenDocRetrievalEvaluator, Optional[AccuracyEvaluator]]:
    """Instantiate Retrieval, Generation, Trajectory, CitedDoc, SeenDoc, and Accuracy evaluators.

    Args:
        qrels:     Qrels dict loaded from the dataset.
        kwargs:    Pipeline configuration dict.
        answers:   Optional mapping of query_id -> ground-truth answer.
                   When provided (e.g. for BrowseComp-Plus), an AccuracyEvaluator
                   is created.
        questions: Optional mapping of query_id -> question text.

    Returns:
        ``(retrieval_evaluator, generation_evaluator, trajectory_evaluator,
          cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator)``
        where ``accuracy_evaluator`` is None when no answers are available.
    """
    k_values = kwargs.get("k_values", [1, 3, 5, 10, 25, 100])
    retrieval_evaluator = RetrievalEvaluator(
        qrels=qrels,
        k_values=k_values,
        fusion_method=kwargs.get("consolidation_fusion_method", "interleaving"),
        interleaving_window=kwargs.get("interleaving_window", 3),
        rrf_k=kwargs.get("rrf_k", 60),
        eval_top_k=kwargs.get("eval_top_k"),
    )
    cited_doc_evaluator = CitedDocRetrievalEvaluator(qrels=qrels, k_values=k_values)
    seen_doc_evaluator = SeenDocRetrievalEvaluator(
        qrels=qrels,
        k_values=k_values,
        fusion_method=kwargs.get("consolidation_fusion_method", "interleaving"),
        interleaving_window=kwargs.get("interleaving_window", 3),
        rrf_k=kwargs.get("rrf_k", 60),
    )
    accuracy_evaluator = None
    if answers:
        judge_kwargs: Dict[str, Any] = {}
        judge_api_url = kwargs.get("judge_api_url")
        if judge_api_url:
            judge_kwargs["judge_api_base"] = judge_api_url
        accuracy_evaluator = AccuracyEvaluator(
            answers=answers,
            questions=questions,
            **judge_kwargs,
        )
    return retrieval_evaluator, GenerationEvaluator(), TrajectoryEvaluator(), cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator


def _load_single_query(run_dir, query_id, retrieval_dir_str, lightweight=False):
    """Load a single query's result files (used by ThreadPoolExecutor)."""
    result = load_result_from_saved_files(run_dir, query_id, lightweight=lightweight)
    if not result:
        trec_path = Path(retrieval_dir_str) / f"{query_id}.trec"
        result = load_result_from_trec(trec_path, query_id)
    return result or None


def load_processed_results(processed: set, retrieval_dir, results: Dict[str, Any], *, lightweight: bool = False) -> None:
    """Reload saved files for queries that were skipped due to resume logic.

    Tries to load the full result from trajectory JSON + generation MD + cited
    docs TREC first (via ``load_result_from_saved_files``).  Falls back to
    TREC-only reconstruction for backwards compatibility with older runs that
    only saved retrieval TREC files.

    Mutates ``results`` in-place by adding entries for any ``query_id`` that
    is in ``processed`` but not yet in ``results``.
    """
    if not processed or retrieval_dir is None:
        return

    retrieval_dir_str = str(retrieval_dir)

    if not Path(retrieval_dir_str).exists():
        return

    # run_dir is the parent of retrieval_dir (e.g. outputs/run_name/)
    run_dir = Path(retrieval_dir_str).parent

    to_load = [qid for qid in processed if qid not in results]
    if not to_load:
        return

    source = "local"

    # ── Try loading from pickle cache ────────────────────────────────────────
    cache_filename = "_eval_cache_lightweight.pkl" if lightweight else "_eval_cache.pkl"
    cache_path = Path(run_dir) / cache_filename

    to_load_set = set(to_load)
    try:
        if Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if to_load_set <= set(cached.keys()):
                for qid in to_load:
                    results[qid] = cached[qid]
                print(f"Loaded {len(to_load)} queries from eval cache ({source})")
                return
    except Exception as e:
        print(f"Warning: eval cache invalid or corrupted ({e}), rebuilding...")

    # ── Parallel load ────────────────────────────────────────────────────────
    max_workers = min(8, len(to_load))
    loaded_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_query, run_dir, qid, retrieval_dir_str, lightweight): qid
            for qid in to_load
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Loading results from {source}", unit="query"):
            qid = futures[future]
            try:
                result = future.result()
                if result:
                    results[qid] = result
                    loaded_count += 1
            except Exception as e:
                print(f"Warning: failed to load {qid}: {e}")

    if loaded_count:
        print(f"Loaded {loaded_count} previously processed queries from {source} for evaluation")

    # ── Save cache for next run ──────────────────────────────────────────────
    if loaded_count > 0:
        try:
            loaded_results = {qid: results[qid] for qid in to_load if qid in results}
            with open(cache_path, "wb") as f:
                pickle.dump(loaded_results, f)
            print(f"Saved eval cache ({loaded_count} queries)")
        except Exception as e:
            print(f"Warning: could not save eval cache: {e}")


def evaluate_and_save(results: Dict[str, Any], generation_evaluator: GenerationEvaluator, trajectory_evaluator: TrajectoryEvaluator, run_dir: Optional[Path], cited_doc_evaluator: Optional[CitedDocRetrievalEvaluator] = None, seen_doc_evaluator: Optional[SeenDocRetrievalEvaluator] = None, accuracy_evaluator: Optional[AccuracyEvaluator] = None, tracker_metrics: Optional[Dict[str, Any]] = None, tracker_evaluator=None) -> None:
    """Run all evaluations, print results, and save summary.json.

    Retrieval metrics are produced per fusion method by
    :func:`run_fusion_eval` and stored under the ``"retrieval"`` key in
    summary.json.  This function handles generation, trajectory,
    cited-doc, seen-doc, accuracy, and tracker evaluation.

    Terminal output order matches the summary.json structure.

    Args:
        results:               Unified results dict keyed by query_id.
        generation_evaluator:  Configured generation evaluator.
        trajectory_evaluator:  Configured trajectory evaluator.
        run_dir:               Output directory; if None, saving is skipped.
        cited_doc_evaluator:   Optional evaluator for cited docs (memory bank).
        seen_doc_evaluator:    Optional evaluator for seen docs (docs passed to LLM).
        accuracy_evaluator:    Optional LLM-as-judge accuracy evaluator
                               (used when ground-truth answers are available, e.g. BrowseComp-Plus).
        tracker_metrics:       Pre-computed tracker metrics dict (legacy).
        tracker_evaluator:     Optional TrackerEvaluator instance.  When
                               provided, ``tracker_metrics`` is ignored and
                               the evaluator is run here so printing follows
                               the canonical summary order.
    """
    # ------------------------------------------------------------------
    # 1. Evaluate everything (no printing yet)
    # ------------------------------------------------------------------
    generation_metrics = generation_evaluator.evaluate(results)
    trajectory_metrics = trajectory_evaluator.evaluate(results)

    cited_doc_metrics = {}
    if cited_doc_evaluator is not None:
        cited_doc_metrics = cited_doc_evaluator.evaluate(results)

    seen_doc_metrics = {}
    if seen_doc_evaluator is not None:
        seen_doc_metrics = seen_doc_evaluator.evaluate(results)

    accuracy_metrics = {}
    if accuracy_evaluator is not None:
        accuracy_metrics = accuracy_evaluator.evaluate(results)

    if tracker_evaluator is not None:
        tracker_metrics = tracker_evaluator.evaluate(results)

    # ------------------------------------------------------------------
    # 2. Guard: all evaluators must process the same number of queries
    # ------------------------------------------------------------------
    num_queries = generation_metrics.get("num_queries", 0)
    _evaluator_counts = {"generation": num_queries}
    if trajectory_metrics:
        _evaluator_counts["trajectory"] = trajectory_metrics.get("num_queries", 0)
    if cited_doc_metrics:
        _evaluator_counts["cited_doc_retrieval"] = cited_doc_metrics.get("num_queries", 0)
    if seen_doc_metrics:
        _evaluator_counts["seen_doc_retrieval"] = seen_doc_metrics.get("num_queries", 0)
    if accuracy_metrics:
        _evaluator_counts["accuracy"] = accuracy_metrics.get("num_evaluated", 0)
    if tracker_metrics:
        _evaluator_counts["tracker"] = tracker_metrics.get("num_queries_with_tracker", 0)

    mismatches = {k: v for k, v in _evaluator_counts.items() if v != num_queries}
    if mismatches:
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(
            "Evaluator query-count mismatch! Expected %d (from generation). "
            "Mismatches: %s", num_queries, mismatches,
        )

    # ------------------------------------------------------------------
    # 3. Print in canonical summary order
    # ------------------------------------------------------------------
    # Extract Metrics@N from cited_doc_metrics for top-level display
    metrics_at_n = {}
    if cited_doc_metrics:
        metrics_at_n = cited_doc_metrics.pop("Metrics@N", {})
        metrics_at_n.pop("num_queries", None)

    # -- num_queries + Metrics@N
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  num_queries: {num_queries}")
    if metrics_at_n:
        print(f"  Metrics@N:")
        print(f"    Recall@N:    {metrics_at_n.get('Recall@N', 0):.4f}")
        print(f"    Precision@N: {metrics_at_n.get('Precision@N', 0):.4f}")
        print(f"    F1@N:        {metrics_at_n.get('F1@N', 0):.4f}")
        print(f"    avg_N:       {metrics_at_n.get('avg_N', 0):.1f}")
    print("=" * 80)

    # -- accuracy
    if accuracy_evaluator is not None:
        accuracy_evaluator.print_results(accuracy_metrics)

    # -- trajectory
    trajectory_evaluator.print_results(trajectory_metrics)

    # -- tracker
    if tracker_metrics:
        if tracker_evaluator is not None:
            tracker_evaluator.print_results(tracker_metrics)
        else:
            print(f"\n{'=' * 80}")
            print("TRACKER STATISTICS")
            print("=" * 80)
            for k, v in tracker_metrics.items():
                print(f"  {k}: {v}")
            print("=" * 80)

    # -- generation
    generation_evaluator.print_results(generation_metrics)

    # -- cited_doc_retrieval
    if cited_doc_evaluator is not None:
        cited_doc_evaluator.print_results(cited_doc_metrics, header="RETRIEVAL EVALUATION RESULTS (CITED DOCS)")

    # -- seen_doc_retrieval
    if seen_doc_evaluator is not None:
        seen_doc_evaluator.print_results(seen_doc_metrics, header="RETRIEVAL EVALUATION RESULTS (SEEN DOCS)")

    # -- Save accuracy detail file
    if accuracy_metrics and run_dir and accuracy_evaluator is not None:
        acc_path = str(Path(str(run_dir)) / "accuracy.json")
        accuracy_evaluator.save_results(accuracy_metrics, acc_path)

    # ------------------------------------------------------------------
    # 4. Build and save summary.json in canonical order
    # ------------------------------------------------------------------
    if run_dir:
        summary = {"num_queries": num_queries}

        if metrics_at_n:
            summary["Metrics@N"] = metrics_at_n

        if accuracy_metrics:
            summary["accuracy"] = {
                "accuracy": accuracy_metrics["accuracy"],
                "num_correct": accuracy_metrics["num_correct"],
                "num_evaluated": accuracy_metrics["num_evaluated"],
            }

        if trajectory_metrics:
            summary["trajectory"] = trajectory_metrics

        if tracker_metrics:
            summary["tracker"] = tracker_metrics

        summary["generation"] = generation_metrics

        if cited_doc_metrics:
            summary["cited_doc_retrieval"] = cited_doc_metrics
        if seen_doc_metrics:
            summary["seen_doc_retrieval"] = seen_doc_metrics

        run_dir_str = str(run_dir)
        summary_content = json.dumps(summary, indent=2)
        with open(Path(run_dir_str) / "summary.json", "w") as f:
            f.write(summary_content)
        print(f"  ✓ Saved summary: {run_dir_str}/summary.json")
