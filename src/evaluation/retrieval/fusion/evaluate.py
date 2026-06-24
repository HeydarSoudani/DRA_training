"""Fusion evaluation: run every fusion method, score it, and persist results.

Builds a fused ranking per method (via :mod:`.builders`), scores it with the
TREC metric functions, writes ``ranking_results_{method}.trec`` files, and
updates ``summary.json``.  CPU methods run sequentially; GPU diversity methods
run one-process-per-GPU.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.ranking_results import save_ranking_results

from ..metrics import evaluate_results
from .builders import (
    ALL_AGGREGATION_FUSION_METHODS,
    DIVERSITY_FUSION_METHODS,
    build_embeddings_dictionary,
    build_ranking_with_fusion,
)


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
