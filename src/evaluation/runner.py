"""evaluation.runner — evaluation orchestration for the inference pipeline.

Wires the evaluator classes together and runs them over a completed inference
run.  Used by ``experiments/dra_inference.py``:

    build_evaluators                Instantiate Retrieval/Generation/Trajectory/
                                    CitedDoc/SeenDoc/Accuracy evaluators.
    load_processed_results          Reload saved files for resumed queries.
    evaluate_and_save               Run all evaluations and write summary.json.

Related code now lives elsewhere:
    * Reranker construction      → ``searcher_component.rerankers.build_reranker_from_config``
    * Multi-method fusion eval   → ``evaluation.retrieval.fusion``
both imported directly by the callers that need them.
"""

import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.io_utils import (
    load_result_from_trec,
    load_result_from_saved_files,
)

from . import (
    SurfacedDocEvaluator,
    GenerationEvaluator,
    TrajectoryEvaluator,
    ControllerEvaluator,
    CitedDocEvaluator,
    SeenDocEvaluator,
    AccuracyEvaluator,
    ReportEvaluator,
)


# ===========================================================================
# Evaluator construction + results loading / evaluation
# ===========================================================================

def build_evaluators(qrels: Dict, kwargs: Dict[str, Any], answers: Optional[Dict[str, str]] = None, questions: Optional[Dict[str, str]] = None) -> Tuple[SurfacedDocEvaluator, GenerationEvaluator, TrajectoryEvaluator, CitedDocEvaluator, SeenDocEvaluator, Optional[AccuracyEvaluator], Optional[ReportEvaluator], ControllerEvaluator]:
    """Instantiate Retrieval, Generation, Trajectory, CitedDoc, SeenDoc, Accuracy, Report, and Controller evaluators.

    Args:
        qrels:     Qrels dict loaded from the dataset.
        kwargs:    Pipeline configuration dict.
        answers:   Optional mapping of query_id -> ground-truth answer.
                   When provided (e.g. for BrowseComp-Plus), an AccuracyEvaluator
                   (short-answer correctness) is created.
        questions: Optional mapping of query_id -> question text.

    Returns:
        ``(retrieval_evaluator, generation_evaluator, trajectory_evaluator,
          cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator,
          report_evaluator, controller_evaluator)`` where ``accuracy_evaluator``
        is None when no answers are available, and ``report_evaluator`` is None
        unless ``kwargs["report_eval"]`` is set.
    """
    k_values = kwargs.get("k_values", [1, 3, 5, 10, 25, 100])
    retrieval_evaluator = SurfacedDocEvaluator(
        qrels=qrels,
        k_values=k_values,
        fusion_method=kwargs.get("consolidation_fusion_method", "interleaving"),
        interleaving_window=kwargs.get("interleaving_window", 3),
        rrf_k=kwargs.get("rrf_k", 60),
        eval_top_k=kwargs.get("eval_top_k"),
    )
    cited_doc_evaluator = CitedDocEvaluator(qrels=qrels, k_values=k_values)
    seen_doc_evaluator = SeenDocEvaluator(
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

    report_evaluator = None
    if kwargs.get("report_eval"):
        report_kwargs: Dict[str, Any] = {}
        judge_api_url = kwargs.get("judge_api_url")
        if judge_api_url:
            report_kwargs["judge_api_base"] = judge_api_url
        report_evaluator = ReportEvaluator(
            questions=questions,
            qrels=qrels,
            **report_kwargs,
        )

    return retrieval_evaluator, GenerationEvaluator(), TrajectoryEvaluator(), cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator, report_evaluator, ControllerEvaluator()


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


def evaluate_and_save(results: Dict[str, Any], generation_evaluator: GenerationEvaluator, trajectory_evaluator: TrajectoryEvaluator, run_dir: Optional[Path], cited_doc_evaluator: Optional[CitedDocEvaluator] = None, seen_doc_evaluator: Optional[SeenDocEvaluator] = None, accuracy_evaluator: Optional[AccuracyEvaluator] = None, controller_metrics: Optional[Dict[str, Any]] = None, controller_evaluator=None, report_evaluator: Optional[ReportEvaluator] = None) -> None:
    """Run all evaluations, print results, and save summary.json.

    Retrieval metrics are produced per fusion method by
    :func:`run_fusion_eval` and stored under the ``"retrieval"`` key in
    summary.json.  This function handles generation, trajectory,
    cited-doc, seen-doc, accuracy, and controller evaluation.

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
        controller_metrics:    Pre-computed controller metrics dict (legacy).
        controller_evaluator:  Optional ControllerEvaluator instance.  When
                               provided, ``controller_metrics`` is ignored and
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

    report_metrics = {}
    if report_evaluator is not None:
        report_metrics = report_evaluator.evaluate(results)

    if controller_evaluator is not None:
        controller_metrics = controller_evaluator.evaluate(results)

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
    if controller_metrics:
        _evaluator_counts["controller"] = controller_metrics.get("num_queries_with_controller", 0)

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

    # -- report (long-form generation)
    if report_evaluator is not None:
        report_evaluator.print_results(report_metrics)

    # -- trajectory
    trajectory_evaluator.print_results(trajectory_metrics)

    # -- controller
    if controller_metrics:
        if controller_evaluator is not None:
            controller_evaluator.print_results(controller_metrics)
        else:
            print(f"\n{'=' * 80}")
            print("CONTROLLER STATISTICS")
            print("=" * 80)
            for k, v in controller_metrics.items():
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

    # -- Save report detail file
    if report_metrics and run_dir and report_evaluator is not None:
        report_path = str(Path(str(run_dir)) / "report_eval.json")
        report_evaluator.save_results(report_metrics, report_path)

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

        if report_metrics:
            summary["report"] = {
                "num_evaluated": report_metrics.get("num_evaluated", 0),
                "rubric": report_metrics.get("rubric", {}),
                "citation_faithfulness": report_metrics.get("citation_faithfulness"),
            }

        if trajectory_metrics:
            summary["trajectory"] = trajectory_metrics

        if controller_metrics:
            summary["controller"] = controller_metrics

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
