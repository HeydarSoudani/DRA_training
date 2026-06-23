"""Controller evaluator: persist and aggregate controller signals.

Saves the per-iteration signal data produced by ``Controller``
into lightweight JSON files (one per query) under a dedicated ``controller/``
output directory.  These files are designed for direct consumption by the
analysis code (correlation, plotting) without needing to recompute signals.

Signal and action names mirror the controller exactly
(``doc_novelty``, ``consec_query_sim``, ``orig_query_sim``, ``marginal_recall``,
``controller_action`` in {continue, intervene, stop}, ``controller_reasoning``).

Per-query JSON schema::

    {
        "qid": "query_123",
        "question": "...",
        "num_iterations": 8,
        "unique_doc_count": 34,
        "unique_doc_ids": ["doc1", "doc2", ...],
        "per_iteration": {
            "0": {
                "iteration": 0,
                "subqueries": ["query about X"],
                "doc_novelty": 1.0,
                "consec_query_sim": null,
                "orig_query_sim": 0.72,
                "marginal_recall": 0.4,
                "num_new_relevant": 2,
                "num_docs_this_step": 5,
                "answer_candidates": [
                    {"candidate": "some answer", "reasoning": "reasoning text", "confidence": 75.0}
                ],
                "criteria_coverage": {
                    "criteria": [{"name": "...", "status": "covered", "evidence": "..."}],
                    "num_covered": 1, "num_partial": 0, "num_not_covered": 2,
                    "total": 3, "critical_gaps": ["..."], "minor_gaps": [],
                    ...
                }
            },
            "1": {
                "iteration": 1,
                "subqueries": ["sq1", "sq2", "sq3"],
                ...
            },
            ...
        }
    }
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


def _controller_score_history(result: Dict[str, Any]):
    """Return the controller score history, accepting the legacy key name."""
    return result.get("controller_score_history") or result.get("tracker_score_history")


class ControllerEvaluator:
    """Persist and aggregate controller signal data.

    Usage::

        evaluator = ControllerEvaluator()

        # Per-query save (inside the query loop)
        evaluator.save_item(query_id, query_text, result, controller_dir)

        # Aggregate evaluation (after all queries)
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    # ------------------------------------------------------------------
    # Per-query persistence
    # ------------------------------------------------------------------

    def save_item(
        self,
        query_id: str,
        question: str,
        result: Dict[str, Any],
        output_dir,
    ) -> None:
        """Save per-query controller signals as a JSON file.

        Reads ``controller_score_history``, ``controller_unique_doc_ids``, and
        ``controller_unique_doc_count`` from *result* (attached by
        ``_attach_controller_stats`` in the agent mixin; legacy ``tracker_*``
        keys are still accepted).

        Supports both local paths and S3 URIs.

        Args:
            query_id:   Query identifier.
            question:   Original query text.
            result:     Unified agent result dict for this query.
            output_dir: Directory where ``{query_id}.json`` will be written.
        """
        score_history = _controller_score_history(result)
        if not score_history:
            return

        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        per_iteration: Dict[str, Any] = {}
        for idx, scores in enumerate(score_history):
            iter_key = str(scores.get("iter_num", idx))
            entry: Dict[str, Any] = {
                "iteration": scores.get("iter_num", idx),
                "subqueries": scores.get("subqueries", []),
                "doc_novelty": scores.get("doc_novelty"),
                "num_novel_docs": scores.get("num_novel_docs"),
                "consec_query_sim": scores.get("consec_query_sim"),
                "orig_query_sim": scores.get("orig_query_sim"),
                "marginal_recall": scores.get("marginal_recall"),
                "num_new_relevant": scores.get("num_new_relevant"),
                "num_repeated_relevant": scores.get("num_repeated_relevant"),
                "num_irrelevant": scores.get("num_irrelevant"),
                "num_docs_this_step": scores.get("num_docs_this_step"),
                "controller_action": scores.get("controller_action"),
                "controller_reasoning": scores.get("controller_reasoning"),
                "answer_candidates": scores.get("answer_candidates", []),
                "criteria_coverage": scores.get("criteria_coverage"),
            }
            per_iteration[iter_key] = entry

        item: Dict[str, Any] = {
            "qid": query_id,
            "question": question,
            "num_iterations": len(per_iteration),
            "unique_doc_count": result.get("controller_unique_doc_count",
                                           result.get("tracker_unique_doc_count", 0)),
            "unique_doc_ids": result.get("controller_unique_doc_ids",
                                         result.get("tracker_unique_doc_ids", [])),
            "per_iteration": per_iteration,
        }

        json_path = f"{output_dir_str.rstrip('/')}/{query_id}.json"
        with open(json_path, "w") as f:
            json.dump(item, f, separators=(",", ":"), default=str)

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate controller statistics over all queries.

        Looks for ``controller_score_history`` (legacy: ``tracker_score_history``)
        in each result dict.

        Returns:
            Nested metrics dict with per-signal statistics.
        """
        if not results:
            return {}

        # Collect per-query summaries
        queries_with_controller = 0
        total_iterations: List[int] = []
        unique_doc_counts: List[int] = []

        # Per-signal value accumulators (across all iterations of all queries)
        all_doc_novelty: List[float] = []
        all_consec_query_sim: List[float] = []
        all_orig_query_sim: List[float] = []
        all_marginal_recall: List[float] = []

        # Answer candidate accumulators
        queries_no_candidate = 0
        queries_with_candidate = 0

        for query_id, result in results.items():
            score_history = _controller_score_history(result)
            if not score_history:
                continue

            queries_with_controller += 1
            total_iterations.append(len(score_history))
            unique_doc_counts.append(result.get("controller_unique_doc_count",
                                                 result.get("tracker_unique_doc_count", 0)))

            query_has_candidate = False
            for scores in score_history:
                if scores.get("doc_novelty") is not None:
                    all_doc_novelty.append(scores["doc_novelty"])
                if scores.get("consec_query_sim") is not None:
                    all_consec_query_sim.append(scores["consec_query_sim"])
                if scores.get("orig_query_sim") is not None:
                    all_orig_query_sim.append(scores["orig_query_sim"])
                if scores.get("marginal_recall") is not None:
                    all_marginal_recall.append(scores["marginal_recall"])
                if scores.get("answer_candidates"):
                    query_has_candidate = True

            if query_has_candidate:
                queries_with_candidate += 1
            else:
                queries_no_candidate += 1

        if queries_with_controller == 0:
            return {}

        def _stats(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            arr = np.array(values, dtype=float)
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "median": round(float(np.median(arr)), 4),
            }

        return {
            "num_queries_with_controller": queries_with_controller,
            "iterations_per_query": _stats(
                [float(x) for x in total_iterations]
            ),
            "unique_docs_per_query": _stats(
                [float(x) for x in unique_doc_counts]
            ),
            "doc_novelty": _stats(all_doc_novelty),
            "consec_query_sim": _stats(all_consec_query_sim),
            "orig_query_sim": _stats(all_orig_query_sim),
            "marginal_recall": _stats(all_marginal_recall),
            "answer_candidate": {
                "queries_no_candidate": queries_no_candidate,
                "queries_with_candidate": queries_with_candidate,
            },
        }

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "CONTROLLER STATISTICS",
    ) -> None:
        """Pretty-print aggregate controller statistics."""
        if not metrics:
            print("  (no controller data available)")
            return

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)

        n = metrics.get("num_queries_with_controller", 0)
        print(f"  Queries with controller data: {n}")

        def _fmt(d: Optional[Dict]) -> str:
            if not d:
                return "n/a"
            return (
                f"{d['mean']:.3f} +/- {d['std']:.3f}"
                f"  (min: {d['min']:.3f}, max: {d['max']:.3f}, med: {d['median']:.3f})"
            )

        iters = metrics.get("iterations_per_query")
        print(f"  Iterations per query:     {_fmt(iters)}")

        udocs = metrics.get("unique_docs_per_query")
        print(f"  Unique docs per query:    {_fmt(udocs)}")

        signal_keys = ["doc_novelty", "consec_query_sim", "orig_query_sim", "marginal_recall"]
        print("  Signal statistics (across all iterations):")
        for name in signal_keys:
            stats = metrics.get(name)
            if stats:
                print(f"    {name:<24s}: {_fmt(stats)}")

        ac = metrics.get("answer_candidate", {})
        if ac:
            print(f"  Answer candidates:")
            print(f"    queries_no_candidate:   {ac.get('queries_no_candidate', 0)}")
            print(f"    queries_with_candidate: {ac.get('queries_with_candidate', 0)}")

        print("=" * 80)
