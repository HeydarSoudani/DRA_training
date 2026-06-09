"""Retrieval evaluation utilities.

This module provides TREC metric computation, result formatting, result persistence,
and the :class:`RetrievalEvaluator` class for evaluating agentic pipelines.

Core evaluation functions modified from:
https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py
"""

import json
import logging
import pytrec_eval
from pathlib import Path
from typing import Dict, List, Any, Optional

from agentic_retrieval_research.utils.s3_utils import is_s3_path, s3_open, s3_makedirs

from ..corpus_dataset.ranking_results import RankingResults, RankingResult, save_ranking_results
from ..searcher_component.fusion import fuse_retrieval_results
from .efficiency_tracker import EfficiencyTracker

logger = logging.getLogger(__name__)


def compute_trec_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
    ignore_identical_ids: bool = True,
    include_all_metrics: bool = True,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Compute TREC evaluation metrics using pytrec_eval (low-level function).

    This is the core evaluation function that computes metrics from raw dictionaries.
    Returns a 6-tuple: (NDCG, MAP, Recall, Precision, F1, Success).

    Args:
        qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}
        results: Search results {query_id: {doc_id: score}}
        k_values: List of k values for metrics (default: [1, 3, 5, 10, 25, 100])
        ignore_identical_ids: Whether to ignore query-doc pairs with identical IDs

    Returns:
        Tuple of (NDCG, MAP, Recall, Precision, F1, Success) dictionaries

    Note: This is a low-level function. For most use cases, prefer:
    - `evaluate_with_logging()` for experiment runs with logging
    - `evaluate_from_ranking_results()` for structured dict output
    """
    k_values = sorted(set(k_values)) if k_values else [1, 3, 5, 10, 25, 100]

    if ignore_identical_ids:
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    f1 = {}
    success = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
        success[f"Success@{k}"] = 0.0

    if include_all_metrics:
        # Add @all metrics (computed across all documents without cutoff)
        ndcg["NDCG@all"] = 0.0
        _map["MAP@all"] = 0.0
        recall["Recall@all"] = 0.0
        precision["P@all"] = 0.0
        success["Success@all"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    success_string = "success." + ",".join([str(k) for k in k_values])
    # NOTE: recall must be evaluated in a separate pytrec_eval call from map_cut.
    # When map_cut.k and recall.k share the same k in one evaluator, pytrec_eval
    # overwrites the recall_k key with MAP-related intermediate values (e.g. 34.5),
    # producing recall > 1. Using a dedicated recall-only evaluator avoids this collision.
    measures = {map_string, ndcg_string, precision_string, success_string}
    if include_all_metrics:
        measures |= {"ndcg", "map"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scores = evaluator.evaluate(results)

    recall_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {recall_string})
    recall_scores = recall_evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += recall_scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
            success[f"Success@{k}"] += scores[query_id]["success_" + str(k)]

        if include_all_metrics:
            # Add @all metrics (without cutoff)
            ndcg["NDCG@all"] += scores[query_id].get("ndcg", 0.0)
            _map["MAP@all"] += scores[query_id].get("map", 0.0)

            # For Recall@all, Precision@all, Success@all: compute across all retrieved docs
            if query_id in qrels and query_id in results:
                relevant_docs = set(doc_id for doc_id, rel in qrels[query_id].items() if rel > 0)
                retrieved_docs = results[query_id].keys()
                num_retrieved = len(retrieved_docs)

                if num_retrieved > 0:
                    relevant_retrieved = relevant_docs & set(retrieved_docs)
                    recall["Recall@all"] += len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
                    precision["P@all"] += len(relevant_retrieved) / num_retrieved
                    success["Success@all"] += 1.0 if len(relevant_retrieved) > 0 else 0.0

    # Guard against division by zero if no scores
    if len(scores) == 0:
        return ndcg, _map, recall, precision, f1, success

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
        success[f"Success@{k}"] = round(success[f"Success@{k}"] / len(scores), 5)
        f1[f"F1@{k}"] = round(
            (
                2
                * (precision[f"P@{k}"] * recall[f"Recall@{k}"])
                / (precision[f"P@{k}"] + recall[f"Recall@{k}"])
                if (precision[f"P@{k}"] + recall[f"Recall@{k}"]) > 0
                else 0
            ),
            5,
        )

    if include_all_metrics:
        # Compute averaged @all metrics
        ndcg["NDCG@all"] = round(ndcg["NDCG@all"] / len(scores), 5)
        _map["MAP@all"] = round(_map["MAP@all"] / len(scores), 5)
        recall["Recall@all"] = round(recall["Recall@all"] / len(scores), 5)
        precision["P@all"] = round(precision["P@all"] / len(scores), 5)
        success["Success@all"] = round(success["Success@all"] / len(scores), 5)
        f1["F1@all"] = round(
            (
                2
                * (precision["P@all"] * recall["Recall@all"])
                / (precision["P@all"] + recall["Recall@all"])
                if (precision["P@all"] + recall["Recall@all"]) > 0
                else 0
            ),
            5,
        )

    return ndcg, _map, recall, precision, f1, success


def evaluate_from_ranking_results(
    qrels: dict[str, dict[str, int]],
    results: RankingResults,
    k_values: list[int] | None = None,
    ignore_identical_ids: bool = True,
) -> dict[str, dict[int, float]]:
    """Evaluate RankingResults and return metrics in structured format (mid-level function).

    Converts RankingResults to dict format, computes metrics, and returns them
    structured by metric name and k value: {metric_name: {k: value}}.

    Args:
        qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}
        results: RankingResults object containing retrieval results
        k_values: List of k values for metrics (default: [1, 3, 5, 10, 25, 100])
        ignore_identical_ids: Whether to ignore query-doc pairs with identical IDs

    Returns:
        Dictionary structured as {metric_name: {k: value}}
        Example: {"NDCG": {1: 0.5, 5: 0.6}, "Recall": {1: 0.3, 5: 0.4}}
    """
    # Normalize scores so that pytrec_eval (which re-sorts by score) sees the
    # same ordering as our rank column.
    results.normalize_scores_by_rank()

    search_results = {
        query_id: {
            result.doc_id: result.rank_score
            for result in results.get_results_for_query(query_id)
        }
        for query_id in results.get_unique_queries()
    }
    evaluation_results = compute_trec_metrics(
        qrels=qrels,
        results=search_results,
        k_values=k_values,
        ignore_identical_ids=ignore_identical_ids,
    )
    final_results = {}
    for metric_dict in evaluation_results:
        if not metric_dict:
            continue
        for metric_name, metric_value in metric_dict.items():
            m, k = metric_name.split("@") if "@" in metric_name else (metric_name, "0")
            # Keep "all" as string, convert numeric k values to int
            if k != "all":
                try:
                    k = int(k)
                except ValueError:
                    k = 0
            if m not in final_results:
                final_results[m] = {}
            final_results[m][k] = metric_value
    return final_results


def evaluate_results(
    results: RankingResults,
    qrels: Dict[str, Dict[str, int]],
    k_values: list,
    ignore_identical_ids: bool = True,
) -> Dict[str, Any]:
    """Evaluate RankingResults with logging and formatted output (high-level function).

    This is the recommended function for experiment runs. It evaluates retrieval results,
    logs key metrics to console, and returns formatted metrics for saving.

    Args:
        results: RankingResults object containing retrieval results
        qrels: Query relevance judgments mapping query_id -> doc_id -> relevance_score
        k_values: List of k values for evaluation metrics (e.g., [1, 5, 10])
        ignore_identical_ids: Whether to ignore query-doc pairs with identical IDs

    Returns:
        Dictionary containing evaluation metrics (NDCG, MAP, Recall, Precision, F1, Success)
        Format: {metric@k: value} e.g., {"NDCG@5": 0.6, "Recall@10": 0.7}
    """
    logger.info("Evaluating retrieval results...")

    # Normalize scores so that pytrec_eval (which re-sorts by score) sees the
    # same ordering as our rank column.
    results.normalize_scores_by_rank()

    # Convert RankingResults to format expected by compute_trec_metrics function
    search_results = {}
    for query_id in results.get_unique_queries():
        query_results = results.get_results_for_query(query_id)
        search_results[query_id] = {
            result.doc_id: result.rank_score
            for result in query_results
        }

    # Run evaluation
    ndcg, map_scores, recall, precision, f1, success = compute_trec_metrics(
        qrels=qrels,
        results=search_results,
        k_values=k_values,
        ignore_identical_ids=ignore_identical_ids,
    )

    # Combine all metrics
    evaluation_results = {
        "NDCG": ndcg,
        "MAP": map_scores,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Success": success,
    }

    # Log key metrics
    logger.info("Evaluation Results:")
    for k in [1, 5, 10]:
        if f"NDCG@{k}" in ndcg:
            logger.info(f"  NDCG@{k}: {ndcg[f'NDCG@{k}']:.4f}")
        if f"Recall@{k}" in recall:
            logger.info(f"  Recall@{k}: {recall[f'Recall@{k}']:.4f}")

    return evaluation_results


def save_results(
    results: RankingResults,
    evaluation_results: Dict[str, Any],
    output_dir,
    tracker: EfficiencyTracker = None,
    llm_outputs: list = None,
):
    """Save retrieval results and evaluation metrics to disk or S3.

    Saves results in multiple formats:
    - JSON format with full metadata
    - TREC format for standard IR evaluation
    - JSONL with LLM outputs per query (if provided)
    - Summary JSON with combined experiment info and metrics

    Supports both local paths and S3 URIs.

    Args:
        results: RankingResults object containing retrieval results
        evaluation_results: Dictionary of evaluation metrics
        output_dir: Directory to save results to (local or S3)
        tracker: Optional EfficiencyTracker for performance metrics
        llm_outputs: Optional list of LLM outputs per query
    """
    output_dir_str = str(output_dir)
    _is_s3 = is_s3_path(output_dir_str)

    if _is_s3:
        from agentic_retrieval_research.utils.s3_utils import s3_write_text
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_dir_str}/...")

    # Save as JSON
    json_path = f"{output_dir_str.rstrip('/')}/results.json" if _is_s3 else output_dir / "results.json"
    save_ranking_results(results, json_path, format_type="json")
    logger.info(f"  ✓ Saved JSON: {json_path}")

    # Save as TREC format
    trec_path = f"{output_dir_str.rstrip('/')}/results.trec" if _is_s3 else output_dir / "results.trec"
    save_ranking_results(results, trec_path, format_type="trec")
    logger.info(f"  ✓ Saved TREC: {trec_path}")

    # Save LLM outputs as JSONL (one query per line)
    if llm_outputs:
        llm_content = "\n".join(json.dumps(o) for o in llm_outputs) + "\n"
        if _is_s3:
            llm_path = f"{output_dir_str.rstrip('/')}/llm_outputs.jsonl"
            s3_write_text(llm_path, llm_content)
        else:
            llm_path = output_dir / "llm_outputs.jsonl"
            with open(llm_path, "w", encoding="utf-8") as f:
                f.write(llm_content)
        logger.info(f"  ✓ Saved LLM outputs: {llm_path}")

    # Save combined summary (includes metrics)
    summary = {
        "experiment_info": results.experiment_info,
        "evaluation_metrics": evaluation_results,
        "num_queries": len(results.get_unique_queries()),
        "num_results": len(results.results),
    }

    # Add efficiency metrics if tracking is enabled
    if tracker and tracker.enabled:
        num_queries = len(results.get_unique_queries())
        summary["efficiency"] = {
            "aggregate": tracker.get_summary(),
            "per_query": tracker.get_per_query_metrics(num_queries),
        }

    summary_content = json.dumps(summary, indent=2)
    if _is_s3:
        summary_path = f"{output_dir_str.rstrip('/')}/summary.json"
        s3_write_text(summary_path, summary_content)
    else:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
    logger.info(f"  ✓ Saved summary: {summary_path}")

    logger.info("All results saved successfully")


# ---------------------------------------------------------------------------
# RetrievalEvaluator — class-based evaluator for agentic pipelines
# ---------------------------------------------------------------------------

def _doc_id(doc) -> str:
    """Return the canonical doc_id, trying 'doc_id' then 'id'."""
    if isinstance(doc, str):
        return doc
    return doc.get("doc_id") or doc.get("id") or ""


def _collect_trajectory_docs(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect all docs from trajectory steps, deduplicated by doc_id (first occurrence wins)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for step in trajectory:
        if "all_docs" in step and step["all_docs"]:
            all_docs = step["all_docs"]
            if all_docs and isinstance(all_docs[0], dict):
                doc_list = all_docs
            else:
                doc_list = [doc for retrieve_docs in all_docs for doc in retrieve_docs]
            for doc in doc_list:
                did = _doc_id(doc)
                if did and did not in seen:
                    seen[did] = doc
        elif "docs" in step and step["docs"]:
            for doc in step["docs"]:
                did = _doc_id(doc)
                if did and did not in seen:
                    seen[did] = doc
    return list(seen.values())


def _extract_seen_iterations(
    trajectory: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Return one doc list per retrieval step using only the *seen* documents.

    "Seen" documents are those actually passed to the LLM (capped at
    ``seen_top_k`` by the agent).  They are stored in the trajectory as
    ``step["output"]["doc_ids"]`` (AgentCPM) or ``step["component_doc_ids"]``
    (reasoning agents such as WebWeaver, GLM, OSS, Tongyi).

    Only steps that represent actual search/retrieval actions are included.
    Write, plan, and other non-search steps are skipped even if they carry
    ``component_doc_ids`` (e.g. CPM-Report write steps store cited doc IDs).

    Returns a list-of-lists of ``{"doc_id": str}`` dicts, one inner list
    per search step, compatible with the fusion and evaluation functions
    that expect doc dicts.
    """
    _NON_SEARCH_ACTIONS = frozenset({
        "write", "init-plan", "init-plan-oracle", "extend-plan", "nop",
        "analyst-init_plan", "analyst-init_plan_oracle", "analyst-extend_plan",
    })

    iterations: List[List[Dict[str, Any]]] = []
    for step in trajectory:
        action = step.get("action_type") or step.get("action") or step.get("_cpm_state") or ""
        if action in _NON_SEARCH_ACTIONS:
            continue

        # AgentCPM stores seen doc IDs in step["output"]["doc_ids"]
        output = step.get("output", {})
        doc_ids = output.get("doc_ids") if isinstance(output, dict) else None
        # Reasoning agents store them in step["component_doc_ids"]
        if not doc_ids:
            doc_ids = step.get("component_doc_ids")
        if doc_ids:
            iterations.append([{"doc_id": did} for did in doc_ids if did])
    return iterations


def _extract_trajectory_iterations(
    trajectory: List[Dict[str, Any]],
    eval_top_k: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Return one doc list per retrieval step from a trajectory.

    Args:
        trajectory: List of per-step dicts from agent execution.
        eval_top_k: If set, keep only the first *eval_top_k* docs per
            iteration for evaluation.  ``None`` means no limit.
    """
    iterations = []
    for step in trajectory:
        if "all_docs" in step and step["all_docs"]:
            all_docs = step["all_docs"]
            if all_docs and isinstance(all_docs[0], dict):
                turn_docs = all_docs
            else:
                turn_docs = [doc for retrieve_docs in all_docs for doc in retrieve_docs]
            if turn_docs:
                if eval_top_k is not None:
                    turn_docs = turn_docs[:eval_top_k]
                iterations.append(turn_docs)
        elif "docs" in step and step["docs"]:
            docs = step["docs"]
            if eval_top_k is not None:
                docs = docs[:eval_top_k]
            iterations.append(docs)
    return iterations


def _rank_score(doc: Dict[str, Any], fallback_rank: int) -> float:
    """Return the rank score for a doc, trying multiple field names."""
    for key in ("rank_score", "rerank_score", "score"):
        val = doc.get(key)
        if val is not None:
            return val
    return 1.0 / fallback_rank


class RetrievalEvaluator:
    """Evaluate retrieval quality from the unified agent result format.

    Accepts the **unified** result format produced by agents:

        {
            query_id: {
                "trajectory":         List[Dict]  -- per-step reasoning trace; each step may
                                                   contain "docs" (List[Dict]) or
                                                   "all_docs" (List[List[Dict]])
                # Optional citation extras:
                "citation_to_doc_id":  Dict[int|str, str]
                "retrieved_docs_per_search": ...
            }
        }

    Usage::

        evaluator = RetrievalEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        ranking_results = evaluator.build_ranking_results(results)
        metrics = evaluator.evaluate(results)
    """

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
        fusion_method: str = "interleaving",
        interleaving_window: Optional[int] = 3,
        rrf_k: int = 60,
        eval_top_k: Optional[int] = None,
    ):
        """Initialise the evaluator.

        Args:
            qrels:               Ground-truth relevance judgements {query_id: {doc_id: score}}.
            k_values:            Cut-off values for TREC metrics.  Defaults to [1,3,5,10,25,100].
            fusion_method:       Method used to consolidate per-iteration ranked lists when
                                 building the final ranking ("interleaving" or "rrf").
            interleaving_window: Block size for interleaving fusion -- consecutive items taken
                                 from each list per round (default: 3).  Ignored for rrf.
            rrf_k:               K constant for reciprocal rank fusion (default: 60).
                                 Ignored when fusion_method is not "rrf".
            eval_top_k:          Number of docs kept per iteration for evaluation.
                                 None means no limit (use all retrieved docs).
        """
        self.qrels = qrels
        self.k_values = k_values or [1, 3, 5, 10, 25, 100]
        self.fusion_method = fusion_method
        self.interleaving_window = interleaving_window
        self.rrf_k = rrf_k
        self.eval_top_k = eval_top_k

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def build_ranking_results(self, results: Dict[str, Dict[str, Any]]) -> RankingResults:
        """Convert the unified results dict into a :class:`RankingResults` object.

        Per-iteration ranked lists are extracted from ``result["trajectory"]`` and
        fused using :attr:`fusion_method` (``"interleaving"`` or ``"rrf"``).
        When only a single iteration is present the list is used as-is.

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            :class:`RankingResults` ready for TREC evaluation or file saving.
        """
        ranking_results = RankingResults(results=[])

        for query_id, result in results.items():
            iterations = _extract_trajectory_iterations(
                result.get("trajectory", []), eval_top_k=self.eval_top_k,
            )
            # Fall back to final_ranked_list (e.g. outline reduction stores no "docs"
            # in its trajectory steps but always populates final_ranked_list).
            if not iterations:
                final_rl = result.get("final_ranked_list", [])
                if final_rl:
                    iterations = [final_rl]
            if not iterations:
                continue
            # Track the first iteration each doc_id appeared in (for run_tag)
            doc_first_iter: Dict[str, int] = {}
            for iter_idx, iteration_docs in enumerate(iterations, 1):
                for doc in iteration_docs:
                    did = _doc_id(doc)
                    if did and did not in doc_first_iter:
                        doc_first_iter[did] = iter_idx
            if len(iterations) == 1:
                docs = iterations[0]
            else:
                docs = fuse_retrieval_results(
                    iterations,
                    fusion_method=self.fusion_method,
                    rrf_k=self.rrf_k,
                    interleaving_window=self.interleaving_window,
                )
            for rank, doc in enumerate(docs, 1):
                did = _doc_id(doc)
                if did:
                    iter_idx = doc_first_iter.get(did, 1)
                    ranking_results.add_result(RankingResult(
                        query_id=query_id,
                        doc_id=did,
                        rank=rank,
                        rank_score=_rank_score(doc, rank),
                        metadata={"run_tag": f"iter_{iter_idx}"},
                    ))

        return ranking_results

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval quality and return a metrics dict.

        Runs TREC evaluation (NDCG, MAP, Recall, Precision, F1, Success) and,
        when citation data is present, also citation-quality metrics.

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            Flat metrics dict, e.g.::

                {
                    "NDCG": {"NDCG@5": 0.42, ...},
                    "Recall": {"Recall@100": 0.70, ...},
                    ...
                    "citation_quality": {...},  # only when citations available
                }
        """
        ranking_results = self.build_ranking_results(results)
        evaluation_metrics: Dict[str, Any] = {}

        if len(ranking_results.results) == 0 or len(self.qrels) == 0:
            logger.warning("No retrieval results or qrels available — skipping evaluation.")
            return evaluation_metrics

        num_eval_queries = len(ranking_results.get_unique_queries())
        avg_docs_per_query = (
            len(ranking_results.results) / num_eval_queries if num_eval_queries > 0 else 0.0
        )
        logger.info(f"Evaluating retrieval for {num_eval_queries} queries ...")

        evaluation_metrics = evaluate_results(
            results=ranking_results,
            qrels=self.qrels,
            k_values=self.k_values,
        )
        evaluation_metrics["num_queries"] = num_eval_queries
        evaluation_metrics["avg_docs_per_query"] = avg_docs_per_query

        # Citation quality — only when at least one result carries citation data
        # Lazy import to avoid circular dependency with citation_evaluator
        has_citations = any(result.get("citation_to_doc_id") for result in results.values())
        if has_citations:
            from .citation_evaluator import evaluate_citation_quality

            logger.info("Evaluating citation quality ...")
            citation_metrics = evaluate_citation_quality(results, self.qrels)
            if citation_metrics:
                evaluation_metrics["citation_quality"] = citation_metrics

        return evaluation_metrics

    def evaluate_per_query(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Evaluate retrieval quality for each query individually.

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            Dict keyed by query_id, each value being the metrics dict from :meth:`evaluate`.
        """
        per_query: Dict[str, Dict[str, Any]] = {}
        for query_id, result in results.items():
            per_query[query_id] = self.evaluate({query_id: result})
        return per_query

    def print_results(self, metrics: Dict[str, Any], header: str = "RETRIEVAL EVALUATION RESULTS") -> None:
        """Pretty-print retrieval evaluation metrics."""
        if not metrics:
            print("  No retrieval results or qrels available for evaluation")
            return

        scalar_keys = {"num_queries", "avg_docs_per_query", "citation_quality"}

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Queries evaluated:      {metrics.get('num_queries', 0)}")
        print(f"  Avg docs per query:     {metrics.get('avg_docs_per_query', 0.0):.1f}")
        print()
        for metric_name, metric_value in metrics.items():
            if metric_name in scalar_keys:
                continue
            if isinstance(metric_value, dict):
                key = f"{metric_name}@10"
                if key in metric_value:
                    print(f"  {key}: {metric_value[key]:.4f}")
        print("=" * 80)

        if "citation_quality" in metrics:
            citation_metrics = metrics["citation_quality"]
            print("\n" + "=" * 80)
            print("CITATION QUALITY EVALUATION")
            print("=" * 80)
            print(f"  Citation Precision:   {citation_metrics.get('citation_precision', 0):.4f}")
            print(f"  Citation Recall:      {citation_metrics.get('citation_recall', 0):.4f}")
            print(f"  Citation F1:          {citation_metrics.get('citation_f1', 0):.4f}")
            print(f"  Retrieval Usage Rate: {citation_metrics.get('retrieval_usage_rate', 0):.4f}")
            print(f"  Retrieval Precision:  {citation_metrics.get('retrieval_precision', 0):.4f}")
            print(f"  Avg Cited Docs:       {citation_metrics.get('num_cited', 0):.1f}")
            print(f"  Avg Retrieved Docs:   {citation_metrics.get('num_retrieved', 0):.1f}")
            print(f"  Avg Relevant Docs:    {citation_metrics.get('num_relevant', 0):.1f}")
            print("=" * 80)

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query retrieval results as a TREC file.

        Docs are written per retrieval step from ``result["trajectory"]``; the
        6th column records the step number (``iter_1``, ``iter_2``, ...).

        Standard TREC columns: ``qid Q0 doc_id rank score run_tag``

        Supports both local paths and S3 URIs.
        """
        output_dir_str = str(output_dir)
        s3_makedirs(output_dir_str)

        iterations = _extract_trajectory_iterations(
            result.get("trajectory", []), eval_top_k=self.eval_top_k,
        )
        if not iterations:
            final_rl = result.get("final_ranked_list", [])
            if final_rl:
                iterations = [final_rl]
        lines: List[str] = []
        for iter_idx, iteration_docs in enumerate(iterations, 1):
            for rank, doc in enumerate(iteration_docs, 1):
                did = _doc_id(doc)
                if not did:
                    continue
                score = _rank_score(doc, rank)
                lines.append(
                    f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_idx}"
                )

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with s3_open(trec_path, "w") as f:
            f.write(content)

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Save retrieval evaluation metrics to a JSON file.

        Supports both local paths and S3 URIs.
        """
        if not metrics:
            return
        output_path_str = str(output_path)
        if is_s3_path(output_path_str):
            from agentic_retrieval_research.utils.s3_utils import s3_write_text
            s3_write_text(output_path_str, json.dumps(metrics, indent=2, default=str))
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved retrieval metrics: {output_path_str}")


# ---------------------------------------------------------------------------
# SeenDocRetrievalEvaluator — evaluates only the docs the LLM actually saw
# ---------------------------------------------------------------------------

class SeenDocRetrievalEvaluator:
    """Evaluate retrieval quality restricted to *seen* documents.

    "Seen" documents are the subset (capped at ``seen_top_k``) that was
    formatted with citations and injected into the LLM prompt during
    planning/writing steps.  They are recorded in each trajectory search
    step as ``step["output"]["doc_ids"]``.

    Per-step seen-doc lists are fused across iterations using the
    configured fusion method, then evaluated with standard TREC metrics
    (NDCG, MAP, Recall, Precision, F1, Success).

    Usage::

        evaluator = SeenDocRetrievalEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
        fusion_method: str = "interleaving",
        interleaving_window: Optional[int] = 3,
        rrf_k: int = 60,
    ):
        self.qrels = qrels
        self.k_values = k_values or [1, 3, 5, 10, 25, 100]
        self.fusion_method = fusion_method
        self.interleaving_window = interleaving_window
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_ranking_results(self, results: Dict[str, Dict[str, Any]]) -> RankingResults:
        """Convert seen-doc iterations into a :class:`RankingResults` object."""
        ranking_results = RankingResults(results=[])

        for query_id, result in results.items():
            iterations = _extract_seen_iterations(result.get("trajectory", []))
            if not iterations:
                iterations = result.get("seen_docs_iterations", [])
            if not iterations:
                continue

            doc_first_iter: Dict[str, int] = {}
            for iter_idx, iteration_docs in enumerate(iterations, 1):
                for doc in iteration_docs:
                    did = _doc_id(doc)
                    if did and did not in doc_first_iter:
                        doc_first_iter[did] = iter_idx

            if len(iterations) == 1:
                docs = iterations[0]
            else:
                docs = fuse_retrieval_results(
                    iterations,
                    fusion_method=self.fusion_method,
                    rrf_k=self.rrf_k,
                    interleaving_window=self.interleaving_window,
                )

            for rank, doc in enumerate(docs, 1):
                did = _doc_id(doc)
                if did:
                    iter_idx = doc_first_iter.get(did, 1)
                    ranking_results.add_result(RankingResult(
                        query_id=query_id,
                        doc_id=did,
                        rank=rank,
                        rank_score=1.0 / rank,
                        metadata={"run_tag": f"iter_{iter_idx}"},
                    ))

        return ranking_results

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval quality for seen documents.

        Returns:
            Metrics dict with NDCG, MAP, Recall, Precision, F1, Success,
            plus ``num_queries`` and ``avg_docs_per_query``.
        """
        ranking_results = self.build_ranking_results(results)
        if len(ranking_results.results) == 0 or len(self.qrels) == 0:
            return {}

        evaluation_metrics = evaluate_results(
            results=ranking_results,
            qrels=self.qrels,
            k_values=self.k_values,
        )
        num_queries = len(ranking_results.get_unique_queries())
        evaluation_metrics["num_queries"] = num_queries
        evaluation_metrics["avg_docs_per_query"] = (
            len(ranking_results.results) / num_queries if num_queries > 0 else 0.0
        )

        # Metrics@N: N = number of seen docs per query (varies per query)
        recall_at_n_values: List[float] = []
        precision_at_n_values: List[float] = []
        f1_at_n_values: List[float] = []
        n_values: List[int] = []
        for query_id in ranking_results.get_unique_queries():
            gold = self.qrels.get(query_id, {})
            gold_ids = {doc_id for doc_id, rel in gold.items() if rel > 0}
            if not gold_ids:
                continue
            seen_ids = {r.doc_id for r in ranking_results.get_results_for_query(query_id)}
            n_q = len(seen_ids)
            n_values.append(n_q)
            hits = len(gold_ids & seen_ids)
            r = hits / len(gold_ids)
            p = hits / n_q if n_q > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            recall_at_n_values.append(r)
            precision_at_n_values.append(p)
            f1_at_n_values.append(f)

        metrics_at_n_summary: Dict[str, Any] = {}
        if recall_at_n_values:
            num_q = len(recall_at_n_values)
            metrics_at_n_summary = {
                "Recall@N": sum(recall_at_n_values) / num_q,
                "Precision@N": sum(precision_at_n_values) / num_q,
                "F1@N": sum(f1_at_n_values) / num_q,
                "avg_N": sum(n_values) / num_q,
                "num_queries": num_q,
            }
        evaluation_metrics["Metrics@N"] = metrics_at_n_summary

        return evaluation_metrics

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "SEEN DOC RETRIEVAL EVALUATION RESULTS",
    ) -> None:
        """Pretty-print seen-doc retrieval metrics."""
        if not metrics:
            print("  No seen-doc retrieval results available")
            return

        scalar_keys = {"num_queries", "avg_docs_per_query"}

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Queries evaluated:      {metrics.get('num_queries', 0)}")
        print(f"  Avg docs per query:     {metrics.get('avg_docs_per_query', 0.0):.1f}")
        print()
        for metric_name, metric_value in metrics.items():
            if metric_name in scalar_keys or metric_name == "Metrics@N":
                continue
            if isinstance(metric_value, dict):
                key = f"{metric_name}@10"
                if key in metric_value:
                    print(f"  {key}: {metric_value[key]:.4f}")
        metrics_at_n = metrics.get("Metrics@N")
        if metrics_at_n:
            print(f"  Recall@N:  {metrics_at_n['Recall@N']:.4f}  "
                  f"Precision@N: {metrics_at_n['Precision@N']:.4f}  "
                  f"F1@N: {metrics_at_n['F1@N']:.4f}  "
                  f"(avg_N={metrics_at_n['avg_N']:.1f})")
        print("=" * 80)

    def save_item(
        self,
        query_id: str,
        result: Dict[str, Any],
        output_dir,
    ) -> None:
        """Save seen-doc retrieval results for a single query as a TREC file.

        Supports both local paths and S3 URIs.
        """
        iterations = _extract_seen_iterations(result.get("trajectory", []))
        if not iterations:
            return

        output_dir_str = str(output_dir)
        s3_makedirs(output_dir_str)

        lines: List[str] = []
        for iter_idx, iteration_docs in enumerate(iterations, 1):
            for rank, doc in enumerate(iteration_docs, 1):
                did = _doc_id(doc)
                if not did:
                    continue
                score = 1.0 / rank
                lines.append(f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_idx}")

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with s3_open(trec_path, "w") as f:
            f.write(content)
