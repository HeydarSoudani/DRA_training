"""Surfaced-doc retrieval evaluator.

"Surfaced" documents are **every** doc the retriever returned in each step
(top-k surfaced, before the agent down-selects what it actually reads).  They
are stored in each trajectory step as ``step["docs"]`` (flat) or
``step["all_docs"]`` (flat list of dicts, or list-of-lists).
"""

from typing import Any, Dict, List, Optional

from .base import BaseDocRetrievalEvaluator, _doc_id, _rank_score


def _extract_trajectory_iterations(
    trajectory: List[Dict[str, Any]],
    eval_top_k: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Return one surfaced doc list per retrieval step from a trajectory.

    Args:
        trajectory: List of per-step dicts from agent execution.
        eval_top_k: If set, keep only the first *eval_top_k* docs per step.
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


class SurfacedDocEvaluator(BaseDocRetrievalEvaluator):
    """Evaluate retrieval quality over all surfaced (retriever-returned) docs.

    Per-step ranked lists are extracted from ``result["trajectory"]`` and fused
    using the configured fusion method.  When citation data is present, citation
    quality metrics are appended.

    Usage::

        evaluator = SurfacedDocEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
    """

    emit_metrics_at_n = False
    default_header = "RETRIEVAL EVALUATION RESULTS"

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
        fusion_method: str = "interleaving",
        interleaving_window: Optional[int] = 3,
        rrf_k: int = 60,
        eval_top_k: Optional[int] = None,
    ):
        """See :class:`BaseDocRetrievalEvaluator`.

        Args:
            eval_top_k: Number of docs kept per step for evaluation.
                        None means no limit (use all surfaced docs).
        """
        super().__init__(
            qrels=qrels,
            k_values=k_values,
            fusion_method=fusion_method,
            interleaving_window=interleaving_window,
            rrf_k=rrf_k,
        )
        self.eval_top_k = eval_top_k

    def _doc_iterations(self, query_id: str, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        iterations = _extract_trajectory_iterations(
            result.get("trajectory", []), eval_top_k=self.eval_top_k,
        )
        # Fall back to final_ranked_list (e.g. outline reduction stores no "docs"
        # in its trajectory steps but always populates final_ranked_list).
        if not iterations:
            final_rl = result.get("final_ranked_list", [])
            if final_rl:
                iterations = [final_rl]
        return iterations

    def _score(self, doc: Dict[str, Any], rank: int) -> float:
        return _rank_score(doc, rank)

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate surfaced-doc retrieval, plus citation quality when available."""
        metrics = super().evaluate(results)
        if not metrics:
            return metrics

        # Citation quality — only when at least one result carries citation data.
        has_citations = any(result.get("citation_to_doc_id") for result in results.values())
        if has_citations:
            from .citation_metrics import evaluate_citation_quality

            citation_metrics = evaluate_citation_quality(results, self.qrels)
            if citation_metrics:
                metrics["citation_quality"] = citation_metrics

        return metrics

    def print_results(self, metrics: Dict[str, Any], header: Optional[str] = None) -> None:
        super().print_results(metrics, header=header)
        if metrics and "citation_quality" in metrics:
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
