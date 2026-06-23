"""Seen-doc retrieval evaluator.

"Seen" documents are the subset (capped at ``seen_top_k``) that was formatted
with citations and injected into the LLM prompt during planning/writing steps.
They are recorded in each trajectory search step as ``step["output"]["doc_ids"]``
(AgentCPM) or ``step["component_doc_ids"]`` (reasoning agents such as WebWeaver,
GLM, OSS, Tongyi).
"""

from typing import Any, Dict, List, Optional

from .base import BaseDocRetrievalEvaluator

_NON_SEARCH_ACTIONS = frozenset({
    "write", "init-plan", "init-plan-oracle", "extend-plan", "nop",
    "analyst-init_plan", "analyst-init_plan_oracle", "analyst-extend_plan",
})


def _extract_seen_iterations(
    trajectory: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Return one doc list per retrieval step using only the *seen* documents.

    Only steps that represent actual search/retrieval actions are included.
    Write, plan, and other non-search steps are skipped even if they carry
    ``component_doc_ids`` (e.g. CPM-Report write steps store cited doc IDs).

    Returns a list-of-lists of ``{"doc_id": str}`` dicts, one inner list per
    search step.
    """
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


class SeenDocEvaluator(BaseDocRetrievalEvaluator):
    """Evaluate retrieval quality restricted to *seen* documents.

    Per-step seen-doc lists are fused across iterations using the configured
    fusion method, then evaluated with standard TREC metrics plus Metrics@N.

    Usage::

        evaluator = SeenDocEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    emit_metrics_at_n = True
    default_header = "SEEN DOC RETRIEVAL EVALUATION RESULTS"

    def _doc_iterations(self, query_id: str, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        iterations = _extract_seen_iterations(result.get("trajectory", []))
        if not iterations:
            iterations = result.get("seen_docs_iterations", [])
        return iterations
