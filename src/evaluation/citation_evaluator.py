"""Citation evaluation utilities.

Provides functions for evaluating the quality of citations in generated reports,
including citation precision/recall, Recall@N (where N = number of cited docs),
and the :class:`CitedDocRetrievalEvaluator` class for evaluating cited-document
retrieval quality via the reduction component's memory bank.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from agentic_retrieval_research.utils.s3_utils import is_s3_path, s3_open, s3_makedirs

logger = logging.getLogger(__name__)


def extract_citations_from_text(text: str) -> set[int]:
    """Extract citation IDs from text (e.g., [1], [22], [6]).

    Args:
        text: Text containing citation markers like [1], [22], etc.

    Returns:
        Set of citation IDs found in the text
    """
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return set(int(m) for m in matches)


def compute_citation_metrics(
    cited_doc_ids: set[str],
    retrieved_doc_ids: set[str],
    relevant_doc_ids: set[str],
) -> dict[str, float]:
    """Compute citation-based evaluation metrics.

    Evaluates two aspects:
    1. Citation quality: How many cited docs are actually relevant?
    2. Retrieval efficiency: How many retrieved docs were actually cited?

    Args:
        cited_doc_ids: Document IDs cited in the final output
        retrieved_doc_ids: Document IDs retrieved during the process
        relevant_doc_ids: Ground truth relevant document IDs from qrels

    Returns:
        Dictionary containing:
        - citation_precision: % of cited docs that are relevant
        - citation_recall: % of relevant docs that were cited
        - citation_f1: F1 score combining precision and recall
        - retrieval_usage_rate: % of retrieved docs that were cited
        - retrieval_precision: % of retrieved docs that are relevant
    """
    metrics = {}

    # Citation quality metrics (cited vs relevant)
    if cited_doc_ids:
        cited_and_relevant = cited_doc_ids & relevant_doc_ids
        metrics["citation_precision"] = len(cited_and_relevant) / len(cited_doc_ids)
    else:
        metrics["citation_precision"] = 0.0

    if relevant_doc_ids:
        cited_and_relevant = cited_doc_ids & relevant_doc_ids
        metrics["citation_recall"] = len(cited_and_relevant) / len(relevant_doc_ids)
    else:
        metrics["citation_recall"] = 0.0

    # Citation F1
    p = metrics["citation_precision"]
    r = metrics["citation_recall"]
    if p + r > 0:
        metrics["citation_f1"] = 2 * p * r / (p + r)
    else:
        metrics["citation_f1"] = 0.0

    # Retrieval efficiency metrics
    if retrieved_doc_ids:
        cited_and_retrieved = cited_doc_ids & retrieved_doc_ids
        metrics["retrieval_usage_rate"] = len(cited_and_retrieved) / len(retrieved_doc_ids)

        retrieved_and_relevant = retrieved_doc_ids & relevant_doc_ids
        metrics["retrieval_precision"] = len(retrieved_and_relevant) / len(retrieved_doc_ids)
    else:
        metrics["retrieval_usage_rate"] = 0.0
        metrics["retrieval_precision"] = 0.0

    # Additional stats
    metrics["num_cited"] = len(cited_doc_ids)
    metrics["num_retrieved"] = len(retrieved_doc_ids)
    metrics["num_relevant"] = len(relevant_doc_ids)

    return metrics


def evaluate_citation_quality(
    agent_results: dict[str, dict],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Evaluate citation quality across all queries.

    Args:
        agent_results: Dictionary mapping query_id to agent result containing:
            - final_report: The generated report text
            - citation_to_doc_id: Mapping from citation_id to doc_id
            - retrieved_docs_per_search: List of retrieved docs per search
        qrels: Query relevance judgments {query_id: {doc_id: relevance}}

    Returns:
        Dictionary containing aggregated citation metrics averaged across queries
    """
    all_metrics = []

    for query_id, result in agent_results.items():
        if query_id not in qrels:
            continue

        # Extract cited doc IDs
        final_report = result.get("final_report", "")
        citation_to_doc_id = result.get("citation_to_doc_id", {})
        cited_citation_ids = extract_citations_from_text(final_report)

        # Map citation IDs to doc IDs
        cited_doc_ids = set()
        for cit_id in cited_citation_ids:
            doc_id = citation_to_doc_id.get(cit_id) or citation_to_doc_id.get(str(cit_id))
            if doc_id:
                cited_doc_ids.add(doc_id)

        # Extract all retrieved doc IDs
        retrieved_docs_per_search = result.get("retrieved_docs_per_search", [])
        retrieved_doc_ids = set()
        for search_docs in retrieved_docs_per_search:
            for keyword_docs in search_docs:
                for doc in keyword_docs:
                    doc_id = doc.get("doc_id") or doc.get("id")
                    if doc_id:
                        retrieved_doc_ids.add(doc_id)

        # Get relevant doc IDs from qrels
        relevant_doc_ids = set(qrels[query_id].keys())

        # Compute metrics for this query
        query_metrics = compute_citation_metrics(
            cited_doc_ids=cited_doc_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
        )
        all_metrics.append(query_metrics)

    # Aggregate metrics across queries
    if not all_metrics:
        return {}

    aggregated = {}
    for key in all_metrics[0].keys():
        aggregated[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return aggregated


def compute_recall_at_n(
    doc_citation_counts: Dict[str, Dict[str, int]],
    qrels: Dict[str, Dict[str, int]],
    query_ids: Optional[List[str]] = None,
    pool_order: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Compute Recall@N, Precision@N, F1@N where N = num cited docs per query.

    For each query, N_q is the number of documents with at least one citation
    (citation count > 0).  Documents are ranked by citation count (descending),
    with ties broken by ``pool_order`` if provided, then by doc_id.
    Metrics are computed at cutoff N_q for that query, then averaged across queries.

    This is a general-purpose function for any pipeline that assigns citation
    counts to documents (e.g., report-aware reranking, CPM-style writers).

    Args:
        doc_citation_counts: ``{query_id: {doc_id: citation_count}}``
        qrels: ``{query_id: {doc_id: relevance}}``
        query_ids: Query IDs to evaluate (defaults to all keys in doc_citation_counts).
        pool_order: Optional ``{query_id: {doc_id: rank_position}}`` for tie-breaking.

    Returns:
        Dict with keys: Recall@N, Precision@N, F1@N, avg_N, per_query, num_queries.
        Returns ``{}`` if no queries can be evaluated.
    """
    if query_ids is None:
        query_ids = list(doc_citation_counts.keys())

    per_query_results: List[Dict[str, Any]] = []

    for qid in query_ids:
        if qid not in doc_citation_counts or qid not in qrels:
            continue

        counts = doc_citation_counts[qid]

        # N_q = number of docs cited at least once
        n_q = sum(1 for c in counts.values() if c > 0)

        if n_q == 0:
            per_query_results.append({
                "query_id": qid, "N": 0,
                "recall": 0.0, "precision": 0.0, "f1": 0.0,
            })
            continue

        # Get relevant doc IDs for this query
        relevant_docs = set(qrels[qid].keys())
        if not relevant_docs:
            per_query_results.append({
                "query_id": qid, "N": n_q,
                "recall": 0.0, "precision": 0.0, "f1": 0.0,
            })
            continue

        # Build ranked list sorted by citation count (descending),
        # then by pool order for ties, then by doc_id
        qid_pool = pool_order.get(qid, {}) if pool_order else {}
        ranked_doc_ids = sorted(
            counts.keys(),
            key=lambda d: (-counts[d], qid_pool.get(d, 9999), d),
        )

        # Top N_q docs
        top_n = set(ranked_doc_ids[:n_q])

        # Compute metrics
        hits = len(top_n & relevant_docs)
        recall = hits / len(relevant_docs)
        precision = hits / n_q
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_query_results.append({
            "query_id": qid, "N": n_q,
            "recall": recall, "precision": precision, "f1": f1,
        })

    if not per_query_results:
        return {}

    num_queries = len(per_query_results)
    avg_n = sum(r["N"] for r in per_query_results) / num_queries
    avg_recall = sum(r["recall"] for r in per_query_results) / num_queries
    avg_precision = sum(r["precision"] for r in per_query_results) / num_queries
    avg_f1 = sum(r["f1"] for r in per_query_results) / num_queries

    return {
        "Recall@N": round(avg_recall, 5),
        "Precision@N": round(avg_precision, 5),
        "F1@N": round(avg_f1, 5),
        "avg_N": round(avg_n, 2),
        "num_queries": num_queries,
        "per_query": per_query_results,
    }


# ---------------------------------------------------------------------------
# CitedDocRetrievalEvaluator — evaluates retrieval using cited/seen documents
# ---------------------------------------------------------------------------

class CitedDocRetrievalEvaluator:
    """Evaluate retrieval quality restricted to cited documents (memory bank).

    Evaluates retrieval quality using only the documents cited/seen by the
    agent via the reduction component's memory bank.

    The ``memory_bank`` field is populated in the unified result only when the
    ``OutlineReduction`` strategy is used.  For all other reduction strategies
    it will be ``None``, and those queries are skipped during evaluation.

    The memory bank maps ``doc_id -> {"title": str, "context": str}`` for every
    document cited at least once across all outline-update iterations.

    Only queries whose ``result["memory_bank"]`` is a non-empty dict are
    included.  Queries without a memory bank (other reduction strategies or
    no citations) are silently skipped.

    Usage::

        evaluator = CitedDocRetrievalEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
    ) -> None:
        self.qrels = qrels
        self.k_values = k_values or [1, 3, 5, 10, 25, 100]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval quality for cited documents.

        For each query with a non-None ``memory_bank``, the cited doc IDs
        (keys of the memory bank) are treated as a ranked retrieval list.
        Docs are assigned descending scores by insertion order so that
        first-cited documents rank highest.

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            Metrics dict with keys ``"NDCG"``, ``"MAP"``, ``"Recall"``,
            ``"Precision"``, ``"F1"``, ``"Success"`` (each mapping
            ``"metric@k" -> value``), plus summary stats
            ``"num_queries_with_memory_bank"`` and ``"avg_cited_docs"``.
            Returns ``{}`` when no queries have a memory bank.
        """
        # Lazy import to avoid circular dependency with retrieval_evaluation
        from .retrieval_evaluation import compute_trec_metrics

        search_results: Dict[str, Dict[str, float]] = {}
        num_cited_per_query: List[int] = []

        for query_id, result in results.items():
            if query_id not in self.qrels:
                continue

            # Prefer cited_docs_ranked_list (properly fused, from parallel pipeline)
            # over memory_bank (insertion-order merge, from sequential pipeline).
            cited_ranked = result.get("cited_docs_ranked_list")
            if cited_ranked:
                doc_ids = [
                    doc.get("doc_id") or doc.get("id") or ""
                    for doc in cited_ranked
                ]
                doc_ids = [d for d in doc_ids if d]
            else:
                memory_bank = result.get("memory_bank")
                if not memory_bank:
                    continue
                doc_ids = list(memory_bank.keys())

            if not doc_ids:
                continue

            num_cited_per_query.append(len(doc_ids))
            search_results[query_id] = {
                doc_id: 1.0 / (rank + 1)
                for rank, doc_id in enumerate(doc_ids)
            }

        if not search_results:
            return {}

        ndcg, map_scores, recall, precision, f1, success = compute_trec_metrics(
            qrels=self.qrels,
            results=search_results,
            k_values=self.k_values,
        )

        # Metrics@N: N = number of cited passages per query (varies per query)
        recall_at_n_values: List[float] = []
        precision_at_n_values: List[float] = []
        f1_at_n_values: List[float] = []
        n_values: List[int] = []
        for query_id, cited_scores in search_results.items():
            gold = self.qrels.get(query_id, {})
            gold_ids = {doc_id for doc_id, rel in gold.items() if rel > 0}
            if not gold_ids:
                continue
            cited_ids = set(cited_scores.keys())
            n_q = len(cited_ids)
            n_values.append(n_q)
            hits = len(gold_ids & cited_ids)
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

        return {
            "NDCG": ndcg,
            "MAP": map_scores,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Success": success,
            "Metrics@N": metrics_at_n_summary,
            "num_queries": len(search_results),
            "avg_docs_per_query": (
                sum(num_cited_per_query) / len(num_cited_per_query)
                if num_cited_per_query else 0.0
            ),
        }

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "CITED DOC RETRIEVAL EVALUATION RESULTS",
    ) -> None:
        """Pretty-print cited-doc retrieval metrics."""
        if not metrics:
            print("  No cited-doc retrieval results available (no memory bank or cited_docs_ranked_list)")
            return

        scalar_keys = {"num_queries", "avg_docs_per_query", "Metrics@N"}

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Queries evaluated:      {metrics.get('num_queries', 0)}")
        print(f"  Avg docs per query:     {metrics.get('avg_docs_per_query', 0.0):.1f}")
        metrics_at_n = metrics.get("Metrics@N", {})
        if metrics_at_n:
            print(f"  Recall@N (mean):        {metrics_at_n.get('Recall@N', 0):.4f}  (avg N={metrics_at_n.get('avg_N', 0):.1f})")
            print(f"  Precision@N (mean):     {metrics_at_n.get('Precision@N', 0):.4f}")
            print(f"  F1@N (mean):            {metrics_at_n.get('F1@N', 0):.4f}")
        print()
        for metric_name, metric_value in metrics.items():
            if metric_name in scalar_keys:
                continue
            if isinstance(metric_value, dict):
                key = f"{metric_name}@10"
                if key in metric_value:
                    print(f"  {key}: {metric_value[key]:.4f}")
        print("=" * 80)

    def save_item(
        self,
        query_id: str,
        result: Dict[str, Any],
        output_dir,
    ) -> None:
        """Save cited-doc retrieval results for a single query as a TREC file.

        Supports both local paths and S3 URIs.
        """
        cited_ranked = result.get("cited_docs_ranked_list")
        if not cited_ranked:
            memory_bank = result.get("memory_bank")
            if not memory_bank:
                return
            cited_ranked = [{"doc_id": doc_id, "iter": 1} for doc_id in memory_bank.keys() if doc_id]

        if not cited_ranked:
            return

        output_dir_str = str(output_dir)
        s3_makedirs(output_dir_str)

        lines: List[str] = []
        for rank, doc in enumerate(cited_ranked, 1):
            doc_id = doc.get("doc_id") or doc.get("id") or ""
            if not doc_id:
                continue
            score = 1.0 / rank
            iter_num = doc.get("iter", 1)
            lines.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} iter_{iter_num}")

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with s3_open(trec_path, "w") as f:
            f.write(content)
