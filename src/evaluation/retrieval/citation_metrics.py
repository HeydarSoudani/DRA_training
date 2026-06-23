"""Pure citation metric functions (no evaluator class, no I/O).

Shared by :class:`evaluation.retrieval.cited.CitedDocEvaluator` and
:class:`evaluation.generation.report.ReportEvaluator`.  Covers citation
precision/recall/F1, retrieval-usage efficiency, and Recall@N (where N = the
number of cited docs).
"""

import logging
from typing import Any, Dict, List, Optional

# Re-exported from the leaf ``utils`` package to keep ``utils`` free of any
# dependency on ``evaluation`` (avoids a utils <-> evaluation import cycle).
from utils.text_utils import extract_citations_from_text  # noqa: F401

logger = logging.getLogger(__name__)


def resolve_cited_doc_ids(result: Dict[str, Any]) -> List[str]:
    """Return the ordered, de-duplicated doc IDs cited by a result's report.

    Resolution priority (same as :class:`CitedDocEvaluator`):
    1. ``cited_docs_ranked_list`` (fused list of doc dicts).
    2. ``memory_bank`` (insertion-order doc-id keys).
    3. ``[N]`` markers parsed from ``generation``/``final_report`` mapped through
       ``citation_to_doc_id``.
    """
    cited_ranked = result.get("cited_docs_ranked_list")
    if cited_ranked:
        ids = [(d.get("doc_id") or d.get("id") or "") if isinstance(d, dict) else d
               for d in cited_ranked]
    else:
        memory_bank = result.get("memory_bank")
        if memory_bank:
            ids = list(memory_bank.keys())
        else:
            report = result.get("generation") or result.get("final_report") or ""
            citation_to_doc_id = result.get("citation_to_doc_id", {})
            ids = []
            for cit_id in extract_citations_from_text(report):
                did = citation_to_doc_id.get(cit_id) or citation_to_doc_id.get(str(cit_id))
                if did:
                    ids.append(did)

    out: List[str] = []
    seen: set = set()
    for did in ids:
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def compute_citation_metrics(
    cited_doc_ids: set[str],
    retrieved_doc_ids: set[str],
    relevant_doc_ids: set[str],
) -> dict[str, float]:
    """Compute citation-based evaluation metrics.

    Evaluates two aspects:
    1. Citation quality: How many cited docs are actually relevant?
    2. Retrieval efficiency: How many retrieved docs were actually cited?

    Returns a dict with citation_precision, citation_recall, citation_f1,
    retrieval_usage_rate, retrieval_precision, and the num_* counts.
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
        agent_results: Mapping of query_id to agent result containing
            ``final_report``, ``citation_to_doc_id``, and
            ``retrieved_docs_per_search``.
        qrels: Query relevance judgments {query_id: {doc_id: relevance}}.

    Returns:
        Aggregated citation metrics averaged across queries (``{}`` if none).
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
    ties broken by ``pool_order`` then doc_id.  Metrics are computed at cutoff
    N_q for that query, then averaged across queries.

    Returns a dict with Recall@N, Precision@N, F1@N, avg_N, per_query,
    num_queries.  Returns ``{}`` if no queries can be evaluated.
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
