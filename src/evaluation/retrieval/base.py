"""Shared base class for the three document-level retrieval evaluators.

All three evaluators answer the same question — *how good is this set of
documents against the qrels?* — but differ only in **which** documents they
look at:

* :class:`~evaluation.retrieval.surfaced.SurfacedDocEvaluator` — every doc the
  retriever returned in each step (top-k surfaced).
* :class:`~evaluation.retrieval.seen.SeenDocEvaluator` — the subset actually
  passed to the agent/LLM (seen-top-k).
* :class:`~evaluation.retrieval.cited.CitedDocEvaluator` — the docs cited by the
  final report.

Subclasses implement a single hook, :meth:`_doc_iterations`, returning one
ranked doc list per retrieval step.  Everything else — fusing those lists into
a final ranking, TREC evaluation, Metrics@N, pretty-printing, per-query TREC
saving — lives here.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.ranking_results import RankingResults, RankingResult
from searcher_component.fusion import fuse_retrieval_results

from .metrics import evaluate_results, metrics_at_n

logger = logging.getLogger(__name__)


def _doc_id(doc) -> str:
    """Return the canonical doc_id, trying 'doc_id' then 'id'."""
    if isinstance(doc, str):
        return doc
    return doc.get("doc_id") or doc.get("id") or ""


def _rank_score(doc: Dict[str, Any], fallback_rank: int) -> float:
    """Return the rank score for a doc, trying multiple field names."""
    for key in ("rank_score", "rerank_score", "score"):
        val = doc.get(key)
        if val is not None:
            return val
    return 1.0 / fallback_rank


class BaseDocRetrievalEvaluator:
    """Evaluate retrieval quality for one document level (surfaced/seen/cited).

    Subclasses override :meth:`_doc_iterations` (required) and optionally
    :meth:`_score` (per-doc rank score) and :attr:`emit_metrics_at_n`.
    """

    #: Whether :meth:`evaluate` adds a ``Metrics@N`` block (N = #docs per query).
    emit_metrics_at_n: bool = True
    #: Default header used by :meth:`print_results`.
    default_header: str = "RETRIEVAL EVALUATION RESULTS"

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
        fusion_method: str = "interleaving",
        interleaving_window: Optional[int] = 3,
        rrf_k: int = 60,
    ):
        """Initialise the evaluator.

        Args:
            qrels:               Ground-truth relevance judgements {query_id: {doc_id: score}}.
            k_values:            Cut-off values for TREC metrics. Defaults to [1,3,5,10,25,100].
            fusion_method:       Method used to consolidate per-step ranked lists
                                 ("interleaving" or "rrf").
            interleaving_window: Block size for interleaving fusion. Ignored for rrf.
            rrf_k:               K constant for reciprocal rank fusion. Ignored otherwise.
        """
        self.qrels = qrels
        self.k_values = k_values or [1, 3, 5, 10, 25, 100]
        self.fusion_method = fusion_method
        self.interleaving_window = interleaving_window
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _doc_iterations(
        self, query_id: str, result: Dict[str, Any],
    ) -> List[List[Dict[str, Any]]]:
        """Return one ranked doc list per retrieval step for *query_id*.

        Each inner list is a list of doc dicts (or doc-id strings).  Return an
        empty list to skip the query.  Subclasses must implement this.
        """
        raise NotImplementedError

    def _score(self, doc: Dict[str, Any], rank: int) -> float:
        """Return the rank score written for *doc* at 1-based *rank*."""
        return 1.0 / rank

    # ------------------------------------------------------------------
    # Shared ranking construction
    # ------------------------------------------------------------------

    def build_ranking_results(self, results: Dict[str, Dict[str, Any]]) -> RankingResults:
        """Convert per-step doc lists into a fused :class:`RankingResults`."""
        ranking_results = RankingResults(results=[])

        for query_id, result in results.items():
            iterations = self._doc_iterations(query_id, result)
            if not iterations:
                continue

            # Track the first step each doc_id appeared in (for the run_tag).
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
                        rank_score=self._score(doc, rank),
                        metadata={"run_tag": f"iter_{iter_idx}"},
                    ))

        return ranking_results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval quality and return a metrics dict.

        Returns a flat dict with NDCG/MAP/Recall/Precision/F1/Success (each
        ``metric@k -> value``) plus ``num_queries`` and ``avg_docs_per_query``,
        and — when :attr:`emit_metrics_at_n` — a ``Metrics@N`` block.
        Returns ``{}`` when there are no results or no qrels.
        """
        ranking_results = self.build_ranking_results(results)
        if len(ranking_results.results) == 0 or len(self.qrels) == 0:
            logger.warning("No retrieval results or qrels available — skipping evaluation.")
            return {}

        num_queries = len(ranking_results.get_unique_queries())
        metrics = evaluate_results(
            results=ranking_results,
            qrels=self.qrels,
            k_values=self.k_values,
        )
        metrics["num_queries"] = num_queries
        metrics["avg_docs_per_query"] = (
            len(ranking_results.results) / num_queries if num_queries > 0 else 0.0
        )

        if self.emit_metrics_at_n:
            metrics["Metrics@N"] = metrics_at_n(self.qrels, ranking_results)

        return metrics

    def evaluate_per_query(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Evaluate retrieval quality for each query individually."""
        return {qid: self.evaluate({qid: result}) for qid, result in results.items()}

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_results(self, metrics: Dict[str, Any], header: Optional[str] = None) -> None:
        """Pretty-print retrieval evaluation metrics."""
        header = header or self.default_header
        if not metrics:
            print(f"  No retrieval results available ({self.default_header})")
            return

        scalar_keys = {"num_queries", "avg_docs_per_query", "Metrics@N", "citation_quality"}

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Queries evaluated:      {metrics.get('num_queries', 0)}")
        print(f"  Avg docs per query:     {metrics.get('avg_docs_per_query', 0.0):.1f}")

        metrics_at_n_block = metrics.get("Metrics@N")
        if metrics_at_n_block:
            print(f"  Recall@N (mean):        {metrics_at_n_block.get('Recall@N', 0):.4f}  "
                  f"(avg N={metrics_at_n_block.get('avg_N', 0):.1f})")
            print(f"  Precision@N (mean):     {metrics_at_n_block.get('Precision@N', 0):.4f}")
            print(f"  F1@N (mean):            {metrics_at_n_block.get('F1@N', 0):.4f}")
        print()
        for metric_name, metric_value in metrics.items():
            if metric_name in scalar_keys:
                continue
            if isinstance(metric_value, dict):
                key = f"{metric_name}@10"
                if key in metric_value:
                    print(f"  {key}: {metric_value[key]:.4f}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query retrieval results as a TREC file.

        Docs are written per retrieval step; the 6th column records the step
        number (``iter_1``, ``iter_2``, ...).

        Standard TREC columns: ``qid Q0 doc_id rank score run_tag``
        """
        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        iterations = self._doc_iterations(query_id, result)
        lines: List[str] = []
        for iter_idx, iteration_docs in enumerate(iterations, 1):
            for rank, doc in enumerate(iteration_docs, 1):
                did = _doc_id(doc)
                if not did:
                    continue
                score = self._score(doc, rank)
                lines.append(f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_idx}")

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with open(trec_path, "w") as f:
            f.write(content)

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Save retrieval evaluation metrics to a JSON file."""
        if not metrics:
            return
        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved retrieval metrics: {output_path_str}")
