"""Cited-doc retrieval evaluator.

"Cited" documents are those referenced by the final report.  The cited ranked
list is resolved in priority order:

1. ``result["cited_docs_ranked_list"]`` — properly fused list from the parallel
   pipeline (each doc may carry an ``iter`` field).
2. ``result["memory_bank"]`` — insertion-order merge from the sequential
   OutlineReduction pipeline (keys are doc IDs).
3. **General fallback** — parse ``[N]`` citation markers out of the report text
   (``generation`` / ``final_report``) and map them through
   ``result["citation_to_doc_id"]``.  This makes "cited" work for any report
   agent, not just CPM-style writers.

Queries with no cited docs (or no qrels entry) are skipped.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDocRetrievalEvaluator, _doc_id
from .citation_metrics import extract_citations_from_text


class CitedDocEvaluator(BaseDocRetrievalEvaluator):
    """Evaluate retrieval quality restricted to documents cited by the report.

    Usage::

        evaluator = CitedDocEvaluator(qrels=qrels, k_values=[1, 5, 10, 100])
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    emit_metrics_at_n = True
    default_header = "CITED DOC RETRIEVAL EVALUATION RESULTS"

    def _cited_docs(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve the ordered list of cited docs for a result (see module doc)."""
        cited_ranked = result.get("cited_docs_ranked_list")
        if cited_ranked:
            docs: List[Dict[str, Any]] = []
            for doc in cited_ranked:
                did = _doc_id(doc)
                if did:
                    docs.append({"doc_id": did, "iter": doc.get("iter", 1)})
            return docs

        memory_bank = result.get("memory_bank")
        if memory_bank:
            return [{"doc_id": did, "iter": 1} for did in memory_bank.keys() if did]

        # General fallback: parse citations out of the report text.
        report = result.get("generation") or result.get("final_report") or ""
        citation_to_doc_id = result.get("citation_to_doc_id", {})
        if not report or not citation_to_doc_id:
            return []
        docs = []
        seen: set = set()
        for cit_id in extract_citations_from_text(report):
            did = citation_to_doc_id.get(cit_id) or citation_to_doc_id.get(str(cit_id))
            if did and did not in seen:
                seen.add(did)
                docs.append({"doc_id": did, "iter": 1})
        return docs

    def _doc_iterations(self, query_id: str, result: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        if query_id not in self.qrels:
            return []
        docs = self._cited_docs(result)
        if not docs:
            return []
        # Cited docs form a single ranked list (no per-step fusion).
        return [docs]

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save cited-doc retrieval results for a single query as a TREC file.

        Overrides the base to preserve each doc's originating ``iter`` in the
        run_tag column rather than collapsing to ``iter_1``.
        """
        docs = self._cited_docs(result)
        if not docs:
            return

        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for rank, doc in enumerate(docs, 1):
            did = _doc_id(doc)
            if not did:
                continue
            score = 1.0 / rank
            iter_num = doc.get("iter", 1)
            lines.append(f"{query_id} Q0 {did} {rank} {score:.6f} iter_{iter_num}")

        content = "\n".join(lines)
        if lines:
            content += "\n"
        trec_path = f"{output_dir_str.rstrip('/')}/{query_id}.trec"
        with open(trec_path, "w") as f:
            f.write(content)
