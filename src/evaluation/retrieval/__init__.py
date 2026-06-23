"""Retrieval evaluation: three document levels sharing one base + metric set.

* :class:`SurfacedDocEvaluator` — all docs the retriever returned per step.
* :class:`SeenDocEvaluator`     — docs actually passed to the agent/LLM.
* :class:`CitedDocEvaluator`    — docs cited by the final report.
"""

from .metrics import (
    compute_trec_metrics,
    evaluate_from_ranking_results,
    evaluate_results,
    metrics_at_n,
)
from .citation_metrics import (
    extract_citations_from_text,
    resolve_cited_doc_ids,
    compute_citation_metrics,
    evaluate_citation_quality,
    compute_recall_at_n,
)
from .base import BaseDocRetrievalEvaluator
from .surfaced import SurfacedDocEvaluator
from .seen import SeenDocEvaluator
from .cited import CitedDocEvaluator

__all__ = [
    "compute_trec_metrics",
    "evaluate_from_ranking_results",
    "evaluate_results",
    "metrics_at_n",
    "extract_citations_from_text",
    "resolve_cited_doc_ids",
    "compute_citation_metrics",
    "evaluate_citation_quality",
    "compute_recall_at_n",
    "BaseDocRetrievalEvaluator",
    "SurfacedDocEvaluator",
    "SeenDocEvaluator",
    "CitedDocEvaluator",
]
