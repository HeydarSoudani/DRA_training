"""Evaluation utilities for agentic retrieval research.

Subsystems
----------
retrieval (package)
    Three document-level evaluators sharing one base class and metric set:
    SurfacedDocEvaluator (retriever-returned), SeenDocEvaluator (passed to the
    agent), CitedDocEvaluator (cited by the report), plus the TREC metric
    functions and the pure citation metric functions
    (``retrieval.citation_metrics``: precision/recall/F1, Recall@N).  The
    multi-method fusion evaluation lives in ``retrieval.fusion`` and is imported
    explicitly (``from evaluation.retrieval.fusion import ...``) to keep
    ``import evaluation`` free of heavy searcher/indexing dependencies.
generation (package)
    ``generation.basic_stats.GenerationEvaluator`` (length, words, citations,
    optional ROUGE), ``generation.short_answer.ShortAnswerEvaluator``
    (LLM-as-judge short-answer correctness; alias ``AccuracyEvaluator``), and
    ``generation.report.ReportEvaluator`` (LLM-judge rubric + citation
    faithfulness for long-form reports).
trajectory_evaluator
    Per-step trajectory statistics (incl. token usage) and persistence.
controller_evaluator
    Per-iteration controller-signal persistence and aggregation.

Per-query/per-step token usage is recorded by the agents (via
``utils.token_meter.TokenMeter`` attached to the LLM clients) and aggregated by
``TrajectoryEvaluator``.
"""

# Retrieval evaluation (functions + classes; legacy aliases preserved)
from .retrieval import (
    compute_trec_metrics,
    evaluate_from_ranking_results,
    evaluate_results,
    metrics_at_n,
    BaseDocRetrievalEvaluator,
    SurfacedDocEvaluator,
    SeenDocEvaluator,
    CitedDocEvaluator,
)

# Pure citation metric functions (live under the retrieval subpackage)
from .retrieval.citation_metrics import (
    extract_citations_from_text,
    compute_citation_metrics,
    evaluate_citation_quality,
    compute_recall_at_n,
)

# Generation evaluation
from .generation import (
    GenerationEvaluator,
    ShortAnswerEvaluator,
    AccuracyEvaluator,
    ReportEvaluator,
)

# Trajectory evaluation
from .trajectory_evaluator import TrajectoryEvaluator

# Controller evaluation (per-iteration signal persistence)
from .controller_evaluator import ControllerEvaluator

__all__ = [
    # Retrieval evaluation
    "compute_trec_metrics",
    "evaluate_from_ranking_results",
    "evaluate_results",
    "metrics_at_n",
    "BaseDocRetrievalEvaluator",
    "SurfacedDocEvaluator",
    "SeenDocEvaluator",
    "CitedDocEvaluator",
    # Citation metric functions
    "extract_citations_from_text",
    "compute_citation_metrics",
    "evaluate_citation_quality",
    "compute_recall_at_n",
    # Generation evaluation
    "GenerationEvaluator",
    "ShortAnswerEvaluator",
    "AccuracyEvaluator",
    "ReportEvaluator",
    # Trajectory evaluation
    "TrajectoryEvaluator",
    # Controller evaluation
    "ControllerEvaluator",
]
