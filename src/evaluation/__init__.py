"""Evaluation utilities for agentic retrieval research.

Modules
-------
retrieval_evaluation
    TREC metric computation (NDCG, MAP, Recall, etc.), result persistence,
    and the RetrievalEvaluator class for evaluating agents.
citation_evaluator
    Citation quality metrics (precision, recall, F1), Recall@N evaluation,
    and the CitedDocRetrievalEvaluator class for cited-document retrieval.
per_search_analysis
    Per-search-step retrieval analysis and plotting.
latency_tracker
    Per-component, per-query wall-clock latency tracking.
efficiency_tracker
    Aggregate pipeline efficiency metrics (LLM calls, tokens, retriever calls).
"""

# Retrieval evaluation (functions + class)
from .retrieval_evaluation import (
    compute_trec_metrics,
    evaluate_from_ranking_results,
    evaluate_results,
    save_results,
    RetrievalEvaluator,
    SeenDocRetrievalEvaluator,
)

# Citation evaluation (functions + class)
from .citation_evaluator import (
    extract_citations_from_text,
    compute_citation_metrics,
    evaluate_citation_quality,
    compute_recall_at_n,
    CitedDocRetrievalEvaluator,
)

# Generation evaluation
from .generation_evaluator import GenerationEvaluator

# Accuracy evaluation (LLM-as-judge)
from .accuracy_evaluator import AccuracyEvaluator

# Trajectory evaluation
from .trajectory_evaluator import TrajectoryEvaluator

# Tracker evaluation (per-iteration signal persistence)
from .tracker_evaluator import TrackerEvaluator

# Tracking
from .latency_tracker import LatencyTracker
from .efficiency_tracker import EfficiencyTracker
from .per_search_analysis import PerSearchAnalysis

__all__ = [
    # Retrieval evaluation
    "compute_trec_metrics",
    "evaluate_from_ranking_results",
    "evaluate_results",
    "save_results",
    "RetrievalEvaluator",
    "SeenDocRetrievalEvaluator",
    # Citation evaluation
    "extract_citations_from_text",
    "compute_citation_metrics",
    "evaluate_citation_quality",
    "compute_recall_at_n",
    "CitedDocRetrievalEvaluator",
    # Generation evaluation
    "GenerationEvaluator",
    # Accuracy evaluation (LLM-as-judge)
    "AccuracyEvaluator",
    # Trajectory evaluation
    "TrajectoryEvaluator",
    # Tracker evaluation
    "TrackerEvaluator",
    # Tracking
    "LatencyTracker",
    "EfficiencyTracker",
    "PerSearchAnalysis",
]
