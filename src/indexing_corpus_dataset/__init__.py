"""
Data processing and format definitions for SLM retrieval research.

This module contains standardized data structures and utilities for handling
search results, relevance judgments, and other research data formats.
"""

from .ranking_results import (
    RankingResult,
    RankingResults,
    load_ranking_results,
    save_ranking_results,
)

__all__ = [
    "RankingResult",
    "RankingResults",
    "load_ranking_results",
    "save_ranking_results",
]
