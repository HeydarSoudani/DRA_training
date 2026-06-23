"""Generation evaluation.

* :class:`GenerationEvaluator` — basic surface stats (length, words, citations, ROUGE).
* :class:`ShortAnswerEvaluator` — LLM-as-judge short-answer correctness
  (alias: ``AccuracyEvaluator``).
* :class:`ReportEvaluator` — LLM-as-judge rubric + citation faithfulness for
  long-form reports.
"""

from .basic_stats import GenerationEvaluator
from .short_answer import ShortAnswerEvaluator, AccuracyEvaluator
from .report import ReportEvaluator

__all__ = [
    "GenerationEvaluator",
    "ShortAnswerEvaluator",
    "AccuracyEvaluator",
    "ReportEvaluator",
]
