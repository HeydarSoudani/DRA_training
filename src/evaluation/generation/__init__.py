"""Generation evaluation.

* :class:`GenerationEvaluator` — basic surface stats (length, words, citations, ROUGE).
* :class:`ShortAnswerEvaluator` — LLM-as-judge short-answer correctness
  (alias: ``AccuracyEvaluator``); used for BrowseComp-Plus.
* :class:`TRQAGenerationEvaluator` — rule-based numeric exact/soft match; used
  for TRQA (no LLM judge).
* :class:`ReportEvaluator` — LLM-as-judge rubric + citation faithfulness for
  long-form reports.
"""

from .basic_stats import GenerationEvaluator
from .short_answer import ShortAnswerEvaluator, AccuracyEvaluator
from .trqa_match import TRQAGenerationEvaluator
from .report import ReportEvaluator

__all__ = [
    "GenerationEvaluator",
    "ShortAnswerEvaluator",
    "AccuracyEvaluator",
    "TRQAGenerationEvaluator",
    "ReportEvaluator",
]
