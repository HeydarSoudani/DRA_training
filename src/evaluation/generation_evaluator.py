"""Generation evaluator for deep-research agents.

Accepts the **unified** result format produced by all agents:

    {
        query_id: {
            "generation":        str  – main text output (report or short answer)
            "citation_to_doc_id": Dict  – optional (AgentCPM)
            "survey":            Dict  – optional (AgentCPM)
        }
    }

Basic stats are always computed.  ROUGE scores are computed when the optional
``rouge_score`` package is available **and** reference texts are supplied.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


logger = logging.getLogger(__name__)


def _count_words(text: str) -> int:
    return len(text.split()) if text else 0


def _count_citations(text: str) -> int:
    """Count citation markers of the form [N]."""
    return len(re.findall(r"\[\d+\]", text))


class GenerationEvaluator:
    """Evaluate generation quality from the unified agent result format.

    Usage::

        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate(results)

        # With reference texts (enables ROUGE):
        references = {"q1": "reference answer …", "q2": "…"}
        evaluator = GenerationEvaluator(references=references)
        metrics = evaluator.evaluate(results)
    """

    def __init__(self, references: Optional[Dict[str, str]] = None):
        """Initialise the evaluator.

        Args:
            references: Optional mapping of ``query_id → reference text`` used for
                        ROUGE computation.  When *None* only basic stats are returned.
        """
        self.references = references or {}

        # Try to import rouge_score; silently skip if not installed.
        self._rouge_scorer = None
        if self.references:
            try:
                from rouge_score import rouge_scorer  # type: ignore
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
            except ImportError:
                logger.warning(
                    "rouge_score not installed — ROUGE metrics will be skipped. "
                    "Install with: pip install rouge-score"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute generation quality metrics aggregated over all queries.

        Always returns:
        - ``avg_generation_length`` (chars)
        - ``avg_generation_words``
        - ``avg_citations`` (citation markers in text, when present)
        - ``num_queries``

        When references are supplied and ``rouge_score`` is installed:
        - ``rouge1``, ``rouge2``, ``rougeL``  (average F1 across queries)

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            Flat dict of aggregated generation metrics.
        """
        char_lengths: List[int] = []
        word_counts: List[int] = []
        citation_counts: List[int] = []
        rouge1_scores: List[float] = []
        rouge2_scores: List[float] = []
        rougeL_scores: List[float] = []

        for query_id, result in results.items():
            generation = result.get("generation", "")

            char_lengths.append(len(generation))
            word_counts.append(_count_words(generation))

            # Citation count (only meaningful if citations are present)
            n_cit = _count_citations(generation)
            if n_cit > 0 or result.get("citation_to_doc_id"):
                citation_counts.append(n_cit)

            # ROUGE (requires references)
            if self._rouge_scorer and query_id in self.references:
                ref = self.references[query_id]
                scores = self._rouge_scorer.score(ref, generation)
                rouge1_scores.append(scores["rouge1"].fmeasure)
                rouge2_scores.append(scores["rouge2"].fmeasure)
                rougeL_scores.append(scores["rougeL"].fmeasure)

        def _avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        metrics: Dict[str, Any] = {
            "num_queries": len(results),
            "avg_generation_length": _avg(char_lengths),
            "avg_generation_words": _avg(word_counts),
        }

        if citation_counts:
            metrics["avg_citations"] = _avg(citation_counts)

        if rouge1_scores:
            metrics["rouge1"] = round(_avg(rouge1_scores), 5)
            metrics["rouge2"] = round(_avg(rouge2_scores), 5)
            metrics["rougeL"] = round(_avg(rougeL_scores), 5)

        return metrics

    def print_results(self, metrics: Dict[str, Any], header: str = "GENERATION STATISTICS") -> None:
        """Pretty-print generation metrics.

        Args:
            metrics: Output of :meth:`evaluate`.
            header:  Section header string.
        """
        if not metrics:
            print("  ⚠ No generation metrics available")
            return

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Queries evaluated:      {metrics.get('num_queries', 0)}")
        print(f"  Avg generation length:  {metrics.get('avg_generation_length', 0):.0f} chars")
        print(f"  Avg generation words:   {metrics.get('avg_generation_words', 0):.0f} words")
        if "avg_citations" in metrics:
            print(f"  Avg citations per doc:  {metrics.get('avg_citations', 0):.1f}")
        if "rouge1" in metrics:
            print(f"  ROUGE-1:  {metrics['rouge1']:.4f}")
            print(f"  ROUGE-2:  {metrics['rouge2']:.4f}")
            print(f"  ROUGE-L:  {metrics['rougeL']:.4f}")
        print("=" * 80)

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query generation output as a Markdown file.

        Saves:
        - ``{query_id}.md``: Raw generation text.

        Supports both local paths and S3 URIs.

        Args:
            query_id:   Query identifier.
            result:     Unified agent result dict for this query.
            output_dir: Directory where files will be written.
        """
        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)
        generation = result.get("generation", "")

        # Save raw markdown
        md_path = f"{output_dir_str.rstrip('/')}/{query_id}.md"
        with open(md_path, "w") as f:
            f.write(generation)

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Save generation metrics to a JSON file.

        Args:
            metrics:     Output of :meth:`evaluate`.
            output_path: Destination file path (parents are created if needed).
        """
        if not metrics:
            return
        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  ✓ Saved generation metrics: {output_path_str}")
