"""Rule-based generation evaluator for TRQA (Total-Recall-QA).

TRQA answers are numeric, so correctness is judged by **numeric matching** rather
than an LLM-as-judge.  The matching logic (number extraction + exact / tolerant
comparison) is ported verbatim from the official Total-Recall-QA repository:

    https://github.com/mahta-r/total-recall-qa/blob/main/c5_task_evaluation/metrics/generation_eval_metrics.py

Metrics reported (matching ``run_evalution.py`` in that repo):
    * ``exact_match``        — fraction with an exact numeric match.
    * ``soft_exact_match``   — fraction within each tolerance % in
                               ``[1, 5, 10, 20, 50, 90]``.

The agent's final short answer is extracted from the (possibly long) generation
with :func:`controller_component.prompts.answer_prompts.extract_answer_candidates`
(the same ``\\boxed{}`` / ``<answer>`` / ``Exact Answer:`` parsers the controller
uses), falling back to the raw generation text when no structured answer is found.

The public interface mirrors
:class:`evaluation.generation.short_answer.AccuracyEvaluator`
(``evaluate`` / ``print_results`` / ``save_results`` / ``save_item``) so the
evaluation runner can use it interchangeably in the accuracy slot.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tolerance percentages reported by the TRQA reference (run_evalution.py).
DEFAULT_TOLERANCE_PCTS = (1.0, 5.0, 10.0, 20.0, 50.0, 90.0)


# ---------------------------------------------------------------------------
# Numeric matching — ported verbatim from total-recall-qa generation_eval_metrics
# ---------------------------------------------------------------------------

def normalize_number(value, decimals=2):
    """Rounds numeric value to fixed decimals for comparison."""
    if value is None or math.isnan(value):
        return None
    return round(float(value), decimals)


def safe_float_convert(value):
    """Safely convert a value to float, return None if not possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# Regex to find first number in text (int or decimal, optional minus, optional comma thousands)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)")


def extract_number_from_string(s):
    """Extract the first number from a string, stripping text/units.

    E.g. "8269 km" -> 8269.0, "1,234.5 units" -> 1234.5.  Returns float or None.
    """
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    s = str(s).strip()
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    num_str = m.group(0).replace(",", "")
    try:
        return float(num_str)
    except ValueError:
        return None


def soft_exact_match(prediction, gold, decimals=3, tolerance_pct=None):
    """Compute soft exact match with optional tolerance.

    Returns a dict with ``exact_match`` (bool) and, when *tolerance_pct* is
    given, ``soft_match`` (bool within ``tolerance_pct`` % of gold).
    """
    pred_float = extract_number_from_string(prediction)
    gold_float = extract_number_from_string(gold)

    # Fallback: pure numeric string without extra text
    if pred_float is None:
        pred_float = safe_float_convert(prediction)
    if gold_float is None:
        gold_float = safe_float_convert(gold)

    if pred_float is None or gold_float is None:
        result = {"exact_match": False}
        if tolerance_pct is not None:
            result["soft_match"] = False
        return result

    npred = normalize_number(pred_float, decimals)
    ngold = normalize_number(gold_float, decimals)

    if npred is None or ngold is None:
        result = {"exact_match": False}
        if tolerance_pct is not None:
            result["soft_match"] = False
        return result

    exact = npred == ngold
    result = {"exact_match": exact}

    if tolerance_pct is not None:
        abs_err = abs(npred - ngold)
        soft = abs_err <= (tolerance_pct / 100.0) * abs(ngold) if ngold != 0 else (abs_err == 0)
        result["soft_match"] = soft

    return result


# ---------------------------------------------------------------------------
# Final-answer extraction from the agent generation
# ---------------------------------------------------------------------------

def _extract_prediction(generation: str) -> str:
    """Pull the agent's final short answer from its (possibly long) generation.

    Reuses the controller's structured-answer parsers (``\\boxed{}``,
    ``<answer>``, ``Exact Answer:`` …).  Falls back to the raw text — with any
    appended ``## References`` block stripped — when nothing structured matches,
    so ``extract_number_from_string`` still has a chance to find the number.
    """
    if not generation:
        return ""

    try:
        from controller_component.prompts.answer_prompts import extract_answer_candidates

        candidates, _matched = extract_answer_candidates(generation)
        if candidates:
            return candidates[0].candidate
    except Exception as e:  # pragma: no cover - extraction is best-effort
        logger.debug(f"Answer extraction failed, using raw generation: {e}")

    # Fallback: strip the References section appended to the generation.
    marker = "\n\n## References\n"
    idx = generation.find(marker)
    if idx != -1:
        generation = generation[:idx]
    return generation.strip()


# ---------------------------------------------------------------------------
# TRQAGenerationEvaluator
# ---------------------------------------------------------------------------

class TRQAGenerationEvaluator:
    """Numeric exact / soft-exact match evaluator for TRQA.

    Usage::

        evaluator = TRQAGenerationEvaluator(answers={"q1": "8269", ...})
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)

    where ``results`` is the unified agent result dict
    ``{query_id: {"generation": str, ...}, ...}``.
    """

    def __init__(
        self,
        answers: Dict[str, str],
        questions: Optional[Dict[str, str]] = None,
        tolerance_pcts=DEFAULT_TOLERANCE_PCTS,
        decimals: int = 3,
    ) -> None:
        """Initialise the evaluator.

        Args:
            answers:        Mapping of ``query_id -> ground-truth answer``.
            questions:      Mapping of ``query_id -> question text`` (unused by
                            the numeric matcher; accepted for interface parity).
            tolerance_pcts: Tolerance percentages for soft match.
            decimals:       Decimals for numeric rounding before comparison.
        """
        self.answers = answers
        self.questions = questions or {}
        self.tolerance_pcts = tuple(tolerance_pcts)
        self.decimals = decimals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute exact-match and soft-exact-match metrics over all answered queries.

        Returns a dict with keys:
            ``exact_match`` (fraction), ``soft_exact_match`` (dict pct->fraction),
            ``accuracy`` (alias of ``exact_match``), ``num_correct``,
            ``num_evaluated``, and ``per_query``.
        """
        evaluable_ids = [
            qid for qid in results
            if qid in self.answers and self.answers[qid]
        ]
        if not evaluable_ids:
            logger.warning("No queries with ground-truth answers to evaluate")
            return {}

        per_query: List[Dict[str, Any]] = []
        exact_count = 0
        soft_counts = {pct: 0 for pct in self.tolerance_pcts}

        for qid in evaluable_ids:
            gold = self.answers[qid]
            generation = results[qid].get("generation", "")
            prediction = _extract_prediction(generation)

            exact = soft_exact_match(
                prediction, gold, decimals=self.decimals, tolerance_pct=None,
            )["exact_match"]
            if exact:
                exact_count += 1

            soft_flags: Dict[str, bool] = {}
            for pct in self.tolerance_pcts:
                soft = soft_exact_match(
                    prediction, gold, decimals=self.decimals, tolerance_pct=pct,
                )["soft_match"]
                soft_flags[str(pct)] = soft
                if soft:
                    soft_counts[pct] += 1

            per_query.append({
                "query_id": qid,
                "prediction": prediction,
                "gold": gold,
                "exact_match": exact,
                "soft_match": soft_flags,
            })

        num_evaluated = len(per_query)
        return {
            "exact_match": round(exact_count / num_evaluated, 5),
            "accuracy": round(exact_count / num_evaluated, 5),  # alias for runner/summary
            "soft_exact_match": {
                str(pct): round(soft_counts[pct] / num_evaluated, 5)
                for pct in self.tolerance_pcts
            },
            "num_correct": exact_count,
            "num_evaluated": num_evaluated,
            "per_query": per_query,
        }

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "TRQA GENERATION EVALUATION (numeric exact / soft match)",
    ) -> None:
        """Pretty-print the numeric-match metrics."""
        if not metrics:
            print("  No TRQA metrics available (no ground-truth answers)")
            return
        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Queries evaluated:  {metrics.get('num_evaluated', 0)}")
        print(f"  Exact Match:        {metrics.get('exact_match', 0):.4f}")
        soft = metrics.get("soft_exact_match", {})
        if soft:
            print("  Soft Exact Match (within tolerance):")
            for pct in sorted(float(p) for p in soft):
                print(f"    {pct:5.1f}% tolerance: {soft[str(pct)]:.4f}")
        print("=" * 80)

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """No-op; per-query results are saved in bulk via :meth:`save_results`."""
        pass

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Save TRQA metrics as a JSONL file (one line per sample).

        Mirrors the trajectory saving convention: line 1 is a
        ``{"record": "meta", ...}`` header carrying the run-level aggregates
        (exact / soft-exact match, correct/evaluated counts); each subsequent
        line is one query's numeric-match result.
        """
        if not metrics:
            return

        meta = {
            "record": "meta",
            "exact_match": metrics.get("exact_match"),
            "accuracy": metrics.get("accuracy"),
            "soft_exact_match": metrics.get("soft_exact_match"),
            "num_correct": metrics.get("num_correct"),
            "num_evaluated": metrics.get("num_evaluated"),
        }

        def _dump(obj: Dict[str, Any]) -> str:
            return json.dumps(obj, separators=(",", ":"), default=str)

        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(_dump(meta) + "\n")
            for r in metrics.get("per_query", []):
                f.write(_dump(r) + "\n")
        print(f"  Saved TRQA generation metrics: {output_path_str}")
