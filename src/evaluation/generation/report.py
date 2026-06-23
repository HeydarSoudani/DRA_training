"""Report evaluator (long-form generation) using LLM-as-judge + citation faithfulness.

For report-style tasks (e.g. cpm_report) there is no single short answer to grade.
Instead this evaluator scores two complementary aspects:

1. **Rubric quality** — an LLM judge rates the report against the question on
   three 1-5 dimensions: *coverage*, *relevance*, and *organization*.
2. **Citation faithfulness** — using the qrels, how many cited documents are
   actually relevant (precision) and how many relevant documents were cited
   (recall).  This reuses :func:`evaluation.retrieval.citation_metrics.compute_citation_metrics`.

The judge plumbing (LiteLLM client, multi-endpoint round-robin, concurrency)
mirrors :class:`evaluation.generation.short_answer.AccuracyEvaluator` so the same
vLLM judge server wiring in the pipeline applies unchanged.
"""

import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from utils.llm_client import LiteLLMClient
from ..retrieval.citation_metrics import compute_citation_metrics, resolve_cited_doc_ids

logger = logging.getLogger(__name__)


RUBRIC_DIMENSIONS = ("coverage", "relevance", "organization")

RUBRIC_TEMPLATE = """\
You are grading a research report written in response to a question. Rate the \
[report] against the [question] on each dimension below using an integer score \
from 1 (very poor) to 5 (excellent).

[question]: {question}

[report]: {report}

Dimensions:
- coverage: Does the report comprehensively address the information need, covering \
the important aspects of the question?
- relevance: Is the content on-topic and free of irrelevant or off-question material?
- organization: Is the report well-structured, coherent, and easy to follow?

Respond in exactly this format and nothing else:

coverage: <1-5>
relevance: <1-5>
organization: <1-5>
justification: <one or two sentences> /no_think"""


def _parse_rubric_response(text: str) -> Dict[str, Any]:
    """Parse ``dimension: <int>`` scores and the justification from judge output."""
    result: Dict[str, Any] = {dim: None for dim in RUBRIC_DIMENSIONS}
    result["justification"] = None
    result["raw_response"] = text
    result["parse_error"] = False

    for dim in RUBRIC_DIMENSIONS:
        m = re.search(rf"(?:^|\n)\s*\*{{0,2}}{dim}\*{{0,2}}\s*:\s*\*{{0,2}}\s*([1-5])",
                      text, re.IGNORECASE)
        if m:
            result[dim] = int(m.group(1))

    jm = re.search(r"(?:^|\n)\s*\*{0,2}justification\*{0,2}\s*:\s*(.+)",
                   text, re.IGNORECASE | re.DOTALL)
    if jm:
        result["justification"] = jm.group(1).strip()

    if all(result[dim] is None for dim in RUBRIC_DIMENSIONS):
        result["parse_error"] = True

    return result


class ReportEvaluator:
    """Evaluate long-form reports with an LLM-judge rubric + citation faithfulness.

    Usage::

        evaluator = ReportEvaluator(
            questions={"q1": "..."}, qrels=qrels,
            judge_api_base="http://localhost:6009/v1",
        )
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    def __init__(
        self,
        questions: Optional[Dict[str, str]] = None,
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
        judge_model: str = "openai/Qwen/Qwen3-32B",
        judge_api_base: str = "http://localhost:6009/v1",
        judge_api_bases: Optional[List[str]] = None,
        max_concurrent_judges: int = 64,
    ) -> None:
        """Initialise the evaluator.

        Args:
            questions:      Mapping of ``query_id -> question text``.
            qrels:          Ground-truth relevance judgements, used for the
                            citation-faithfulness metrics.  When None, only the
                            rubric scores are produced.
            judge_model:    LiteLLM model identifier for the judge.
            judge_api_base: OpenAI-compatible API base for a single judge server.
            judge_api_bases: Optional list of judge servers (round-robin).
            max_concurrent_judges: Max in-flight judge requests per server.
        """
        self.questions = questions or {}
        self.qrels = qrels or {}
        self.judge_model = judge_model
        self.judge_api_bases: List[str] = judge_api_bases if judge_api_bases else [judge_api_base]
        self.max_concurrent_judges = max_concurrent_judges
        self._clients: List[LiteLLMClient] = []

    def _get_clients(self) -> List[LiteLLMClient]:
        """Lazily initialise one LLM client per judge endpoint."""
        if not self._clients:
            for base in self.judge_api_bases:
                self._clients.append(LiteLLMClient(
                    model=self.judge_model,
                    api_base=base,
                    api_key="EMPTY",
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    max_tokens=4096,
                ))
        return self._clients

    @staticmethod
    def _clean_report(text: str) -> str:
        """Strip the appended ``## References`` block before judging."""
        marker = "\n\n## References\n"
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
        return text.strip()

    def _judge_single(self, query_id: str, report: str, client: LiteLLMClient) -> Dict[str, Any]:
        """Run the rubric judge on a single query/report pair."""
        question = self.questions.get(query_id, query_id)
        prompt = RUBRIC_TEMPLATE.format(question=question, report=self._clean_report(report))
        try:
            raw = client.complete([{"role": "user", "content": prompt}])
            parsed = _parse_rubric_response(raw)
        except Exception as e:
            logger.error(f"Report judge failed for {query_id}: {e}")
            parsed = {dim: None for dim in RUBRIC_DIMENSIONS}
            parsed.update({"justification": f"Judge error: {e}", "parse_error": True, "raw_response": ""})
        parsed["query_id"] = query_id
        return parsed

    def _citation_faithfulness(self, query_id: str, result: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Citation precision/recall/F1 of cited docs against the qrels."""
        if query_id not in self.qrels:
            return None
        cited_ids = set(resolve_cited_doc_ids(result))
        relevant_ids = {d for d, rel in self.qrels[query_id].items() if rel > 0}
        m = compute_citation_metrics(cited_ids, retrieved_doc_ids=set(), relevant_doc_ids=relevant_ids)
        return {
            "citation_precision": m["citation_precision"],
            "citation_recall": m["citation_recall"],
            "citation_f1": m["citation_f1"],
            "num_cited": m["num_cited"],
            "num_relevant": m["num_relevant"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all reports; return aggregate rubric + citation-faithfulness metrics."""
        to_judge = [
            (qid, results[qid].get("generation", ""))
            for qid in results
            if results[qid].get("generation")
        ]
        if not to_judge:
            logger.warning("No non-empty reports to evaluate")
            return {}

        per_query: List[Dict[str, Any]] = []
        clients = self._get_clients()
        num_clients = len(clients)
        bar = tqdm(total=len(to_judge), desc=f"[ReportJudge x{num_clients}]",
                   bar_format="{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
                   dynamic_ncols=True)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent_judges * num_clients
        ) as executor:
            future_to_qid = {
                executor.submit(self._judge_single, qid, gen, clients[i % num_clients]): qid
                for i, (qid, gen) in enumerate(to_judge)
            }
            for future in concurrent.futures.as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    judged = future.result()
                except Exception as e:
                    logger.error(f"Report judge failed for {qid}: {e}")
                    judged = {dim: None for dim in RUBRIC_DIMENSIONS}
                    judged.update({"query_id": qid, "justification": f"Judge error: {e}",
                                   "parse_error": True})
                faithfulness = self._citation_faithfulness(qid, results[qid])
                if faithfulness:
                    judged.update(faithfulness)
                per_query.append(judged)
                bar.update(1)
        bar.close()

        def _avg(key: str) -> Optional[float]:
            vals = [r[key] for r in per_query if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        metrics: Dict[str, Any] = {
            "num_evaluated": len(per_query),
            "rubric": {dim: _avg(dim) for dim in RUBRIC_DIMENSIONS},
        }
        dim_means = [metrics["rubric"][d] for d in RUBRIC_DIMENSIONS if metrics["rubric"][d] is not None]
        metrics["rubric"]["overall"] = round(sum(dim_means) / len(dim_means), 4) if dim_means else None

        if any("citation_f1" in r for r in per_query):
            metrics["citation_faithfulness"] = {
                "citation_precision": _avg("citation_precision"),
                "citation_recall": _avg("citation_recall"),
                "citation_f1": _avg("citation_f1"),
                "avg_num_cited": _avg("num_cited"),
            }
        metrics["per_query"] = per_query
        return metrics

    def print_results(self, metrics: Dict[str, Any], header: str = "REPORT EVALUATION RESULTS (LLM-as-Judge)") -> None:
        """Pretty-print report-evaluation metrics."""
        if not metrics:
            print("  No report metrics available")
            return
        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Judge model:        {self.judge_model}")
        print(f"  Reports evaluated:  {metrics.get('num_evaluated', 0)}")
        rubric = metrics.get("rubric", {})
        for dim in RUBRIC_DIMENSIONS:
            val = rubric.get(dim)
            print(f"  {dim.capitalize():<13s} (1-5): {val if val is not None else 'n/a'}")
        overall = rubric.get("overall")
        print(f"  Overall      (1-5): {overall if overall is not None else 'n/a'}")
        cf = metrics.get("citation_faithfulness")
        if cf:
            print(f"  Citation Precision: {cf.get('citation_precision')}")
            print(f"  Citation Recall:    {cf.get('citation_recall')}")
            print(f"  Citation F1:        {cf.get('citation_f1')}")
        print("=" * 80)

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir) -> None:
        """No-op; per-report results are saved in bulk via :meth:`save_results`."""
        pass

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Save report metrics to JSON (drops raw judge responses to stay small)."""
        if not metrics:
            return
        save_data = {
            "num_evaluated": metrics.get("num_evaluated", 0),
            "rubric": metrics.get("rubric", {}),
            "citation_faithfulness": metrics.get("citation_faithfulness"),
            "judge_model": self.judge_model,
            "per_query": [
                {k: v for k, v in r.items() if k != "raw_response"}
                for r in metrics.get("per_query", [])
            ],
        }
        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"  Saved report metrics: {output_path_str}")
