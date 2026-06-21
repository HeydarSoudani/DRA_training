"""Accuracy evaluator using LLM-as-judge (BrowseComp-Plus methodology).

Evaluates whether the agent's generated answer is correct by comparing it
against the ground-truth answer using an LLM judge.

The prompt and scoring methodology are taken from BrowseComp-Plus
(Chen et al., 2025, arXiv:2508.06600), which itself adopts the grading
prompt from OpenAI's BrowseComp simple-evals.

Judge model: Qwen3-32B served via vLLM — matches the official
BrowseComp-Plus / AgentIR leaderboard.

Scoring: binary (correct / incorrect).  Accuracy = #correct / #total.
"""

import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from utils.llm_client import LiteLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grader prompt — verbatim from BrowseComp-Plus Appendix F / search_agent/prompts.py
# ---------------------------------------------------------------------------

GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise \
and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on \
[correct_answer], in the context of this [question]. You should judge whether the \
extracted_final_answer is semantically equivalent to [correct_answer], allowing the \
extracted_final_answer to be string variations of [correct_answer]. You should also allow \
the extracted_final_answer to be more precise or verbose than [correct_answer], as long as \
its additional details are correct. Do not comment on any background to the problem, do not \
attempt to solve the problem, do not argue for any answer different than [correct_answer], \
focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, \
or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if \
there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is \
incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put \
100 if there is no confidence score available. /no_think"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the structured fields from the judge LLM's response.

    Returns a dict with keys:
        extracted_final_answer, reasoning, correct (bool), confidence (int).
    """
    result: Dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
        "raw_response": text,
    }

    # Field prefix pattern: handles "field:", "**field:**", and "**field**:"
    def _field_re(name: str) -> str:
        return rf"\*{{0,2}}{name}\*{{0,2}}:\*{{0,2}}"

    # Extract "correct:" field (yes/no)
    correct_match = re.search(rf"(?:^|\n)\s*{_field_re('correct')}\s*(yes|no)", text, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).strip().lower() == "yes"

    # Extract "extracted_final_answer:" field
    answer_match = re.search(
        rf"(?:^|\n)\s*{_field_re('extracted_final_answer')}\s*(.+?)(?=\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract "reasoning:" field
    reasoning_match = re.search(
        rf"(?:^|\n)\s*{_field_re('reasoning')}\s*(.+?)(?=\n\s*(?:{_field_re('correct')}|{_field_re('confidence')})|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract "confidence:" field
    confidence_match = re.search(rf"(?:^|\n)\s*{_field_re('confidence')}\s*(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    if result["correct"] is None:
        result["parse_error"] = True

    return result


# ---------------------------------------------------------------------------
# AccuracyEvaluator
# ---------------------------------------------------------------------------

class AccuracyEvaluator:
    """Evaluate answer correctness using LLM-as-judge (BrowseComp-Plus method).

    Usage::

        answers = {"q1": "ground truth answer 1", "q2": "ground truth answer 2"}
        evaluator = AccuracyEvaluator(
            answers=answers,
            judge_model="openai/Qwen/Qwen3-32B",
            judge_api_base="http://localhost:6009/v1",
        )
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)

    Where ``results`` is the unified agent result dict:
        {query_id: {"generation": str, ...}, ...}
    """

    def __init__(
        self,
        answers: Dict[str, str],
        questions: Optional[Dict[str, str]] = None,
        judge_model: str = "openai/Qwen/Qwen3-32B",
        judge_api_base: str = "http://localhost:6009/v1",
        judge_api_bases: Optional[List[str]] = None,
        max_concurrent_judges: int = 64,
    ) -> None:
        """Initialise the evaluator.

        Args:
            answers:        Mapping of ``query_id -> ground-truth answer``.
            questions:      Mapping of ``query_id -> question text``.
                            If None, a generic placeholder is used.
            judge_model:    LiteLLM model identifier for the judge
                            (default: Qwen3-32B via local vLLM).
            judge_api_base: OpenAI-compatible API base URL for a single
                            vLLM judge server (used when *judge_api_bases*
                            is not provided).
            judge_api_bases: List of API base URLs for multiple judge
                            servers.  Requests are distributed round-robin.
            max_concurrent_judges: Max in-flight judge requests per server.
        """
        self.answers = answers
        self.questions = questions or {}
        self.judge_model = judge_model
        self.judge_api_bases: List[str] = (
            judge_api_bases if judge_api_bases else [judge_api_base]
        )
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
    def _clean_response_for_judge(text: str) -> str:
        """Clean agent generation text before sending to the LLM judge.

        Strips noise that is irrelevant to correctness evaluation:
        1. The appended ``## References`` section (document metadata/snippets,
           often 5-10x longer than the actual answer).
        2. Lenticular-bracket citation markers (``【12345】``) injected by
           tool-calling agents (OSS, GLM, CPM Explore) that reference internal
           document IDs meaningless to the judge.
        """
        # 1. Strip References section
        marker = "\n\n## References\n"
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()

        # 2. Strip lenticular-bracket citation markers: 【docid】
        text = re.sub(r"【[^】]*】", "", text)

        return text.strip()

    def _judge_single(
        self, query_id: str, response_text: str, client: LiteLLMClient,
    ) -> Dict[str, Any]:
        """Run the judge on a single query-response pair.

        Returns parsed judge result dict.
        """
        question = self.questions.get(query_id, query_id)
        correct_answer = self.answers[query_id]

        response_text = self._clean_response_for_judge(response_text)

        prompt = GRADER_TEMPLATE.format(
            question=question,
            response=response_text,
            correct_answer=correct_answer,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            raw_response = client.complete(messages)
            parsed = _parse_judge_response(raw_response)
            parsed["query_id"] = query_id
            parsed["judge_input"] = response_text
            return parsed
        except Exception as e:
            logger.error(f"Judge failed for {query_id}: {e}")
            return {
                "query_id": query_id,
                "extracted_final_answer": None,
                "judge_input": response_text,
                "reasoning": f"Judge error: {e}",
                "correct": False,
                "confidence": 0,
                "raw_response": "",
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate accuracy across all queries that have ground-truth answers.

        Args:
            results: Unified agent results keyed by *query_id*.
                     Each value must contain a ``"generation"`` key.

        Returns:
            Dict with keys:
                - ``accuracy``: fraction of correct answers
                - ``num_correct``: count of correct answers
                - ``num_evaluated``: count of evaluated queries
                - ``per_query``: list of per-query judge results
        """
        per_query: List[Dict[str, Any]] = []

        evaluable_ids = [
            qid for qid in results
            if qid in self.answers and self.answers[qid]
        ]

        if not evaluable_ids:
            logger.warning("No queries with ground-truth answers to evaluate")
            return {}

        to_judge: List[tuple] = []
        for query_id in evaluable_ids:
            generation = results[query_id].get("generation", "")
            if not generation:
                logger.warning(f"Empty generation for {query_id}, marking incorrect")
                per_query.append({
                    "query_id": query_id,
                    "correct": False,
                    "extracted_final_answer": None,
                    "judge_input": "",
                    "reasoning": "Empty generation",
                    "confidence": 0,
                })
            else:
                to_judge.append((query_id, generation))

        if to_judge:
            clients = self._get_clients()
            num_clients = len(clients)
            total_workers = self.max_concurrent_judges * num_clients
            bar = tqdm(
                total=len(to_judge),
                desc=f"[Judge x{num_clients}]",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
                dynamic_ncols=True,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=total_workers
            ) as executor:
                future_to_qid = {
                    executor.submit(
                        self._judge_single, qid, gen, clients[i % num_clients],
                    ): qid
                    for i, (qid, gen) in enumerate(to_judge)
                }
                for future in concurrent.futures.as_completed(future_to_qid):
                    qid = future_to_qid[future]
                    try:
                        judge_result = future.result()
                    except Exception as e:
                        logger.error(f"Judge failed for {qid}: {e}")
                        judge_result = {
                            "query_id": qid,
                            "extracted_final_answer": None,
                            "judge_input": "",
                            "reasoning": f"Judge error: {e}",
                            "correct": False,
                            "confidence": 0,
                            "raw_response": "",
                        }
                    per_query.append(judge_result)
                    status = "CORRECT" if judge_result["correct"] else "INCORRECT"
                    bar.set_postfix_str(f"{qid}: {status}")
                    bar.update(1)
            bar.close()

        num_correct = sum(1 for r in per_query if r["correct"])
        num_evaluated = len(per_query)
        accuracy = num_correct / num_evaluated if num_evaluated > 0 else 0.0

        return {
            "accuracy": round(accuracy, 5),
            "num_correct": num_correct,
            "num_evaluated": num_evaluated,
            "per_query": per_query,
        }

    def print_results(
        self,
        metrics: Dict[str, Any],
        header: str = "ACCURACY EVALUATION RESULTS (LLM-as-Judge)",
    ) -> None:
        """Pretty-print accuracy metrics.

        Args:
            metrics: Output of :meth:`evaluate`.
            header:  Section header string.
        """
        if not metrics:
            print("  No accuracy metrics available (no ground-truth answers)")
            return

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)
        print(f"  Judge model:        {self.judge_model}")
        print(f"  Queries evaluated:  {metrics.get('num_evaluated', 0)}")
        print(f"  Correct:            {metrics.get('num_correct', 0)}")
        print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print("=" * 80)

    def save_item(self, query_id: str, result: Dict[str, Any], output_dir: Path) -> None:
        """Save per-query accuracy result as a JSON file.

        This is a no-op placeholder to match the evaluator interface used
        in the pipeline loop.  Per-query accuracy is saved in bulk via
        :meth:`save_results`.
        """
        pass

    def save_results(self, metrics: Dict[str, Any], output_path) -> None:
        """Save accuracy metrics to a JSON file.

        Supports both local paths and S3 URIs.

        Args:
            metrics:     Output of :meth:`evaluate`.
            output_path: Destination file path.
        """
        if not metrics:
            return

        # Save summary (without raw_response to keep file small)
        save_data = {
            "accuracy": metrics["accuracy"],
            "num_correct": metrics["num_correct"],
            "num_evaluated": metrics["num_evaluated"],
            "judge_model": self.judge_model,
            "per_query": [
                {
                    "query_id": r["query_id"],
                    "correct": r["correct"],
                    "extracted_final_answer": r.get("extracted_final_answer"),
                    "judge_input": r.get("judge_input"),
                    "reasoning": r.get("reasoning"),
                    "confidence": r.get("confidence"),
                }
                for r in metrics.get("per_query", [])
            ],
        }
        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"  Saved accuracy results: {output_path_str}")
