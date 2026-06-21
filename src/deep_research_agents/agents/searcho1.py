"""SearchO1 model with web search integration."""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LiteLLMClient
from prompts.searcho1.prompts import (
    get_multiqa_search_o1_instruction,
    get_task_instruction_openqa,
    get_webpage_to_reasonchain_instruction,
)

from .base_agent import BasicAgent, passages2string
from controller_component import TrackerCriticalThinkResult, TrackerEarlyStopResult
from controller_component.prompts.answer_prompts import FINAL_ANSWER_INSTRUCTION, BOXED_FORMAT

logger = logging.getLogger(__name__)


class SearchO1_Agent(BasicAgent):
    """SearchO1 model with web search integration."""

    AGENT_NAME = "SearchO1"

    def __init__(self, llm_client: LiteLLMClient, retriever: Any, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self.BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
        self.END_SEARCH_QUERY = "<|end_search_query|>"
        self.BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
        self.END_SEARCH_RESULT = "<|end_search_result|>"
        self.MAX_SEARCH_LIMIT = max_iteration
        self.with_reason_in_documents = True
        self.instruction = get_multiqa_search_o1_instruction(self.MAX_SEARCH_LIMIT)
        self.current_step_template = '\n{think}\n<|begin_search_query|>{search_query}<|end_search_query|>\n<|begin_search_result|>{search_result}<|end_search_result|>\n'

    def _rebuild_step_text(self, step):
        think = step.get("think", "")
        sq = step.get("search_query", "")
        docs = step.get("docs", [])
        is_stripped = step.get("_docs_stripped", False)

        if is_stripped:
            search_result = "(earlier search results omitted for brevity)"
        else:
            search_result = step.get("reason_in_docs") or passages2string(docs[:self.seen_top_k])

        return (
            f"\n{think}\n"
            f"<|begin_search_query|>{sq}<|end_search_query|>\n"
            f"<|begin_search_result|>{search_result}<|end_search_result|>\n"
        )

    def get_reasoning_think(self, text: str) -> Optional[str]:
        """Extract reasoning before search query."""
        pattern = re.compile(rf"(.*?){re.escape(self.BEGIN_SEARCH_QUERY)}", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0].strip() if matches else None

    def get_search_query(self, text: str) -> Optional[str]:
        """Extract search query."""
        pattern = re.compile(rf"{re.escape(self.BEGIN_SEARCH_QUERY)}(.*?){re.escape(self.END_SEARCH_QUERY)}", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0].strip() if matches else None

    def get_search_results(self, text: str) -> Optional[str]:
        """Extract search results from reason-in-docs output."""
        match = re.search(r"\*\*Final Information\*\*\s*(.*)", text, re.DOTALL)
        return match.group(1).replace("\n", " ").strip() if match else None

    def get_last_think(self, text: str) -> Optional[str]:
        """Extract last thinking before boxed answer."""
        match = re.search(r"^(.*?)\\boxed\{.*?\}", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def get_boxed_answer(self, text: str) -> Optional[str]:
        """Extract answer from \boxed{} format."""
        match = re.search(r"\\boxed\{(.*?)\}", text)
        return match.group(1).strip() if match else None

    def reason_in_documents(self, path: List[Dict], search_query: str, docs_text: str) -> str:
        """Extract relevant information from documents."""
        prev_reasoning = ' '.join([step.get('think', '') for step in path if 'think' in step])
        rid_input_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, docs_text)
        rid_messages = [{"role": "user", "content": rid_input_prompt}]
        rid_output_text = self.generator.complete(rid_messages, max_tokens=1000)
        return self.get_search_results(rid_output_text) or docs_text

    def inference(self, question: str, generation_temp: float = 0.7) -> tuple:
        """Generate answer with SearchO1 approach.

        Returns:
            (reasoning_path, prediction): List of reasoning steps and final prediction
        """
        input_prompt = self.instruction + get_task_instruction_openqa(question)
        messages = [{"role": "user", "content": input_prompt}]

        reasoning_path = []
        early_stop_result = None
        for iter_idx in range(1, self.MAX_SEARCH_LIMIT + 1):
            iter_num = iter_idx
            self._notify_progress("think", iter_num)
            try:
                output_text = self.generator.complete(messages, temperature=generation_temp, max_tokens=2000)
            except Exception as e:
                logger.warning(f"Iteration {iter_num} API error: {e}")
                self._force_answer_on_context_limit(
                    question, reasoning_path,
                    f"{BOXED_FORMAT}\n\\boxed{{",
                    lambda out: (out or "").split("}")[0].strip(),
                    generation_temp=generation_temp,
                    initial_prompt=self.instruction + get_task_instruction_openqa(question),
                    answer_nudge=f"\n{FINAL_ANSWER_INSTRUCTION}\n\\boxed{{",
                )
                break

            # Check if answer is provided (contains \boxed{})
            if '\\boxed{' in output_text:
                last_think = self.get_last_think(output_text) or output_text
                pred_answer = self.get_boxed_answer(output_text)
                self._vprint(iter_num, "think", last_think or "(no think)")
                self._notify_progress("answer", iter_num)
                self._vprint(iter_num, "answer", pred_answer or "(no answer)")
                reasoning_path.append({'think': last_think, 'prediction': pred_answer})
                break

            # Extract search query
            tmp_think = self.get_reasoning_think(output_text) or ''
            tmp_think = tmp_think.replace("\n", ' ').replace("\n\n", ' ')
            tmp_query = self.get_search_query(output_text)
            self._vprint(iter_num, "think", tmp_think or "(no think)")

            if tmp_query:
                self._notify_progress("search", iter_num)
                self._vprint(iter_num, "search", tmp_query)
                if self.search_tool is not None:
                    search_docs = self.search_tool.execute(
                        tmp_query,
                        original_query=question,
                        reasoning=tmp_think if tmp_think else None,
                    )
                else:
                    search_docs = self.retrieve_documents(tmp_query, original_query=question)
                # seen_top_k docs are passed to the LLM; all retrieved docs go to TREC
                docs_text = passages2string(search_docs[:self.seen_top_k])
            else:
                search_docs, docs_text = [], ''

            # Trajectory tracker: may inject observation, critical_think, or early stop
            seen_docs = search_docs[:self.seen_top_k]
            tracker_result = self.post_search_evaluate(
                subquery=tmp_query or "", docs=seen_docs,
                iter_num=iter_num, original_query=question,
                thinking=tmp_think,
                seen_docs=seen_docs if seen_docs else None,
                trajectory=input_prompt,
            )
            if isinstance(tracker_result, TrackerEarlyStopResult):
                early_stop_result = tracker_result

            reasoning_path.append({
                'think': tmp_think,
                'search_query': tmp_query,
                'docs': search_docs,  # all top_k docs — used for TREC file
                'component_doc_ids': [d.get('doc_id', '') for d in search_docs[:self.seen_top_k]],
            })

            # Reason in documents
            if self.with_reason_in_documents and tmp_query:
                rid_output_text = self.reason_in_documents(reasoning_path, tmp_query, docs_text)
                reasoning_path[-1]['reason_in_docs'] = rid_output_text
                self._vprint(iter_num, "reason-in-docs", rid_output_text or "(empty)")
                search_result_txt = rid_output_text
            else:
                search_result_txt = docs_text

            # Update prompt
            current_step_text = self.current_step_template.format(
                think=tmp_think,
                search_query=tmp_query,
                search_result=search_result_txt
            )
            input_prompt += current_step_text

            # Early stopping: break loop, force answer after
            if early_stop_result is not None:
                break

            # Inject tracker critical_think as an additional full turn
            if isinstance(tracker_result, TrackerCriticalThinkResult):
                reasoning_path.append(self._critical_think_to_reasoning_entry(tracker_result))
                critical_think_step = self.current_step_template.format(
                    think=tracker_result.critical_think,
                    search_query=tracker_result.critical_search_query,
                    search_result=tracker_result.critical_observation,
                )
                input_prompt += critical_think_step
            messages = [{"role": "user", "content": input_prompt}]

        pred_answer = reasoning_path[-1].get('prediction') if reasoning_path else None

        if not pred_answer:
            action_type = 'early_stop' if early_stop_result is not None else 'max_iter_force'
            force_input = input_prompt + f"\n{FINAL_ANSWER_INSTRUCTION}\n\\boxed{{"
            force_messages = [{"role": "user", "content": force_input}]
            force_output = self.generator.complete(force_messages, temperature=generation_temp)
            pred_answer = (force_output or "").split("}")[0].strip()
            self._vprint(iter_num, "think", FINAL_ANSWER_INSTRUCTION)
            self._vprint(iter_num, "finish", pred_answer or "(no answer)")
            reasoning_path.append({
                'action_type': action_type,
                'think': FINAL_ANSWER_INSTRUCTION,
                'prediction': pred_answer,
            })

        return reasoning_path, pred_answer, iter_num
