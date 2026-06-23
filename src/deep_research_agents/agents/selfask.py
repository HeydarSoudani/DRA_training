"""Self-Ask model with follow-up questions."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


from utils.llm_client import LiteLLMClient

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "selfask"
SELF_ASK_PROMPT_MULTI_HOP = (_PROMPT_DIR / "system.txt").read_text()

from deep_research_agents.prompts.selfask.user_prompt import USER_PROMPT as _SELFASK_USER_PROMPT

from .base_agent import BasicAgent
from controller_component import TrackerCriticalThinkResult, TrackerEarlyStopResult
from controller_component.prompts.answer_prompts import FINAL_ANSWER_INSTRUCTION, SELFASK_FORMAT

logger = logging.getLogger(__name__)


class SelfAsk_Agent(BasicAgent):
    """Self-Ask model with follow-up questions."""

    AGENT_NAME = "SelfAsk"

    def __init__(self, llm_client: LiteLLMClient, retriever: Any, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self.system_prompt = SELF_ASK_PROMPT_MULTI_HOP
        self.user_prompt = _SELFASK_USER_PROMPT

    def documents2string(self, retrieval_result: List[Dict[str, Any]]) -> str:
        """Convert retrieval results to context string."""
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            text = doc_item.get("relevant_text", "")
            format_reference += f"Context{idx+1}: {text}\n"
        return format_reference

    def extract_follow_up(self, text: str) -> str:
        """Extract follow-up question."""
        match = re.search(r'Follow up:\s*(.*?)\nIntermediate answer:', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_intermediate(self, text: str) -> str:
        """Extract intermediate answer."""
        match = re.search(r'(.*?)(?:Follow up:|So the final answer is:)', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer."""
        parts = text.split("So the final answer is: ", 1)
        if len(parts) <= 1:
            return None
        pred = parts[1].strip()
        pred = re.sub(r"\.?</s>", "", pred)
        pred = pred.rstrip(".?!")
        return pred

    def inference(self, question: str, generation_temp: float = 0.7) -> tuple:
        """Generate answer with Self-Ask approach.

        Returns:
            (reasoning_path, prediction): List of reasoning steps and final prediction
        """
        reasoning_path, text = [], ""

        # Initial retrieval (Iter 0: retrieval only, no think)
        search_query = question
        self._notify_progress("search", 0)
        self._vprint(0, "search", search_query)
        if self.search_tool is not None:
            cur_search_docs = self.search_tool.execute(
                search_query,
                original_query=question,
            )
        else:
            cur_search_docs = self.retrieve_documents(search_query, original_query=question)
        # Trajectory tracker: may inject observation or critical_think for initial search
        seen_docs_init = cur_search_docs[:self.seen_top_k]
        tracker_result_init = self.post_search_evaluate(
            subquery=search_query, docs=seen_docs_init,
            iter_num=0, original_query=question,
            seen_docs=seen_docs_init,
        )

        # seen_top_k docs are passed to the LLM; all retrieved docs go to TREC
        docs_text = self.documents2string(cur_search_docs[:self.seen_top_k])
        if isinstance(tracker_result_init, TrackerCriticalThinkResult):
            cur_search_docs = tracker_result_init.critical_docs
            docs_text = tracker_result_init.critical_observation
        user_input_prompt = self.user_prompt.format(
            documents=docs_text,
            question=question
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input_prompt}
        ]
        reasoning_path.append({
            'think': '',
            'search_query': search_query,
            'docs': cur_search_docs,  # all top_k docs — used for TREC file
            'component_doc_ids': [d.get('doc_id', '') for d in cur_search_docs[:self.seen_top_k]],
            'tokens': self._step_tokens(),
        })

        output_text = ""
        for idx in range(self.max_iteration):
            iter_num = idx + 1  # +1 because initial retrieval is iter 0
            self._notify_progress("think", iter_num)
            try:
                output_text = self.generator.complete(messages, temperature=generation_temp, max_tokens=500)
            except Exception as e:
                logger.warning(f"Iteration {iter_num} API error: {e}")
                answer = self._force_answer_on_context_limit(
                    question, reasoning_path,
                    SELFASK_FORMAT,
                    lambda out: self.extract_final_answer(out) or (out or "").strip(),
                    generation_temp=generation_temp,
                    max_tokens=100,
                    system_prompt=self.system_prompt,
                )
                if answer is not None:
                    return reasoning_path, answer, iter_num
                break

            if "So the final answer is:" in output_text:
                self._notify_progress("answer", iter_num)
                pred = self.extract_final_answer(output_text)
                self._vprint(iter_num, "answer", pred or output_text[:200])
                text += output_text
                break

            intermediate_ans = self.extract_intermediate(output_text)
            search_query = self.extract_follow_up(output_text)
            self._vprint(iter_num, "think", intermediate_ans or "(no intermediate)")
            if search_query:
                self._notify_progress("search", iter_num)
                self._vprint(iter_num, "follow-up", search_query)
            if search_query:
                if self.search_tool is not None:
                    cur_search_docs = self.search_tool.execute(
                        search_query,
                        original_query=question,
                        reasoning=intermediate_ans if intermediate_ans else None,
                    )
                else:
                    cur_search_docs = self.retrieve_documents(search_query, original_query=question)
            else:
                cur_search_docs = []
            # Trajectory tracker: may inject observation, critical_think, or early stop
            seen_docs = cur_search_docs[:self.seen_top_k]
            tracker_result = self.post_search_evaluate(
                subquery=search_query or question, docs=seen_docs,
                iter_num=iter_num, original_query=question,
                thinking=intermediate_ans or "",
                seen_docs=seen_docs if seen_docs else None,
                trajectory=messages,
            )
            _early_stop_triggered = isinstance(tracker_result, TrackerEarlyStopResult)

            # Handle critical_think: extend docs pool with critical docs
            if not _early_stop_triggered and isinstance(tracker_result, TrackerCriticalThinkResult):
                cur_search_docs = cur_search_docs + tracker_result.critical_docs

            # Aggregate component-visible docs from all steps (seen_top_k per step)
            tmp_docs = [
                doc for step in reasoning_path
                for doc in step['docs'][:self.seen_top_k]
            ] + cur_search_docs[:self.seen_top_k]
            unq_tmp_doc = self.get_unique_docs(tmp_docs)

            reasoning_path.append({
                'think': intermediate_ans,
                'search_query': search_query,
                'docs': cur_search_docs,  # all top_k docs — used for TREC file
                'component_doc_ids': [d.get('doc_id', '') for d in cur_search_docs[:self.seen_top_k]],
                'tokens': self._step_tokens(),
            })

            # Early stopping: break loop and force final answer
            if _early_stop_triggered:
                break

            if idx == 0:
                text += f"Follow up: {search_query}\nIntermediate answer: "
            else:
                text += f"{intermediate_ans}\nFollow up: {search_query}\nIntermediate answer: "

            docs_str = self.documents2string(unq_tmp_doc)
            if isinstance(tracker_result, TrackerCriticalThinkResult):
                pass  # keep aggregated docs_str with critical_think docs included
            user_input_prompt = self.user_prompt.format(
                documents=docs_str,
                question=question
            ) + text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]

        # If no final answer yet, generate one
        if "So the final answer is:" not in output_text:
            text += f"{output_text}.\nSo the final answer is: "

            tmp_docs = [doc for step in reasoning_path for doc in step['docs'][:self.seen_top_k]]
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            user_input_prompt = self.user_prompt.format(
                documents=self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
            output_text = self.generator.complete(messages, temperature=generation_temp, max_tokens=100)
            pred_answer = self.extract_final_answer(output_text) or output_text
            self._vprint(iter_num, "think", output_text or "(no thought)")
            self._vprint(iter_num, "finish", pred_answer or "(no answer)")
            reasoning_path.append({'think': output_text, 'prediction': pred_answer})
        else:
            intermediate_ans = self.extract_intermediate(output_text)
            pred_answer = self.extract_final_answer(output_text)
            reasoning_path.append({'think': intermediate_ans, 'prediction': pred_answer})

        num_iterations = iter_num + 1 if reasoning_path else 0
        return reasoning_path, pred_answer, num_iterations
