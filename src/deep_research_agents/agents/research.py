"""ReSearch model with iterative search and reasoning."""

import re
from pathlib import Path
from typing import Any, Optional


from utils.llm_client import LiteLLMClient
_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "research"
SYSTEM_PROMPT_RESEARCH_INST = (_PROMPT_DIR / "system.txt").read_text()

from .base_agent import TagReasoningAgent, passages2string


class ReSearch_Agent(TagReasoningAgent):
    """ReSearch model with iterative search and reasoning."""

    AGENT_NAME = "ReSearch"

    def __init__(self, llm_client: LiteLLMClient, retriever: Any, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k, verbose)
        self._system_prompt = SYSTEM_PROMPT_RESEARCH_INST
        self.curr_step_template = '\n{output_text}<result>{search_results}</result>\n'

    def _format_initial_prompt(self, question: str) -> str:
        return question

    def get_query(self, text: str) -> Optional[str]:
        pattern = re.compile(r"<search>\s*search query:\s*(.*?)\s*</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0].strip() if matches else None

    def _has_answer(self, output_text: str) -> bool:
        return '</answer>' in output_text or '\\boxed{' in output_text

    def _extract_prediction(self, output_text: str) -> Optional[str]:
        boxed = re.search(r"\\boxed\{(.*?)\}", output_text)
        if boxed:
            return boxed.group(1).strip()
        return self.get_answer(output_text)

    def _rebuild_step_text(self, step):
        think = step.get("think", "")
        sq = step.get("search_query", "")
        docs = step.get("docs", [])
        is_stripped = step.get("_docs_stripped", False)

        output_text = ""
        if think:
            output_text += f"<think>{think}</think>\n"
        if sq:
            output_text += f"<search>search query: {sq}</search>"

        if is_stripped:
            search_results = "(earlier search results omitted for brevity)"
        else:
            search_results = passages2string(docs[:self.seen_top_k])

        return f"\n{output_text}<result>{search_results}</result>\n"
