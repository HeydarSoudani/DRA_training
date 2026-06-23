"""SearchR1 model with thinking and searching."""

from pathlib import Path
from typing import Any


from utils.llm_client import LiteLLMClient
from deep_research_agents.prompts.searchr1.user_prompt import PROMPT_SEARCHR1

from .base_agent import TagReasoningAgent


class SearchR1_Agent(TagReasoningAgent):
    """SearchR1 model with thinking and searching."""

    AGENT_NAME = "SearchR1"

    def __init__(self, llm_client: LiteLLMClient, retriever: Any, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k, verbose)

    def _format_initial_prompt(self, question: str) -> str:
        return PROMPT_SEARCHR1.format(question=question)
