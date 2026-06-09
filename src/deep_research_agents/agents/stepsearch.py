"""StepSearch model with planned searching."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
from prompts.stepsearch.user_prompt import PROMPT_STEPSEARCH

from .base_agent import TagReasoningAgent


class StepSearch_Agent(TagReasoningAgent):
    """StepSearch model with planned searching."""

    AGENT_NAME = "StepSearch"

    def __init__(self, llm_client: LiteLLMClient, retriever: Any, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k, verbose)

    def _format_initial_prompt(self, question: str) -> str:
        return PROMPT_STEPSEARCH.format(question=question)
