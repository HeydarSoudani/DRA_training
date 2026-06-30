"""Qwen3-Thinking agent.

Exposes the Qwen3-*-Thinking-2507 reasoning models as deep-research agents.
These are reasoning models that emit ``reasoning_content`` and call tools via
vLLM's ``--tool-call-parser hermes`` (the same parser Tongyi uses), so they fit
the GLM agent's OpenAI Chat Completions + tool-calling inference loop exactly.

This class therefore subclasses :class:`GLM_Agent` and only swaps the model
identity (default model name, API base/key env vars).  The generic deep-research
system prompt and query template under ``prompts/glm`` are reused as-is — they
contain no GLM-specific text.

Two CLI names map onto this single agent (see ``AGENTIC_MODEL_TO_LLM`` /
``AGENTIC_MODEL_ALIAS`` in ``utils.config``):

    qwen3_4b_thinking  → Qwen/Qwen3-4B-Thinking-2507
    qwen3_30b_thinking → Qwen/Qwen3-30B-A3B-Thinking-2507

The concrete model is selected by ``build_agent`` via the ``model_name`` kwarg.
"""

import os
from typing import Optional

from .glm_agent import GLM_Agent


class Qwen3_Agent(GLM_Agent):
    """Qwen3-Thinking deep research agent (Chat Completions + hermes tools)."""

    AGENT_NAME = "Qwen3"

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        max_iteration: int = 100,
        seen_top_k: int = 5,
        model_url: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-30B-A3B-Thinking-2507",
        max_output_tokens: int = 20000,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Resolve the vLLM endpoint from a Qwen3-specific env var, falling back
        # to the shared agent port (6008) that the vLLM manager serves on.
        model_url = model_url or os.getenv(
            "QWEN3_API_BASE", "http://localhost:6008/v1"
        )
        api_key = api_key or os.getenv("QWEN3_API_KEY", "EMPTY")
        super().__init__(
            llm_client=llm_client,
            retriever=retriever,
            max_iteration=max_iteration,
            seen_top_k=seen_top_k,
            model_url=model_url,
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            system_prompt=system_prompt,
            verbose=verbose,
            api_key=api_key,
            **kwargs,
        )

        # Thinking models spend a large share of every turn on the reasoning
        # block, so the GLM defaults (4096 per step / 1024 per answer candidate)
        # truncate mid-thought and starve the actual action/answer.  Give the
        # reasoning room to complete before a tool call or final answer is due.
        self._max_tokens_per_call = 12288
        self._answer_candidate_max_tokens = 4096
