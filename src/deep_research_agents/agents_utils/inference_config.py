"""Inference configuration: single source of truth for LLM call parameters.

Every agent builds one ``InferenceConfig`` in its ``__init__``.  The main loop,
force-answer, and answer-candidate components all read from it, ensuring
identical API type, model, max_tokens, reasoning effort, system prompt, and
format instructions across the three paths.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    """Immutable bundle of parameters shared by the main loop, force answer,
    and answer candidate components of an agent."""

    # API dispatch
    api_type: str = "chat_completion"  # "responses_api" | "chat_completion"

    # Model identity
    model_name: str = ""
    api_base: Optional[str] = None
    api_key: str = "EMPTY"

    # Generation limits
    max_output_tokens: int = 20000

    # Reasoning (Responses API only; None disables the reasoning block)
    reasoning_effort: Optional[str] = "high"

    # Prompt scaffolding
    system_prompt: Optional[str] = None
    format_instructions: str = ""
