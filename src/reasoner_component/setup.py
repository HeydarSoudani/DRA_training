"""Pipeline LLM setup helpers (CLI wiring).

``setup_llm`` builds the generator the pipeline runs on, picking the backend from
the parsed CLI args via ``create_generator`` (API / vLLM / HF).
"""

import os
import logging
from typing import Optional

from utils.config import SELF_MANAGED_LLM_AGENTS, is_local_finetuned
from .base import BaseGenerator
from .factory import create_generator

logger = logging.getLogger(__name__)


def setup_llm(args, num_gpus: int) -> Optional[BaseGenerator]:
    """Instantiate the generator from parsed CLI args.

    Picks the backend behind a uniform ``BaseGenerator``:
      * self-managed agents       → ``None`` (they create their own connection)
      * local finetuned model     → ``VLLMGenerator`` (OpenAI-compatible vLLM server)
      * in-process HF model        → ``HFGenerator`` (``hf/`` prefix or a known slug)
      * everything else           → ``APIGenerator`` (Azure/Anthropic/OpenAI/OpenRouter)

    The final branch lets ``create_generator`` infer the backend from the model
    name, so an ``hf/``-prefixed (or registered) model routes to ``HFGenerator``
    while ordinary slugs route to ``APIGenerator``.
    """
    if args.agentic_model in SELF_MANAGED_LLM_AGENTS:
        return None

    if is_local_finetuned(args.agentic_model, args.llm_model):
        hf_model = args.llm_model
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:6008/v1")
        generator = create_generator(
            hf_model,
            backend="vllm",
            api_base=api_base,
            api_key="EMPTY",
            litellm_prefix="openai",  # preserve model string "openai/<hf_model>"
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens_per_call,
            request_timeout=args.request_timeout,
        )
        print(f"Using vLLM-served finetuned model: {hf_model} at {api_base}")
        return generator

    return create_generator(
        args.llm_model,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        max_completion_tokens=args.llm_max_tokens_per_call,
        metadata={"model": args.llm_model},
        request_timeout=args.request_timeout,
    )
