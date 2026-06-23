"""LLM reasoner component: the single home for all LLM/generation logic.

Generators (all behind ``BaseGenerator``):
  * ``APIGenerator``  — hosted APIs: Claude / OpenAI / OpenRouter
  * ``VLLMGenerator`` — self-hosted OpenAI-compatible vLLM server
  * ``HFGenerator``   — weights loaded in-process via transformers

Use ``create_generator(model_name, backend=...)`` to pick automatically.

Low-level transports (``LiteLLMClient``, ``VLLMClient``), provider routing
(``get_litellm_client``, ``resolve_provider_config``) and the CLI setup helper
(``setup_llm``) also live here.
"""

from .base import BaseGenerator, LiteLLMBackedGenerator
from .litellm_client import LiteLLMClient
from .api import APIGenerator, get_litellm_client, resolve_provider_config
from .vllm import VLLMGenerator, VLLMClient, get_openai_client
from .factory import create_generator
from .setup import setup_llm

__all__ = [
    # Generators
    "BaseGenerator",
    "LiteLLMBackedGenerator",
    "APIGenerator",
    "VLLMGenerator",
    "HFGenerator",
    "create_generator",
    # Transports
    "LiteLLMClient",
    "VLLMClient",
    "get_openai_client",
    # Provider routing
    "get_litellm_client",
    "resolve_provider_config",
    # CLI setup
    "setup_llm",
]


def __getattr__(name):
    # Lazy import so callers that only need API/vLLM don't pull in torch.
    if name == "HFGenerator":
        from .hf import HFGenerator

        return HFGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
