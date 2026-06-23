"""Factory for creating the right generator from a model name.

Three backends, each behind ``BaseGenerator``:
  * ``api``   → ``APIGenerator``   (Claude / OpenAI / OpenRouter, hosted)
  * ``vllm``  → ``VLLMGenerator``  (self-hosted OpenAI-compatible vLLM server)
  * ``hf``    → ``HFGenerator``    (weights loaded in-process via transformers)
"""

from typing import Optional

from .base import BaseGenerator

# HuggingFace model repo slugs that must be loaded in-process (HF backend).
# Currently empty: every finetuned model in the flow is served via an
# OpenAI-compatible vLLM server (see is_local_finetuned / setup_llm). To route a
# future model to in-process loading, add its slug here or address it with an
# ``hf/`` prefix; create_generator picks the HF backend automatically.
_HF_MODELS: set[str] = set()


def _infer_backend(model_name: str) -> str:
    """Infer the backend from *model_name* prefixes / known slugs."""
    if model_name.startswith("openrouter/"):
        return "api"
    if model_name.startswith("vllm/"):
        return "vllm"
    if model_name in _HF_MODELS or model_name.startswith("hf/"):
        return "hf"
    return "api"


def create_generator(
    model_name: str,
    backend: Optional[str] = None,
    **kwargs,
) -> BaseGenerator:
    """Instantiate the correct generator for *model_name*.

    Args:
        model_name: Friendly model name. Recognised prefixes:
            ``openrouter/`` → api, ``vllm/`` → vllm, ``hf/`` or a known HF slug → hf.
        backend: Optional explicit backend (``"api"`` / ``"vllm"`` / ``"hf"``) that
            overrides prefix inference.
        **kwargs: Forwarded to the generator constructor.
    """
    backend = (backend or _infer_backend(model_name)).lower()

    if backend == "hf":
        # Import lazily so callers that only need API/vLLM don't pull in torch.
        from .hf import HFGenerator, ensure_transformers_version

        bare = model_name.removeprefix("hf/")
        ensure_transformers_version(bare)
        return HFGenerator(bare, **kwargs)

    if backend == "vllm":
        from .vllm import VLLMGenerator

        return VLLMGenerator(model_name, **kwargs)

    if backend == "api":
        from .api import APIGenerator

        return APIGenerator(model_name, **kwargs)

    raise ValueError(f"Unknown backend '{backend}' for model '{model_name}'.")
