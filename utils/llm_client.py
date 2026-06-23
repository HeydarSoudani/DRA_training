"""Backward-compatibility shim.

All LLM/generation logic moved to the ``reasoner_component`` package. This
module re-exports the names historically imported from ``utils.llm_client`` so
existing callers keep working during the migration.

Prefer importing from ``reasoner_component`` directly; this shim will be removed
once all callers are migrated.

Note: imports come from submodules (not the package ``__init__``) so importing
``utils.llm_client`` does not pull in torch/transformers.
"""

from reasoner_component.litellm_client import LiteLLMClient
from reasoner_component.api import get_litellm_client
from reasoner_component.vllm import VLLMClient
from reasoner_component.setup import setup_llm

__all__ = [
    "LiteLLMClient",
    "get_litellm_client",
    "VLLMClient",
    "setup_llm",
]
