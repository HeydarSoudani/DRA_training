"""Factory for creating the right generator from a model name."""

from .base import BaseGenerator
from .api_generator import APIGenerator, OpenRouterGenerator
from .hf_generator import HFGenerator

# Known HuggingFace model repo slugs that require local loading.
# Add new local models here to make them routable via create_generator.
_HF_MODELS: set[str] = {
    "openbmb/AgentCPM-Report",
    "rl-research/DR-Tulu-8B",
}


def _is_hf_model(model_name: str) -> bool:
    return model_name in _HF_MODELS or model_name.startswith("hf/")


def create_generator(model_name: str, **kwargs) -> BaseGenerator:
    """Instantiate the correct generator for *model_name*.

    Routing rules (in priority order):
      1. ``openrouter/<name>``  → OpenRouterGenerator
      2. known HF slug or ``hf/<name>`` prefix → HFGenerator
      3. everything else → APIGenerator (OpenAI / Anthropic via LiteLLM)

    All extra *kwargs* are forwarded to the generator constructor.
    """
    if model_name.startswith("openrouter/"):
        return OpenRouterGenerator(model_name, **kwargs)
    if _is_hf_model(model_name):
        bare = model_name.removeprefix("hf/")
        return HFGenerator(bare, **kwargs)
    return APIGenerator(model_name, **kwargs)
