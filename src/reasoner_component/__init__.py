"""LLM generator components for deep research agents."""

from .base import BaseGenerator
from .api_generator import APIGenerator, OpenRouterGenerator
from .hf_generator import HFGenerator
from .factory import create_generator

# Backward-compatible re-exports
from utils.llm_client import LiteLLMClient, get_litellm_client

__all__ = [
    # New interface
    "BaseGenerator",
    "APIGenerator",
    "OpenRouterGenerator",
    "HFGenerator",
    "create_generator",
    # Backward compat
    "LiteLLMClient",
    "get_litellm_client",
]
