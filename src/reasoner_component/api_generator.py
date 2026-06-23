"""API-based LLM generators: OpenAI/Anthropic (APIGenerator) and OpenRouter (OpenRouterGenerator)."""

import os
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .base import BaseGenerator
from utils.llm_client import LiteLLMClient, get_litellm_client

T = TypeVar("T", bound=BaseModel)


class _APIBase(BaseGenerator):
    """Delegates all generator calls to an internal LiteLLMClient."""

    _client: LiteLLMClient

    @property
    def token_meter(self):
        """Cumulative token usage of the underlying LiteLLM client."""
        return self._client.token_meter

    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return self._client.complete(messages, **kwargs)

    async def acomplete(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return await self._client.acomplete(messages, **kwargs)

    def complete_with_structured_output(
        self, messages: List[Dict[str, Any]], response_format: Type[T], **kwargs
    ) -> T:
        return self._client.complete_with_structured_output(messages, response_format, **kwargs)

    async def acomplete_with_structured_output(
        self, messages: List[Dict[str, Any]], response_format: Type[T], **kwargs
    ) -> T:
        return await self._client.acomplete_with_structured_output(messages, response_format, **kwargs)

    def complete_with_stop_on_pattern(
        self, messages: List[Dict[str, Any]], stop_patterns: List[str], **kwargs
    ) -> tuple[str, Optional[str]]:
        return self._client.complete_with_stop_on_pattern(messages, stop_patterns, **kwargs)

    async def acomplete_with_stop_on_pattern(
        self, messages: List[Dict[str, Any]], stop_patterns: List[str], **kwargs
    ) -> tuple[str, Optional[str]]:
        return await self._client.acomplete_with_stop_on_pattern(messages, stop_patterns, **kwargs)

    def complete_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs
    ) -> tuple[str, List[Dict[str, Any]]]:
        return self._client.complete_with_tools(messages, tools, **kwargs)

    async def acomplete_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], **kwargs
    ) -> tuple[str, List[Dict[str, Any]]]:
        return await self._client.acomplete_with_tools(messages, tools, **kwargs)

    def cleanup(self) -> None:
        self._client.cleanup()


class APIGenerator(_APIBase):
    """LLM generator for OpenAI (Azure / direct) and Anthropic/Claude APIs.

    Model name is resolved to the correct provider config via environment variables
    (see get_litellm_client for the full mapping). Additional kwargs are forwarded
    to LiteLLM directly (e.g. top_p, request_timeout).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self._client = get_litellm_client(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
            **kwargs,
        )


class OpenRouterGenerator(_APIBase):
    """LLM generator for the OpenRouter API (https://openrouter.ai).

    Accepts model names with or without the ``openrouter/`` prefix,
    e.g. ``"openrouter/anthropic/claude-3.5-sonnet"`` or
    ``"anthropic/claude-3.5-sonnet"``.

    The API key is read from the ``OPENROUTER_API_KEY`` environment variable
    unless passed explicitly.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        bare_name = model_name.removeprefix("openrouter/")
        config: Dict[str, Any] = {
            "model": f"openrouter/{bare_name}",
            "api_base": self.BASE_URL,
            "api_key": api_key or os.getenv("OPENROUTER_API_KEY"),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self._client = LiteLLMClient(config, metadata=metadata)
