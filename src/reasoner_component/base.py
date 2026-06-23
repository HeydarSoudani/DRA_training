"""Base classes for all LLM generators.

``BaseGenerator`` is the abstract interface every generator implements.
``LiteLLMBackedGenerator`` is the shared base for generators that speak to their
backend through a ``LiteLLMClient`` (``APIGenerator`` and ``VLLMGenerator``);
they differ only in how that client is configured.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .litellm_client import LiteLLMClient

T = TypeVar("T", bound=BaseModel)


class BaseGenerator(ABC):

    @abstractmethod
    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> str: ...

    @abstractmethod
    async def acomplete(self, messages: List[Dict[str, Any]], **kwargs) -> str: ...

    def complete_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        response_format: Type[T],
        **kwargs,
    ) -> T:
        raise NotImplementedError

    async def acomplete_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        response_format: Type[T],
        **kwargs,
    ) -> T:
        raise NotImplementedError

    def complete_with_stop_on_pattern(
        self,
        messages: List[Dict[str, Any]],
        stop_patterns: List[str],
        **kwargs,
    ) -> tuple[str, Optional[str]]:
        raise NotImplementedError

    async def acomplete_with_stop_on_pattern(
        self,
        messages: List[Dict[str, Any]],
        stop_patterns: List[str],
        **kwargs,
    ) -> tuple[str, Optional[str]]:
        raise NotImplementedError

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> tuple[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    async def acomplete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> tuple[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class LiteLLMBackedGenerator(BaseGenerator):
    """Delegates all generator calls to an internal ``LiteLLMClient``.

    Subclasses only need to set ``self._client`` in ``__init__``; the full
    ``BaseGenerator`` interface (structured output, tools, stop patterns) is
    implemented here by forwarding to that client.
    """

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
