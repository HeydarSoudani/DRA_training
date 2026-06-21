"""Abstract base class for all LLM generators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

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
