"""Tool environment — the single retriever the agent searches against.

Thin adapter over the finalized ``searcher_component.RetrievalSearchTool`` (used
READ-ONLY).  A ``MockToolEnv`` returns deterministic fake docs so rollouts run on
CPU with no index.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class ToolEnv(abc.ABC):
    """A search tool: ``search(query) -> ranked docs`` with per-rollout ``reset``."""

    @abc.abstractmethod
    def search(self, query: str, *, original_query: Optional[str] = None,
               reasoning: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        ...

    def reset(self) -> None:
        return None


class MockToolEnv(ToolEnv):
    """Returns fake docs; the first doc for prompt ``syn-i`` is its gold doc so
    marginal-recall-style process rewards have something to score later."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._calls = 0

    def reset(self) -> None:
        self._calls = 0

    def search(self, query: str, *, original_query=None, reasoning=None, top_k=None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        base = self._calls * 100
        self._calls += 1
        return [
            {"doc_id": f"doc-{base + j}", "text": f"[mock] result {j} for '{query[:40]}'"}
            for j in range(k)
        ]


class RetrievalToolEnv(ToolEnv):
    """Adapter over ``searcher_component.RetrievalSearchTool`` (built by the caller).

    The caller constructs the ``RetrievalSearchTool`` exactly as the inference
    pipeline does (via ``searcher_component`` / ``orchestration``) and hands it in
    here — we only translate the ``search``/``reset`` calls.
    """

    def __init__(self, search_tool: Any, top_k: int = 5) -> None:
        self._tool = search_tool
        self.top_k = top_k

    def reset(self) -> None:
        self._tool.reset()

    def search(self, query: str, *, original_query=None, reasoning=None, top_k=None) -> List[Dict[str, Any]]:
        return self._tool.execute(
            query, original_query=original_query, reasoning=reasoning, top_k=top_k or self.top_k
        )
