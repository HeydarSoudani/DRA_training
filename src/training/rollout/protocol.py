"""Agent action protocol — parsing the policy's tagged output.

Mirrors the Search-R1 / SmartSearch tag convention so trajectories are
interoperable with the inference-side format:

    <think> ... </think>       reasoning (trained on, not an action)
    <search> query </search>   issue a retrieval query
    <answer> text </answer>    finish with a final answer

This is intentionally tiny and self-contained; the exact tokenizer/tags can be
swapped without touching the rollout loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class Action:
    kind: str                 # "search" | "answer" | "stop"
    query: Optional[str] = None
    answer: Optional[str] = None


def parse_action(text: str) -> Action:
    """Extract the action from one generation step.

    ``<answer>`` takes precedence (terminal); else ``<search>``; else the step is
    malformed and we stop (``kind == 'stop'``) so the loop can end gracefully.
    """
    m_ans = _ANSWER_RE.search(text)
    if m_ans:
        return Action(kind="answer", answer=m_ans.group(1).strip())
    m_search = _SEARCH_RE.search(text)
    if m_search:
        q = m_search.group(1).strip()
        if q:
            return Action(kind="search", query=q)
    return Action(kind="stop")
