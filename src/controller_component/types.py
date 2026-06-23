"""Dataclasses and constants for the controller component."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


INTERVENTION_NONE = "none"


@dataclass
class Decision:
    action: str  # "continue" | "intervene" | "stop"
    scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingOutput:
    """Deterministic output for the early-stopping intervention."""
    reasoning: str


@dataclass
class CriticalThinkingOutput:
    """Structured output from the critical-thinking LLM call."""
    reasoning: str
    search_query: str


@dataclass
class TrajectoryDecision:
    """Result of a trajectory evaluation step.

    Attributes:
        action: One of ``"continue"``, ``"intervene"``, ``"stop"``.
        scores: Per-signal scores for logging / analysis.
        critical_thinking_output: Set when action is ``"intervene"``.
            The agent handles retrieval and re-evaluation.
        early_stopping_output: Set when action is ``"stop"``.
    """
    action: str  # "continue" | "intervene" | "stop"
    scores: Dict[str, float] = field(default_factory=dict)
    critical_thinking_output: Optional[CriticalThinkingOutput] = None
    early_stopping_output: Optional[EarlyStoppingOutput] = None


@dataclass
class TrajectoryTurn:
    """A single turn in the search trajectory."""
    thinking: str
    search_query: str


@dataclass
class TrajectoryContext:
    """Snapshot of the trajectory state passed to critical-thinking generators."""
    original_query: str
    last_turns: List[TrajectoryTurn]
    current_subqueries: List[str]
    action: str  # "intervene"
    current_scores: Dict[str, Any]
    score_history: List[Dict[str, Any]]
