"""Dataclasses and constants for the trajectory tracker."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


AFFECT_NONE = "none"


@dataclass
class Decision:
    action: str  # "continue" | "intervene" | "stop"
    scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingOutput:
    """Deterministic output for the early-stopping affect."""
    reasoning: str


@dataclass
class CriticalThinkingOutput:
    """Structured output from the critical-thinking LLM call."""
    reasoning: str
    search_query: str


@dataclass
class TrackerCriticalThinkResult:
    """Active critical_think: tracker generated a critical query and the mixin executed it.

    Returned by ``post_search_evaluate`` when action is ``"intervene"``.
    The agent should inject this as an additional full turn in its
    trajectory (critical_think -> critical_search_query -> critical_observation).
    """
    critical_think: str
    critical_search_query: str
    critical_docs: List[Dict[str, Any]]
    critical_observation: str
    critical_think_scores: Dict[str, Any]
    critical_think_iter: int = 0


@dataclass
class TrackerCriticalThinkDeferred:
    """Deferred critical_think: tracker decided to intervene but retrieval
    has not been executed yet.

    Returned by ``post_search_evaluate`` when ``defer_critical_search=True``
    and action is ``"intervene"``.  Call ``_execute_deferred_critical_search``
    to perform the retrieval and get a full ``TrackerCriticalThinkResult``.
    """
    critical_think: str
    critical_search_query: str
    scores: Dict[str, Any] = field(default_factory=dict)
    iter_num: int = 0


@dataclass
class TrackerEarlyStopResult:
    """Early-stopping signal: tracker decided the agent should stop searching
    and produce its final answer with the evidence collected so far.

    Returned by ``post_search_evaluate`` when action is ``"stop"``.
    The agent should break its search loop and force answer generation.
    """
    reasoning: str
    scores: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryDecision:
    """Result of a trajectory evaluation step.

    Attributes:
        action: One of ``"continue"``, ``"intervene"``, ``"stop"``.
        scores: Per-signal scores for logging / analysis.
        critical_thinking_output: Set when action is ``"intervene"``.
            The mixin handles retrieval and re-evaluation.
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
    """Snapshot of the trajectory state passed to notice generators."""
    original_query: str
    last_turns: List[TrajectoryTurn]
    current_subqueries: List[str]
    action: str  # "intervene"
    current_scores: Dict[str, Any]
    score_history: List[Dict[str, Any]]
