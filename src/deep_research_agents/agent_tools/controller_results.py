"""Agent-side result types for the controller's post-search evaluation.

These describe how an *agent* consumes a controller decision: they carry the
retrieved ``critical_docs`` and ``critical_observation`` that the agent itself
produces while acting on an ``"intervene"`` decision, so they live in the
agent layer rather than inside ``controller_component``.

Returned by ``BasicAgent.post_search_evaluate``:
- ``CriticalThinkResult``  — active intervene: critical query executed.
- ``CriticalThinkDeferred`` — intervene decided, retrieval not yet executed.
- ``EarlyStopResult``      — stop: force final answer generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CriticalThinkResult:
    """Active critical_think: the controller generated a critical query and
    the agent executed it.

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
class CriticalThinkDeferred:
    """Deferred critical_think: the controller decided to intervene but
    retrieval has not been executed yet.

    Returned by ``post_search_evaluate`` when ``defer_critical_search=True``
    and action is ``"intervene"``.  Call ``_execute_deferred_critical_search``
    to perform the retrieval and get a full ``CriticalThinkResult``.
    """
    critical_think: str
    critical_search_query: str
    scores: Dict[str, Any] = field(default_factory=dict)
    iter_num: int = 0


@dataclass
class EarlyStopResult:
    """Early-stopping signal: the controller decided the agent should stop
    searching and produce its final answer with the evidence collected so far.

    Returned by ``post_search_evaluate`` when action is ``"stop"``.
    The agent should break its search loop and force answer generation.
    """
    reasoning: str
    scores: Dict[str, Any] = field(default_factory=dict)
