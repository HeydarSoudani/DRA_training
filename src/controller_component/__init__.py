"""Controller component: post-search quality gate for deep research agents.

After each search step the controller receives the sub-query and retrieved
documents, computes signals (doc novelty, consecutive query similarity,
original query similarity, supervised marginal recall, and criteria
coverage), and decides whether the agent should continue normally or
receive an injected observation.

Decisions:
- **continue**: trajectory is healthy — continue normally.
- **intervene**: trajectory is stale — inject critical thinking and a new search.
- **stop**: trajectory is exhausted — force final answer generation.

Module layout:
    types              — dataclasses and constants
    signals            — DocNoveltySignal, ConsecQuerySimilaritySignal,
                         OrigQuerySimilaritySignal, MarginalRecallSignal,
                         CriteriaCoverageSignal, encode_fn_from_retriever
    controller_policies — ControllerPolicy, LLMControllerPolicy
    generators         — CriticalThinkingGenerator, LLMCriticalThinkingGenerator,
                         LLMAnswerCandidateGenerator, ResponsesAPICandidateGenerator
    controller         — Controller
"""

from .types import (
    INTERVENTION_NONE,
    Decision,
    EarlyStoppingOutput,
    CriticalThinkingOutput,
    TrajectoryDecision,
    TrajectoryTurn,
    TrajectoryContext,
)
from .signals import (
    DocNoveltySignal,
    ConsecQuerySimilaritySignal,
    OrigQuerySimilaritySignal,
    MarginalRecallSignal,
    CriteriaCoverageSignal,
    encode_fn_from_retriever,
)
from .controller_policies import (
    ControllerPolicy,
    LLMControllerPolicy,
)
from .generators import (
    CriticalThinkingGenerator,
    LLMCriticalThinkingGenerator,
    LLMAnswerCandidateGenerator,
    ResponsesAPICandidateGenerator,
)
from .controller import Controller

__all__ = [
    "INTERVENTION_NONE",
    "Decision",
    "EarlyStoppingOutput",
    "CriticalThinkingOutput",
    "TrajectoryDecision",
    "TrajectoryTurn",
    "TrajectoryContext",
    "DocNoveltySignal",
    "ConsecQuerySimilaritySignal",
    "OrigQuerySimilaritySignal",
    "MarginalRecallSignal",
    "CriteriaCoverageSignal",
    "encode_fn_from_retriever",
    "ControllerPolicy",
    "LLMControllerPolicy",
    "CriticalThinkingGenerator",
    "LLMCriticalThinkingGenerator",
    "LLMAnswerCandidateGenerator",
    "ResponsesAPICandidateGenerator",
    "Controller",
]
