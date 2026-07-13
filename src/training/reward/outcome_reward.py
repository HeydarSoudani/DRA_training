"""Outcome reward — DESIGNABLE, stubbed.

Terminal answer reward (e.g. EM/F1 vs gold answers).  The metric choice and
normalization are designable; later this can reuse the finalized
``evaluation`` metrics (READ-ONLY).  The stub returns 0.0 so the composite
reward and the whole loop run without a designed metric.
"""

from __future__ import annotations

from .base import OutcomeComponent
from ..config import RewardConfig
from ..data.schema import PromptRecord, Trajectory


class OutcomeReward(OutcomeComponent):
    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def outcome(self, traj: Trajectory, prompt: PromptRecord) -> float:
        # TODO(reward-outcome): compute EM/F1 (or LLM-judge) of traj.final_answer
        # vs prompt.answers using the finalized evaluation metrics (read-only).
        # Returning 0.0 keeps the pipeline runnable until the metric is designed.
        return 0.0


class MockOutcomeReward(OutcomeComponent):
    """Deterministic non-zero signal for smoke tests: 1.0 iff the (mock) answer
    matches a gold answer, else 0.0.  Lets the loop show reward variance on CPU."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg

    def outcome(self, traj: Trajectory, prompt: PromptRecord) -> float:
        golds = prompt.answers or []
        return 1.0 if traj.final_answer and traj.final_answer in golds else 0.0
