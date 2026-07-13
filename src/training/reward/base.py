"""Reward interface + composite combiner.

The interface and the outcome+process combination are INFRASTRUCTURE (fixed
here).  The actual outcome metric and per-turn process credit are DESIGNABLE and
live as stubs in ``outcome_reward.py`` / ``process_reward.py``.

Combination (keep outcome dominant to resist process-reward hacking — see
MEMORY: agent-sft-rl-training-plan):
    total = outcome + lambda_process * sum(per_turn)      [process optionally clamped]
"""

from __future__ import annotations

import abc
from typing import List

from ..config import RewardConfig
from ..data.schema import PromptRecord, Trajectory, RewardBreakdown


class RewardModel(abc.ABC):
    """Scores a single trajectory against its prompt's gold signals."""

    @abc.abstractmethod
    def score(self, traj: Trajectory, prompt: PromptRecord) -> RewardBreakdown:
        ...


class OutcomeComponent(abc.ABC):
    @abc.abstractmethod
    def outcome(self, traj: Trajectory, prompt: PromptRecord) -> float:
        ...


class ProcessComponent(abc.ABC):
    @abc.abstractmethod
    def per_turn(self, traj: Trajectory, prompt: PromptRecord) -> List[float]:
        """One credit per SEARCH turn (len == traj.num_search_turns)."""
        ...


class CompositeReward(RewardModel):
    """Combines an outcome component and a process component into RewardBreakdown."""

    def __init__(self, outcome: OutcomeComponent, process: ProcessComponent, cfg: RewardConfig) -> None:
        self.outcome = outcome
        self.process = process
        self.cfg = cfg

    def score(self, traj: Trajectory, prompt: PromptRecord) -> RewardBreakdown:
        out = float(self.outcome.outcome(traj, prompt))
        per_turn = [float(x) for x in self.process.per_turn(traj, prompt)]
        proc_sum = sum(per_turn)
        if self.cfg.clamp_process:
            # Bound process contribution to |outcome| + 1 so it can't dominate.
            cap = abs(out) + 1.0
            proc_contrib = max(-cap, min(cap, self.cfg.lambda_process * proc_sum))
        else:
            proc_contrib = self.cfg.lambda_process * proc_sum
        total = out + proc_contrib
        return RewardBreakdown(
            outcome=out,
            per_turn=per_turn,
            total=total,
            info={"proc_sum": proc_sum, "lambda": self.cfg.lambda_process},
        )
