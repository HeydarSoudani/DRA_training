"""Trainer backend interface.

Separates the *plumbing* (advantage computation, versioning, checkpointing,
weight push) from the *designable* weight-update math (GRPO/FSDP), which is
delegated to a backend (AReaL/veRL adapter) — see ``grpo_backend.py``.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List

from ..data.schema import Trajectory


class TrainerBackend(abc.ABC):
    """Consumes GRPO groups and updates the policy.

    Contract:
      * ``update(groups)`` computes advantages, performs one optimization step,
        bumps the weight version, and returns scalar stats.
      * ``version`` is the current weight version (rollouts stamp it; the buffer
        bounds staleness against it).
      * ``push_weights(policy_clients)`` makes the updated weights visible to the
        generation servers (real backends broadcast; mock just sets the version).
    """

    @property
    @abc.abstractmethod
    def version(self) -> int:
        ...

    @abc.abstractmethod
    def update(self, groups: List[List[Trajectory]]) -> Dict[str, Any]:
        ...

    def push_weights(self, policy_clients: List[Any]) -> None:
        for pc in policy_clients:
            if hasattr(pc, "set_version"):
                pc.set_version(self.version)

    def save(self, step: int) -> None:
        return None
