"""Weight synchronization: trainer -> policy servers.

Interface + a local (mock) implementation that just advances the served version.
The real transport (GPU-GPU / NCCL broadcast or checkpoint reload every k steps)
is a TODO tied to the chosen backend.
"""

from __future__ import annotations

import abc
from typing import Any, List


class WeightSync(abc.ABC):
    @abc.abstractmethod
    def sync(self, trainer, policy_clients: List[Any]) -> None:
        ...


class LocalVersionSync(WeightSync):
    """Mock transport: propagate the trainer's version to the policy clients.

    Sufficient for CPU validation (MockPolicyClient reads its version to change
    generations across updates, so staleness/versioning is exercised)."""

    def sync(self, trainer, policy_clients: List[Any]) -> None:
        trainer.push_weights(policy_clients)


class BroadcastWeightSync(WeightSync):  # pragma: no cover
    """TODO(weight-sync): real GPU-GPU / NCCL weight broadcast every k steps."""

    def sync(self, trainer, policy_clients: List[Any]) -> None:
        raise NotImplementedError(
            "Real weight sync is a TODO. Broadcast updated actor weights to the "
            "SGLang/vLLM servers (GPU-GPU/NCCL or checkpoint reload)."
        )
