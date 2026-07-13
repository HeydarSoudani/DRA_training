"""Real GRPO backend — TODO (AReaL / veRL adapter).

The weight-update math (FSDP actor, KL to ref, importance-sampling / staleness
correction, optimizer step) is delegated to a mature async RL engine rather than
hand-rolled.  Recommended: fork/adapt ASearcher-on-AReaL (single-retriever,
GRPO, retrieved-token masking already implemented) and drive it through this
adapter — see MEMORY: agent-sft-rl-training-plan.

TODO(grpo-backend):
  * construct the AReaL/veRL trainer from cfg.trainer.extra,
  * in update(): feed token_ids + loss_mask + per_token advantages (from
    ``advantage.per_token_advantages``) into the engine's optimization step,
  * apply KL(cfg.kl_coef) to the SFT ref and the decoupled IS/staleness term,
  * bump the weight version and implement push_weights() as a GPU-GPU broadcast
    to the policy servers (see weight_sync.py).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import TrainerBackend
from ..config import TrainerConfig
from ..data.schema import Trajectory


class GRPOBackend(TrainerBackend):
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self._version = 0
        # TODO(grpo-backend): initialize AReaL/veRL engine, ref model, optimizer.

    @property
    def version(self) -> int:
        return self._version

    def update(self, groups: List[List[Trajectory]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "GRPOBackend is a TODO. Delegate the GRPO/FSDP update to AReaL/veRL "
            "(advantages come from trainer.advantage). Use MockTrainer to validate "
            "the loop plumbing first."
        )
