"""Mock trainer — no GPUs, no real weights.

Runs the FULL training loop plumbing (advantage computation, versioning, weight
push, stats) so masking / group formation / staleness / the async driver can all
be validated on CPU before the real GRPO backend is wired.  It does everything a
real backend does EXCEPT the gradient step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import TrainerBackend
from .advantage import group_advantages, per_token_advantages
from ..data.schema import Trajectory

logger = logging.getLogger(__name__)


class MockTrainer(TrainerBackend):
    def __init__(self, save_dir: Optional[str] = None) -> None:
        self._version = 0
        self.save_dir = save_dir

    @property
    def version(self) -> int:
        return self._version

    def update(self, groups: List[List[Trajectory]]) -> Dict[str, Any]:
        all_advs: List[float] = []
        all_returns: List[float] = []
        n_tokens = 0
        n_trained_tokens = 0
        for group in groups:
            advs = group_advantages(group)
            all_advs.extend(advs)
            all_returns.extend(t.reward.total if t.reward else 0.0 for t in group)
            for t in group:
                mask = t.loss_mask()
                n_tokens += len(mask)
                n_trained_tokens += sum(mask)
                # exercise the broadcast so shape bugs surface here, not later
                pta = per_token_advantages(t)
                assert len(pta) == len(mask)

        # (no gradient step — this is the mock)
        self._version += 1

        n = len(all_returns) or 1
        mean_ret = sum(all_returns) / n
        mean_adv = sum(all_advs) / (len(all_advs) or 1)
        stats = {
            "version": self._version,
            "groups": len(groups),
            "trajectories": len(all_returns),
            "mean_return": mean_ret,
            "mean_advantage": mean_adv,
            "tokens": n_tokens,
            "trained_tokens": n_trained_tokens,
            "masked_frac": 1.0 - (n_trained_tokens / n_tokens) if n_tokens else 0.0,
        }
        return stats

    def save(self, step: int) -> None:
        if not self.save_dir:
            return
        p = Path(self.save_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"mock_ckpt_step{step}.json").write_text(json.dumps({"version": self._version, "step": step}))
