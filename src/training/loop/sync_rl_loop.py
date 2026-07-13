"""Synchronous GRPO driver — build/validate this BEFORE async.

Single weight version at a time: sample groups -> score -> update -> (implicitly
new version).  No buffer, no staleness — the simplest thing that exercises
masking, group formation, advantage shaping, and the trainer contract.  Prove
correctness here, then switch to the async driver.
"""

from __future__ import annotations

import logging
from typing import List

from ..config import TrainingConfig
from ..data.schema import PromptRecord
from ..rollout.sampler import Sampler
from ..reward.base import RewardModel
from ..trainer.base import TrainerBackend
from ..utils.logging import log_stats

logger = logging.getLogger(__name__)


async def run_sync_rl(
    cfg: TrainingConfig,
    prompts: List[PromptRecord],
    sampler: Sampler,
    reward: RewardModel,
    trainer: TrainerBackend,
) -> None:
    rl = cfg.rl
    idx = 0
    for step in range(rl.total_steps):
        # take the next slice of prompts (wrap around)
        batch: List[PromptRecord] = []
        for _ in range(rl.prompts_per_step):
            batch.append(prompts[idx % len(prompts)])
            idx += 1

        groups = []
        for p in batch:
            group = await sampler.sample_group(p)
            for t in group:
                t.reward = reward.score(t, p)
            groups.append(group)

        stats = trainer.update(groups)
        # keep the policy version in lockstep with the trainer (on-policy)
        trainer.push_weights([sampler.policy])
        log_stats(f"[sync step {step + 1}/{rl.total_steps}]", stats)

        if cfg.trainer.save_every and (step + 1) % cfg.trainer.save_every == 0:
            trainer.save(step + 1)
