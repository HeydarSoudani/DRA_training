"""Asynchronous GRPO driver — production long-horizon path.

Decouples generation from training (AReaL/ASearcher-style):

    producers  : continuously sample groups at the current served version and
                 push them into the trajectory buffer.
    consumer   : pops ready (same-version, non-stale) groups, scores them, runs a
                 trainer update, then weight-syncs the new version to the policy.

The buffer enforces group consistency + staleness; the trainer/weight_sync are
pluggable so the real GRPO backend drops in without touching this driver.  With
the mock components this whole thing runs on CPU.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from ..config import TrainingConfig
from ..data.schema import PromptRecord
from ..rollout.sampler import Sampler
from ..reward.base import RewardModel
from ..trainer.base import TrainerBackend
from ..trainer.weight_sync import WeightSync
from ..buffer.trajectory_buffer import TrajectoryBuffer
from ..utils.logging import log_stats

logger = logging.getLogger(__name__)


async def run_async_rl(
    cfg: TrainingConfig,
    prompts: List[PromptRecord],
    sampler: Sampler,
    reward: RewardModel,
    trainer: TrainerBackend,
    buffer: TrajectoryBuffer,
    weight_sync: WeightSync,
) -> None:
    rl = cfg.rl
    stop = asyncio.Event()
    prompt_cursor = {"i": 0}

    async def producer() -> None:
        """Keep generating groups until the consumer signals stop."""
        while not stop.is_set():
            p = prompts[prompt_cursor["i"] % len(prompts)]
            prompt_cursor["i"] += 1
            try:
                group = await sampler.sample_group(p)
                for t in group:
                    await buffer.push(t)
            except Exception:
                logger.exception("producer rollout failed for %s", p.id)
            await asyncio.sleep(0)  # yield

    async def consumer() -> None:
        updates = 0
        while updates < rl.total_steps:
            group = await buffer.get_group(timeout=30.0)
            if group is None:
                buffer.prune_partial()
                continue
            prompt = _find_prompt(prompts, group[0].prompt_id)
            for t in group:
                t.reward = reward.score(t, prompt)
            # NOTE: batch size 1 group per update for simplicity; batching multiple
            # ready groups per update is a straightforward extension.
            stats = trainer.update([group])
            buffer.set_current_version(trainer.version)
            weight_sync.sync(trainer, [sampler.policy])
            updates += 1
            stats["buffer_dropped_stale"] = buffer.dropped_stale
            log_stats(f"[async update {updates}/{rl.total_steps}]", stats)
            if cfg.trainer.save_every and updates % cfg.trainer.save_every == 0:
                trainer.save(updates)
        stop.set()

    n_producers = max(1, cfg.rollout.concurrency // max(1, cfg.rollout.group_size))
    producers = [asyncio.create_task(producer()) for _ in range(n_producers)]
    consumer_task = asyncio.create_task(consumer())

    await consumer_task
    for pr in producers:
        pr.cancel()
    await asyncio.gather(*producers, return_exceptions=True)


def _find_prompt(prompts: List[PromptRecord], prompt_id: str) -> PromptRecord:
    for p in prompts:
        if p.id == prompt_id:
            return p
    raise KeyError(prompt_id)
