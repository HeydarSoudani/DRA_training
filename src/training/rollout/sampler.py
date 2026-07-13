"""Sampler — turns prompts into GRPO groups of trajectories.

A "group" = ``group_size`` rollouts of the SAME prompt at the SAME policy version
(the set GRPO normalizes advantages over).  Concurrency is bounded so a real
policy server isn't overwhelmed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from ..config import RolloutConfig
from ..data.schema import PromptRecord, Trajectory
from .policy_client import PolicyClient
from .tool_env import ToolEnv
from .agent_rollout import rollout_once

logger = logging.getLogger(__name__)


class Sampler:
    """Produces trajectory groups from prompts using a policy + tool."""

    def __init__(self, policy: PolicyClient, tool: ToolEnv, cfg: RolloutConfig) -> None:
        self.policy = policy
        self.tool = tool
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)

    async def _one(self, prompt: PromptRecord, group_id: str) -> Trajectory:
        async with self._sem:
            return await rollout_once(prompt, self.policy, self.tool, self.cfg, group_id=group_id)

    async def sample_group(self, prompt: PromptRecord) -> List[Trajectory]:
        """Sample ``group_size`` trajectories for one prompt at the current version.

        The group is stamped with the version captured at dispatch time; if a
        weight update lands mid-flight the async loop's staleness check handles it.
        """
        version = self.policy.version
        group_id = f"{prompt.id}@v{version}"
        tasks = [self._one(prompt, group_id) for _ in range(self.cfg.group_size)]
        trajs = await asyncio.gather(*tasks)
        # Pin the group's version to dispatch time for buffer keying.
        for t in trajs:
            t.policy_version = version
            t.group_id = group_id
        return list(trajs)
