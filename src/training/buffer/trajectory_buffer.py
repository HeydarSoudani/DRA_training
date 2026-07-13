"""Group-consistent, staleness-bounded trajectory buffer (async).

The seam between samplers (producers) and the trainer (consumer) in the async
loop.  Two guarantees that async GRPO needs:

  1. GROUP CONSISTENCY — a group emitted to the trainer contains exactly
     ``group_size`` rollouts of the SAME (prompt_id, policy_version), so the
     group-relative advantage baseline is unbiased.
  2. STALENESS BOUND — groups generated more than ``staleness_k`` versions behind
     the current trainer version are DROPPED (logged, not silently), keeping the
     policy update near-on-policy.

In-memory reference implementation; the interface is what matters — swap for
Redis / AReaL's buffer without touching samplers or the loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..data.schema import Trajectory

logger = logging.getLogger(__name__)

_Key = Tuple[str, int]  # (group_id, policy_version)


class TrajectoryBuffer:
    def __init__(self, group_size: int, staleness_k: int = 1, capacity: int = 4096) -> None:
        self.group_size = group_size
        self.staleness_k = staleness_k
        self.capacity = capacity
        self._partial: Dict[_Key, List[Trajectory]] = defaultdict(list)
        self._ready: "asyncio.Queue[List[Trajectory]]" = asyncio.Queue(maxsize=capacity)
        self._current_version = 0
        self._lock = asyncio.Lock()
        self.dropped_stale = 0

    def set_current_version(self, v: int) -> None:
        self._current_version = v

    def _too_stale(self, version: int) -> bool:
        return (self._current_version - version) > self.staleness_k

    async def push(self, traj: Trajectory) -> None:
        """Add one rollout; when its (group, version) reaches group_size, enqueue it."""
        async with self._lock:
            if self._too_stale(traj.policy_version):
                self.dropped_stale += 1
                logger.debug("Dropping stale rollout %s (v%d, cur v%d)",
                             traj.prompt_id, traj.policy_version, self._current_version)
                return
            key: _Key = (traj.group_id, traj.policy_version)
            bucket = self._partial[key]
            bucket.append(traj)
            if len(bucket) >= self.group_size:
                group = self._partial.pop(key)
                await self._ready.put(group)

    async def get_group(self, timeout: Optional[float] = None) -> Optional[List[Trajectory]]:
        """Pop a ready group (blocks up to ``timeout``; None on timeout).

        Stale-at-emit groups are skipped so the trainer never updates on a group
        older than the current version by more than ``staleness_k``.
        """
        try:
            if timeout is None:
                group = await self._ready.get()
            else:
                group = await asyncio.wait_for(self._ready.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        async with self._lock:
            if group and self._too_stale(group[0].policy_version):
                self.dropped_stale += len(group)
                return None
        return group

    def prune_partial(self) -> int:
        """Drop partial (never-completed) stale groups. Returns count dropped."""
        stale = [k for k in self._partial if self._too_stale(k[1])]
        n = 0
        for k in stale:
            n += len(self._partial.pop(k))
        if n:
            self.dropped_stale += n
        return n
