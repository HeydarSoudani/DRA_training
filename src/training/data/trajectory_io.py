"""Trajectory <-> JSONL serialization.

Layout mirrors the inference ``trajectory/{query_id}.jsonl`` idea: a meta line
followed by one line per turn.  Kept deliberately simple and lossless so RL
rollouts can be inspected / replayed / used for offline analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

from .schema import Trajectory, Turn, RewardBreakdown


def write_trajectory(path: Union[str, Path], traj: Trajectory) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        meta = {
            "_type": "meta",
            "prompt_id": traj.prompt_id,
            "group_id": traj.group_id,
            "policy_version": traj.policy_version,
            "final_answer": traj.final_answer,
            "stop_reason": traj.stop_reason,
            "reward": asdict(traj.reward) if traj.reward else None,
            "advantage": traj.advantage,
            "num_turns": len(traj.turns),
        }
        fh.write(json.dumps(meta) + "\n")
        for t in traj.turns:
            fh.write(json.dumps({"_type": "turn", **asdict(t)}) + "\n")


def read_trajectory(path: Union[str, Path]) -> Trajectory:
    path = Path(path)
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    meta = lines[0]
    turns = [Turn(**{k: v for k, v in ln.items() if k != "_type"}) for ln in lines[1:]]
    reward = RewardBreakdown(**meta["reward"]) if meta.get("reward") else None
    return Trajectory(
        prompt_id=meta["prompt_id"],
        group_id=meta["group_id"],
        policy_version=meta["policy_version"],
        turns=turns,
        final_answer=meta.get("final_answer", ""),
        stop_reason=meta.get("stop_reason", "answer"),
        reward=reward,
        advantage=meta.get("advantage"),
    )
