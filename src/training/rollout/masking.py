"""Loss-mask helpers and trajectory validation.

CORRECTNESS-CRITICAL.  The single most important agentic-RL detail is that only
the policy's own generated tokens contribute to the loss; injected tool
observations are masked out (Search-R1 ``state_masking``).  A silent misalignment
here trains the model on retrieved text and quietly wrecks RL, so everything is
asserted.
"""

from __future__ import annotations

from typing import List

from ..data.schema import Trajectory, Turn


def make_observation_text(docs: List[dict], *, max_docs: int = 5, max_chars: int = 500) -> str:
    """Render retrieved docs into the observation string injected into context."""
    lines = []
    for d in docs[:max_docs]:
        did = d.get("doc_id") or d.get("id") or ""
        txt = d.get("text") or d.get("relevant_text") or ""
        lines.append(f"[{did}] {txt[:max_chars]}")
    return "<result>\n" + "\n".join(lines) + "\n</result>"


def validate_turn(turn: Turn) -> None:
    ids = turn.token_ids
    mask = turn.loss_mask
    assert len(ids) == len(mask), (
        f"Turn {turn.index}: token/mask length mismatch ({len(ids)} != {len(mask)})"
    )
    # generated tokens must be trained on (mask 1), observation tokens masked (0)
    assert mask[: len(turn.gen_token_ids)] == [1] * len(turn.gen_token_ids), \
        f"Turn {turn.index}: generated tokens must have loss_mask == 1"
    assert mask[len(turn.gen_token_ids):] == [0] * len(turn.obs_token_ids), \
        f"Turn {turn.index}: observation tokens must have loss_mask == 0"
    if turn.gen_logprobs is not None:
        assert len(turn.gen_logprobs) == len(turn.gen_token_ids), \
            f"Turn {turn.index}: logprob/token length mismatch"


def validate_trajectory(traj: Trajectory) -> None:
    for t in traj.turns:
        validate_turn(t)
    assert len(traj.token_ids()) == len(traj.loss_mask()), \
        "Trajectory: assembled token/mask length mismatch"
