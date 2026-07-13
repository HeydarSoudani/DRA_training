"""Core data contracts shared across the whole training pipeline.

These dataclasses are the *spine* every service agrees on:

    PromptRecord   — one training question (+ gold signals for reward)
    Turn           — one agent step: generated text + injected observation,
                     with token-level ids + loss mask
    Trajectory     — a full multi-turn rollout for one prompt
    RewardBreakdown— the reward server's verdict for a trajectory

Two invariants (enforced in ``rollout.masking``):
    * ``len(token_ids) == len(loss_mask)`` for every Turn.
    * ``loss_mask == 0`` on every observation (retrieved-doc) token; only the
      policy's own generated tokens are trained on (Search-R1 ``state_masking``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt (input to a rollout)
# ---------------------------------------------------------------------------

@dataclass
class PromptRecord:
    """A single training prompt plus whatever gold signals reward may need.

    ``answers`` / ``qrels`` / ``criteria`` are optional so the same record type
    works for RL (needs reward signals) and for smoke tests (needs none).
    """

    id: str
    question: str
    answers: Optional[List[str]] = None            # gold answers -> outcome reward
    qrels: Optional[Dict[str, int]] = None         # {doc_id: relevance} -> marginal-recall process reward
    criteria: Optional[List[str]] = None           # info-need criteria -> coverage process reward
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Turn + Trajectory (output of a rollout)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One agent step.

    ``gen_*`` are the policy's own tokens (think + action) — trained on.
    ``obs_*`` are the injected tool observation tokens — masked out.
    The concatenated (gen ++ obs) view is what the trainer consumes; use
    :meth:`token_ids` / :meth:`loss_mask`.
    """

    index: int
    text: str = ""                                 # human-readable generated text (think + action)
    action_kind: str = "search"                    # "search" | "answer" | "stop"
    query: Optional[str] = None                    # the search query, if action_kind == "search"

    gen_token_ids: List[int] = field(default_factory=list)
    gen_logprobs: Optional[List[float]] = None     # per-token logprob under the behavior policy

    obs_token_ids: List[int] = field(default_factory=list)
    obs_text: str = ""
    obs_doc_ids: List[str] = field(default_factory=list)

    @property
    def token_ids(self) -> List[int]:
        return list(self.gen_token_ids) + list(self.obs_token_ids)

    @property
    def loss_mask(self) -> List[int]:
        return [1] * len(self.gen_token_ids) + [0] * len(self.obs_token_ids)


@dataclass
class RewardBreakdown:
    """Reward server verdict for one trajectory.

    ``per_turn`` has one entry per *search* turn (process credit); ``outcome`` is
    the terminal answer reward; ``total`` is the combined scalar the trainer uses
    as the trajectory return before group-normalization.
    """

    outcome: float = 0.0
    per_turn: List[float] = field(default_factory=list)
    total: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A full multi-turn rollout for one prompt."""

    prompt_id: str
    group_id: str                                  # rollouts sharing a prompt+version form a GRPO group
    policy_version: int                            # weight version that GENERATED this trajectory
    turns: List[Turn] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = "answer"                    # "answer" | "max_turns" | "error"

    reward: Optional[RewardBreakdown] = None       # filled by the reward server
    advantage: Optional[float] = None              # trajectory-level GRPO advantage (filled by trainer)

    # ---- convenience views for masking / trainer ----
    @property
    def num_search_turns(self) -> int:
        return sum(1 for t in self.turns if t.action_kind == "search")

    def token_ids(self) -> List[int]:
        out: List[int] = []
        for t in self.turns:
            out.extend(t.token_ids)
        return out

    def loss_mask(self) -> List[int]:
        out: List[int] = []
        for t in self.turns:
            out.extend(t.loss_mask)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
