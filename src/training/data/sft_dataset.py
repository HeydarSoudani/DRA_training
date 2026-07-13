"""SFT data generation — DESIGNABLE, stubbed.

The SFT cold-start dataset (format-competent think->search->answer trajectories)
is a designable component (teacher distillation / reject sampling / trajectory
formatting).  Only the INTERFACE is fixed here; the construction is TODO.

Interface contract:
    build_sft_dataset(cfg) -> writes a training file (chat/jsonl) to cfg.data_out
                              and returns its path.
Each example should be a multi-turn chat where OBSERVATION tokens are masked
from the loss (same invariant as RL) — the chosen SFT backend must honor that.
"""

from __future__ import annotations

from typing import List

from .schema import PromptRecord
from ..config import SFTConfig


def build_sft_dataset(cfg: SFTConfig, prompts: List[PromptRecord]) -> str:
    """Construct the SFT cold-start dataset. TODO(sft-data-gen).

    Planned options (to be designed):
      * distill trajectories from a strong teacher via the inference agents,
      * reject-sample: keep only correct + well-formed trajectories,
      * format as masked multi-turn chat and write to ``cfg.data_out``.
    """
    raise NotImplementedError(
        "SFT data generation is a designable component — not yet implemented. "
        "Fix the format (masked multi-turn chat) and a construction strategy "
        "(teacher distillation / reject sampling) here."
    )
