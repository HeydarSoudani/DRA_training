"""SFT backend — TODO.

Cold-start supervised fine-tuning on masked multi-turn trajectories.  Delegate to
a standard trainer (LLaMA-Factory / TRL / veRL fsdp_sft) rather than hand-roll.

TODO(sft-backend):
  * take the dataset from ``data.sft_dataset.build_sft_dataset``,
  * run the chosen SFT trainer honoring OBSERVATION token masking,
  * emit the reference/init checkpoint consumed by the RL stage.
"""

from __future__ import annotations

from ..config import SFTConfig


def run_sft_backend(cfg: SFTConfig, data_path: str) -> str:
    raise NotImplementedError(
        "SFT backend is a TODO. Wire LLaMA-Factory / TRL / veRL-fsdp here; it must "
        "honor observation-token masking. Returns the cold-start checkpoint path."
    )
