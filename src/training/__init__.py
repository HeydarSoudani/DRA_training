"""DRA agent training infrastructure (SFT + async GRPO).

This package holds the *training* pipeline plumbing.  It is a sibling of the
finalized *inference* pipeline (``experiments/dra_inference.py`` +
``deep_research_agents`` / ``searcher_component`` / ``reasoner_component`` /
``controller_component``) and consumes those components **read-only** — nothing
here mutates the inference code path.

Design stance (see MEMORY: agent-sft-rl-training-plan):
  * INFRASTRUCTURE FIRST.  The rollout orchestration, trajectory buffer, loop
    drivers, and component interfaces are implemented here.
  * DESIGNABLE parts (reward formula, SFT data generation, the GRPO/FSDP update
    math, weight-sync transport) are left as ``# TODO`` stubs behind stable
    interfaces so they can be designed without reshaping the plumbing.
  * A ``mock`` policy / tool / trainer let the ENTIRE loop run on CPU with no
    GPUs, so masking / group formation / advantage shaping can be validated
    before any real training compute is wired.

Entry point: ``experiments/dra_train.py`` -> ``training.pipeline.run_sft`` /
``training.pipeline.run_rl``.
"""

__all__ = ["config", "pipeline"]
