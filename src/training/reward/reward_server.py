"""Standalone reward server — TODO.

CaRR pattern: serve a RewardModel over HTTP so reward compute (esp. any LLM-judge
signal) scales independently of the trainer and doesn't starve it on long
rollouts.  For sync/async CPU validation the reward runs in-process via the
RewardModel directly; this server is only needed at scale.

TODO(reward-server):
  * FastAPI app exposing POST /score {trajectory} -> RewardBreakdown,
  * batching + independent replica scaling,
  * health/version endpoints.
"""

from __future__ import annotations


def build_app(reward_model):  # pragma: no cover - not needed for CPU validation
    raise NotImplementedError(
        "Standalone reward server is a TODO. In-process RewardModel is used for "
        "sync/async validation; add FastAPI here when reward compute must scale."
    )
