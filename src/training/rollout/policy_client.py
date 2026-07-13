"""Policy client — the weight-updatable generation endpoint for RL rollouts.

The inference ``reasoner_component`` generators return TEXT only (no token ids /
logprobs), so they cannot drive RL as-is.  This module defines a training-only
client that returns token-level data, WITHOUT importing or modifying
``reasoner_component``.

    PolicyClient        — interface: generate(prompt) -> Generation; .version
    MockPolicyClient    — pure-Python, CPU-only; emits a valid think/search/answer
                          protocol so the whole loop is testable with no server.
    ServerPolicyClient  — TODO: SGLang/vLLM OpenAI-compatible client returning
                          token ids + logprobs.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Generation:
    """One generation step's output."""
    text: str
    token_ids: List[int] = field(default_factory=list)
    logprobs: Optional[List[float]] = None
    finish_reason: str = "stop"


class PolicyClient(abc.ABC):
    """Generation endpoint whose weights change across training steps.

    ``version`` is the weight version currently served; rollouts stamp it onto
    their trajectories so the buffer can form same-version GRPO groups and bound
    staleness.
    """

    def __init__(self) -> None:
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def set_version(self, v: int) -> None:
        self._version = v

    @abc.abstractmethod
    async def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> Generation:
        ...

    async def close(self) -> None:  # optional cleanup hook
        return None


# ---------------------------------------------------------------------------
# Mock (CPU, no server) — deterministic-ish given (prompt, turn, version)
# ---------------------------------------------------------------------------

def _toy_tokenize(text: str) -> List[int]:
    """Whitespace toy tokenizer -> stable int ids (no real tokenizer needed)."""
    return [(abs(hash(tok)) % 50000) + 1 for tok in text.split()]


class MockPolicyClient(PolicyClient):
    """Emits a valid protocol trajectory for smoke-testing the loop on CPU.

    For the first ``search_turns`` calls in a rollout it emits a ``<search>``
    action; after that it emits an ``<answer>``.  The rollout loop tells it which
    turn it is via the ``turn`` marker embedded in the prompt (see AgentRollout).
    """

    def __init__(self, search_turns: int = 2) -> None:
        super().__init__()
        self.search_turns = search_turns

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> Generation:
        # The rollout appends a "[[turn=N]]" marker to the prompt; parse it.
        turn = 0
        marker = "[[turn="
        if marker in prompt:
            try:
                turn = int(prompt.rsplit(marker, 1)[1].split("]]", 1)[0])
            except Exception:
                turn = 0
        if turn < self.search_turns:
            text = f"<think>need more evidence for step {turn}</think>" \
                   f"<search>query for step {turn} v{self._version}</search>"
            finish = "stop"
        else:
            text = f"<think>enough evidence</think><answer>answer-0</answer>"
            finish = "stop"
        return Generation(text=text, token_ids=_toy_tokenize(text), logprobs=None, finish_reason=finish)


# ---------------------------------------------------------------------------
# Server client (SGLang / vLLM) — TODO
# ---------------------------------------------------------------------------

class ServerPolicyClient(PolicyClient):
    """OpenAI-compatible completions client returning token ids + logprobs.

    TODO(policy-server):
      * POST to {server_url}/v1/completions with logprobs enabled,
      * parse per-token ids + logprobs from the response,
      * map the served weight version (set via weight_sync after each update)
        onto ``self._version``.
    Deliberately NOT wired to reasoner_component so the inference path is untouched.
    """

    def __init__(self, server_url: str, model: str) -> None:
        super().__init__()
        self.server_url = server_url
        self.model = model

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> Generation:
        raise NotImplementedError(
            "ServerPolicyClient is a TODO. Implement the SGLang/vLLM completions "
            "call returning token ids + logprobs. Use MockPolicyClient for CPU tests."
        )
