"""Lightweight token accounting for LLM clients.

``TokenMeter`` accumulates prompt/completion token usage across LLM calls.  It
lives in the leaf ``utils`` package (no dependency on ``evaluation``) so the LLM
client can record usage without import cycles.

Two reading patterns:

* **Per-query total** — snapshot at the start of a query and read cumulative
  totals at the end (or diff two snapshots).
* **Per-step delta** — call :meth:`since_last_step` at each trajectory-step
  boundary to get the tokens consumed since the previous boundary.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TokenMeter:
    """Accumulate LLM token usage for a single agent/client."""

    input_tokens: int = 0
    output_tokens: int = 0
    num_calls: int = 0

    # Cursor marking the totals at the last :meth:`since_last_step` boundary.
    _step_cursor: Dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "num_calls": 0}
    )

    def record(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record one LLM call's usage (missing/None counts treated as 0)."""
        self.input_tokens += int(prompt_tokens or 0)
        self.output_tokens += int(completion_tokens or 0)
        self.num_calls += 1

    def record_usage(self, usage) -> None:
        """Record usage from a provider response ``usage`` object or dict.

        Accepts anything exposing ``prompt_tokens``/``completion_tokens`` as
        attributes or dict keys.  Silently ignores ``None``.
        """
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
        # Chat Completions uses prompt_tokens/completion_tokens; the Responses API
        # uses input_tokens/output_tokens.
        prompt = get("prompt_tokens")
        if prompt is None:
            prompt = get("input_tokens")
        completion = get("completion_tokens")
        if completion is None:
            completion = get("output_tokens")
        self.record(prompt, completion)

    def snapshot(self) -> Dict[str, int]:
        """Return cumulative totals as a plain dict."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "num_calls": self.num_calls,
        }

    def since_last_step(self) -> Dict[str, int]:
        """Return usage accumulated since the previous call, then advance cursor."""
        delta = {
            "input_tokens": self.input_tokens - self._step_cursor["input_tokens"],
            "output_tokens": self.output_tokens - self._step_cursor["output_tokens"],
            "num_calls": self.num_calls - self._step_cursor["num_calls"],
        }
        delta["total_tokens"] = delta["input_tokens"] + delta["output_tokens"]
        self._step_cursor = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "num_calls": self.num_calls,
        }
        return delta

    def reset(self) -> None:
        """Zero all counters and the step cursor (e.g. between queries)."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.num_calls = 0
        self._step_cursor = {"input_tokens": 0, "output_tokens": 0, "num_calls": 0}
