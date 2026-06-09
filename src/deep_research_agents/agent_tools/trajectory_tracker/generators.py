"""Critical thinking and answer candidate generators for the trajectory tracker.

Provides generators that produce:
- Critical thinking outputs (intervention queries when trajectory is stale)
- Answer candidate outputs (mid-loop best-effort answers)
"""

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .tracker_types import CriticalThinkingOutput, TrajectoryContext
from prompts.trajectory_tracker.answer_prompts import (
    CANDIDATE_GENERATION_INSTRUCTION,
    TAG_FORMAT,
    AnswerCandidateOutput,
    extract_answer_candidates,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load prompt templates
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "trajectory_tracker"


def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


_CRITICAL_THINKING_SYSTEM = _load_prompt("critical_thinking/system_prompt.txt")

from prompts.trajectory_tracker.critical_thinking.user_prompt import CRITICAL_THINKING_USER_TEMPLATE as _CRITICAL_THINKING_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Critical thinking generators
# ---------------------------------------------------------------------------

class CriticalThinkingGenerator:
    """Generate trajectory intervention notices via an LLM call.

    Placeholder — ``generate()`` raises ``NotImplementedError``.
    Subclass or replace this with a concrete implementation that calls
    an LLM to produce context-aware notices.
    """

    def generate(self, context: TrajectoryContext) -> CriticalThinkingOutput:
        raise NotImplementedError(
            "CriticalThinkingGenerator.generate() is not yet implemented. "
            "Provide a concrete subclass."
        )


class LLMCriticalThinkingGenerator(CriticalThinkingGenerator):
    """Generate context-aware trajectory notices via an LLM call.

    Returns a ``CriticalThinkingOutput`` with ``reasoning`` and ``search_query``
    parsed from the LLM's JSON response.  When the LLM call or JSON parsing
    fails, returns an output with an empty ``search_query`` so the caller
    can fall back to a fixed notice.
    """

    def __init__(self, llm_client: Any, n_last_turns: Optional[int] = 10) -> None:
        self._llm = llm_client
        self._n_last_turns = n_last_turns

    @staticmethod
    def _build_coverage_section(ac_data: Optional[Dict[str, Any]]) -> str:
        """Format covered, critical-gap, and minor-gap aspects into a prompt section."""
        if ac_data is None:
            return ""

        aspects = ac_data.get("aspects", [])
        covered = [a["name"] for a in aspects if a.get("status") == "covered"]
        critical = ac_data.get("critical_gaps", [])
        minor = ac_data.get("minor_gaps", [])

        if not covered and not critical and not minor:
            return ""

        total = ac_data.get("total", len(aspects))
        num_covered = ac_data.get("num_covered", len(covered))
        num_partial = ac_data.get("num_partial", len(minor))
        num_not_covered = ac_data.get("num_not_covered", len(critical))

        parts = [
            f"Overall: {num_covered}/{total} covered, "
            f"{num_partial}/{total} partial, "
            f"{num_not_covered}/{total} not covered"
        ]
        if covered:
            bullet_list = "\n".join(f"- {name}" for name in covered)
            parts.append(
                f"Covered aspects (already have strong evidence — do NOT target these):\n{bullet_list}"
            )
        if critical:
            bullet_list = "\n".join(f"- {name}" for name in critical)
            parts.append(
                f"Critical gaps (no evidence found — highest priority):\n{bullet_list}"
            )
        if minor:
            bullet_list = "\n".join(f"- {name}" for name in minor)
            parts.append(
                f"Minor gaps (partial evidence — could be strengthened):\n{bullet_list}"
            )

        return "\n== Aspect Coverage ==\n" + "\n\n".join(parts) + "\n"

    def generate(self, context: TrajectoryContext) -> CriticalThinkingOutput:
        turn_lines = []
        for i, turn in enumerate(context.last_turns, 1):
            turn_lines.append(f"{i}. Think: {turn.thinking}")
            turn_lines.append(f"   Search query: {turn.search_query}")

        user_msg = _CRITICAL_THINKING_USER_TEMPLATE.format(
            original_query=context.original_query,
            num_turns=len(context.last_turns),
            last_turns="\n".join(turn_lines) if turn_lines else "(no prior turns)",
        )

        messages = [
            {"role": "system", "content": _CRITICAL_THINKING_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._llm.complete(messages, max_tokens=512, temperature=0.3)
        except Exception:
            logger.warning("LLMCriticalThinkingGenerator LLM call failed", exc_info=True)
            return CriticalThinkingOutput(reasoning="", search_query="")

        try:
            parsed = json.loads(raw.strip())
            return CriticalThinkingOutput(
                reasoning=parsed.get("reasoning", ""),
                search_query=parsed.get("search_query", ""),
            )
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError, ValueError):
            logger.warning("LLMCriticalThinkingGenerator JSON parse failed: %s", raw[:200])
            search_query = ""
            query_m = re.search(r'"search_query"\s*:\s*"([^"]+)"', raw)
            if not query_m:
                query_m = re.search(r'[Ss]earch(?:\s+for)?:\s*["\']?(.+?)["\']?\s*(?:\n|$)', raw)
            if query_m:
                search_query = query_m.group(1).strip()
            return CriticalThinkingOutput(reasoning=raw.strip(), search_query=search_query)


# ---------------------------------------------------------------------------
# Answer candidate generators
# ---------------------------------------------------------------------------

class LLMAnswerCandidateGenerator:
    """Generate answer candidates from the trajectory evidence via an LLM call.

    Called during ``TrajectoryTracker.evaluate()`` *before* the decision
    maker, so that each iteration produces best-effort answer candidates
    alongside the signals.
    """

    def __init__(
        self,
        llm_client: Any,
        format_instructions: str = TAG_FORMAT,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> None:
        self._llm = llm_client
        self._format_instructions = format_instructions
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _build_messages(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[Dict[str, str]]:
        if not isinstance(trajectory, list):
            messages: List[Dict[str, str]] = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            instruction = (
                f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
                f"{self._format_instructions}"
            )
            messages.append({"role": "user", "content": instruction})
            return messages

        messages = copy.deepcopy(trajectory)
        instruction = (
            f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
            f"{self._format_instructions}"
        )
        messages.append({"role": "user", "content": instruction})
        return messages

    def _get_model_id(self) -> str:
        if hasattr(self._llm, "config"):
            return self._llm.config.get("model", "unknown")
        return "unknown"

    def generate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
        messages = self._build_messages(
            original_query, trajectory,
            reasoning_path=reasoning_path, seen_top_k=seen_top_k,
        )

        call_kwargs: Dict[str, Any] = {
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            call_kwargs["temperature"] = self._temperature

        try:
            raw = self._llm.complete(
                messages,
                strip_think=False,
                **call_kwargs,
            )
        except Exception:
            logger.warning("LLMAnswerCandidateGenerator LLM call failed", exc_info=True)
            return [AnswerCandidateOutput(candidate="no candidate", reasoning="LLM call failed")]

        if not raw or not raw.strip():
            return [AnswerCandidateOutput(candidate="no candidate", reasoning="LLM returned empty response")]

        candidates, format_matched = extract_answer_candidates(raw)
        if not candidates and not format_matched:
            logger.warning(
                "Answer candidate LLM (%s) returned text but no candidates were extracted. "
                "Raw (first 300 chars): %s",
                self._get_model_id(), raw[:300],
            )
            return [AnswerCandidateOutput(candidate="no candidate", reasoning=raw.strip()[:500])]
        return candidates


class ResponsesAPICandidateGenerator:
    """Answer candidate generator using the OpenAI Responses API.

    Used for agents that run on the Responses API (e.g. OSS).
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        format_instructions: str,
        max_output_tokens: int = 20000,
        reasoning_effort: str = "high",
    ) -> None:
        self._api_base = api_base
        self._api_key = api_key
        self._model_name = model_name
        self._format_instructions = format_instructions
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort

    def generate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
        if not isinstance(trajectory, list):
            logger.warning(
                "ResponsesAPICandidateGenerator requires a list of messages "
                "(Responses API format). Got %s; skipping.",
                type(trajectory).__name__,
            )
            return [AnswerCandidateOutput(
                candidate="no candidate",
                reasoning="trajectory is not a Responses API message list",
            )]

        import openai

        messages = copy.deepcopy(trajectory)
        instruction = (
            f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
            f"{self._format_instructions}"
        )
        messages.append({"role": "user", "content": instruction})

        request = {
            "model": self._model_name,
            "max_output_tokens": self._max_output_tokens,
            "input": messages,
            "truncation": "auto",
            "reasoning": {
                "effort": self._reasoning_effort,
                "summary": "detailed",
            },
        }

        try:
            client = openai.OpenAI(base_url=self._api_base, api_key=self._api_key)
            response = client.responses.create(**request)
        except Exception:
            logger.warning("ResponsesAPICandidateGenerator API call failed", exc_info=True)
            return [AnswerCandidateOutput(candidate="no candidate", reasoning="LLM call failed")]

        raw = ""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                raw = "\n".join(p.text for p in item.content)

        if not raw or not raw.strip():
            return [AnswerCandidateOutput(candidate="no candidate", reasoning="LLM returned empty response")]

        candidates, format_matched = extract_answer_candidates(raw)
        if not candidates and not format_matched:
            logger.warning(
                "ResponsesAPICandidateGenerator returned text but no candidates extracted. "
                "Raw (first 300 chars): %s",
                raw[:300],
            )
            return [AnswerCandidateOutput(candidate="no candidate", reasoning=raw.strip()[:500])]
        return candidates
