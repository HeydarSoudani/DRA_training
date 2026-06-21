"""Controller classes for the trajectory tracker.

Provides the abstract ``ControllerPolicy`` interface and the LLM-based
implementation that selects continue/intervene/stop based on configurable
signal combinations (prompt variants).
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tracker_types import Decision
from .prompts.criteria_coverage import (
    CriteriaCoverageSummary,
    format_summary_for_controller,
)

logger = logging.getLogger(__name__)

_VALID_CONTROLLER_ACTIONS = {"continue", "intervene", "stop"}
_VALID_CONTROLLER_VARIANTS = {"nov", "nov_cov", "nov_sim", "nov_cov_sim", "sim", "cov_sim"}

# ---------------------------------------------------------------------------
# Load system prompts from disk
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


_CONTROLLER_SYSTEM_PROMPTS = {
    "nov": _load_prompt("controller/nov.txt"),
    "nov_cov": _load_prompt("controller/nov_cov.txt"),
    "nov_sim": _load_prompt("controller/nov_sim.txt"),
    "nov_cov_sim": _load_prompt("controller/nov_cov_sim.txt"),
    "sim": _load_prompt("controller/sim.txt"),
    "cov_sim": _load_prompt("controller/cov_sim.txt"),
}

from .prompts.controller.user_prompt import CONTROLLER_USER_TEMPLATES as _CONTROLLER_USER_TEMPLATES


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ControllerPolicy(ABC):
    reward_signal: Optional[str] = None

    @abstractmethod
    def decide(
        self, signals: Dict[str, Optional[float]], iter_num: int, **kwargs: Any,
    ) -> Decision: ...

    def update(self, reward: float) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# LLM-based controller
# ---------------------------------------------------------------------------

class LLMControllerPolicy(ControllerPolicy):
    """LLM-based controller that uses an LLM to decide continue/intervene/stop.

    Signal composition is controlled by ``controller_prompt_variant``:

    - ``"nov"``: doc_novelty only
    - ``"nov_cov"``: doc_novelty + criteria_coverage
    - ``"nov_sim"``: doc_novelty + consec_query_sim + orig_query_sim
    - ``"nov_cov_sim"``: doc_novelty + criteria_coverage + consec_query_sim + orig_query_sim
    - ``"sim"``: consec_query_sim (primary) + orig_query_sim (guardrail)
    - ``"cov_sim"``: criteria_coverage (primary) + consec_query_sim + orig_query_sim
    """

    def __init__(self, llm_client: Any, n_last_turns: Optional[int] = 3, controller_prompt_variant: str = "nov_cov_sim", history_window: Optional[int] = None, max_iteration: Optional[int] = None) -> None:
        if controller_prompt_variant not in _VALID_CONTROLLER_VARIANTS:
            raise ValueError(f"Unknown controller_prompt_variant: {controller_prompt_variant!r}. Must be one of {sorted(_VALID_CONTROLLER_VARIANTS)}")
        self._llm = llm_client
        self._n_last_turns = n_last_turns
        self._controller_prompt_variant = controller_prompt_variant
        self._history_window = history_window
        self._max_iteration = max_iteration

        self._has_nov = controller_prompt_variant not in ("sim", "cov_sim")
        self._has_sim = controller_prompt_variant in ("sim", "nov_sim", "nov_cov_sim", "cov_sim")
        self._has_nov_only = controller_prompt_variant == "nov"
        self._has_ac = False
        self._has_cov = controller_prompt_variant in ("nov_cov", "nov_cov_sim", "cov_sim")

        self._system_prompt = _CONTROLLER_SYSTEM_PROMPTS[controller_prompt_variant]
        self._user_template = _CONTROLLER_USER_TEMPLATES[controller_prompt_variant]

    @staticmethod
    def _format_candidate(candidates: List[Dict[str, str]]) -> str:
        if not candidates:
            return "no candidate"
        parts = []
        for ac in candidates:
            text = ac.get("candidate", "no candidate")
            escaped = text.replace('\\', '\\\\').replace('"', '\\"')
            parts.append(f'"{escaped}"')
        return "; ".join(parts)

    @staticmethod
    def _format_coverage_compact(ac_data: Optional[Dict[str, Any]]) -> str:
        if not ac_data:
            return "N/A"
        covered = ac_data.get('num_covered', 0)
        total = ac_data.get('total', 0)
        critical = len(ac_data.get('critical_gaps', []))
        minor = len(ac_data.get('minor_gaps', []))
        return f"{covered}/{total} covered ({critical} critical, {minor} minor gaps)"

    def decide(
        self, signals: Dict[str, Optional[float]], iter_num: int, **kwargs: Any,
    ) -> Decision:
        original_query: str = kwargs.get("original_query", "")
        score_history: List[Dict[str, Any]] = kwargs.get("score_history", [])
        answer_candidates: List[Dict[str, str]] = kwargs.get("answer_candidates", [])
        criteria_summary = kwargs.get("criteria_summary")

        signal_lines = []
        action_lines = []
        history_slice = score_history if self._history_window is None else score_history[-self._history_window:]
        history_offset = len(score_history) - len(history_slice)
        for j, scores in enumerate(history_slice):
            i = scores.get('iter_num', history_offset + j)
            parts = [f"  iter {i}:"]
            if self._has_nov:
                _nov = scores.get('doc_novelty', 'N/A')
                _n_novel = scores.get('num_novel_docs', '?')
                _n_total = scores.get('num_total_docs', '?')
                parts.append(f"novelty={_nov} ({_n_novel} out of {_n_total} docs)")
            if self._has_sim:
                parts.append(f"consec_sim={scores.get('consec_query_sim', 'N/A')}")
            if self._has_ac:
                iter_acs = scores.get('answer_candidates', [])
                parts.append(f"answer_candidate={self._format_candidate(iter_acs)}")
            if self._has_cov:
                parts.append(f"criterion_cov={self._format_coverage_compact(scores.get('criteria_coverage'))}")
            if self._has_sim:
                parts.append(f"orig_sim={scores.get('orig_query_sim', 'N/A')}")
            signal_lines.append(" ".join(parts[:1]) + " " + ", ".join(parts[1:]))
            action_lines.append(
                f"  iter {i}: {scores.get('controller_action', 'N/A')}"
            )

        template_kwargs = {
            "original_query": original_query,
            "iter_num": iter_num,
            "max_iter": self._max_iteration or "?",
            "signal_history": "\n".join(signal_lines) if signal_lines else "(first iteration)",
            "action_history": "\n".join(action_lines) if action_lines else "(first iteration)",
        }
        if self._has_nov:
            num_novel_docs = kwargs.get("num_novel_docs", "?")
            num_total_docs = kwargs.get("num_total_docs", "?")
            template_kwargs["doc_novelty"] = signals.get("doc_novelty", "N/A")
            template_kwargs["num_novel_docs"] = num_novel_docs
            template_kwargs["num_total_docs"] = num_total_docs
        if self._has_sim:
            template_kwargs["consec_query_sim"] = signals.get("consec_query_sim", "N/A")
            template_kwargs["orig_query_sim"] = signals.get("orig_query_sim", "N/A")
        if self._has_ac:
            template_kwargs["current_answer_candidate"] = self._format_candidate(answer_candidates)
        if self._has_cov:
            if criteria_summary is not None:
                template_kwargs["current_criteria_coverage"] = format_summary_for_controller(criteria_summary)
            else:
                template_kwargs["current_criteria_coverage"] = "N/A"

        user_msg = self._user_template.format(**template_kwargs)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._llm.complete(messages, max_tokens=512, temperature=0.0)
        except Exception:
            logger.warning("LLMControllerPolicy LLM call failed", exc_info=True)
            return Decision(
                action="continue",
                scores={"model_type": "controller", "controller_action": "continue"},
            )

        try:
            parsed = json.loads(raw.strip())
            controller_action = parsed.get("action", "continue")
            reasoning = parsed.get("reasoning", "")

            if controller_action not in _VALID_CONTROLLER_ACTIONS:
                controller_action = "continue"

            return Decision(
                action=controller_action,
                scores={
                    "model_type": "controller",
                    "controller_reasoning": reasoning,
                    "controller_action": controller_action,
                },
            )
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError, ValueError):
            logger.warning("LLMControllerPolicy JSON parse failed: %s", raw.strip())
            return Decision(
                action="continue",
                scores={"model_type": "controller", "controller_action": "continue", "controller_reasoning": raw.strip()[:200]},
            )

    def reset(self) -> None:
        pass
