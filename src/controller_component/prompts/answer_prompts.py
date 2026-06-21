"""Answer prompts: instructions, format strings, and extraction utilities.

Centralises everything related to forcing a final answer and generating
mid-loop answer candidates:

- FINAL_ANSWER_INSTRUCTION        -- loop-end: agent must produce a definitive answer.
- CANDIDATE_GENERATION_INSTRUCTION -- mid-loop: agent may respond "no candidate".
- Per-agent format instructions (TAG, OSS, REACT, BOXED, DRTULU, SELFASK).
- Agent-to-format mapping + helper.
- AnswerCandidateOutput dataclass + extraction helpers.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ======================================================================
# Instructions
# ======================================================================

FINAL_ANSWER_INSTRUCTION = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer."
)

CANDIDATE_GENERATION_INSTRUCTION = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer.\n"
    "If the evidence is insufficient to answer, respond with 'no candidate'."
)

TONGYI_FORCE_ANSWER = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer "
    "in the following format:"
    "<think>your final thinking</think>\n"
    "<answer>your answer</answer>"
)

TONGYI_CANDIDATE_ANSWER = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information "
    "above, think again and provide what you consider the most likely answer "
    "in the following format:"
    "<think>your final thinking</think>\n"
    "<answer>your answer</answer>\n"
    "If the evidence is truly insufficient, respond with "
    "<answer>no candidate</answer>"
)

# ======================================================================
# Per-agent format instructions
# ======================================================================
TAG_FORMAT = (
    "Provide your answer in the following format:\n"
    "<think>your final thinking</think>\n"
    "<answer>your answer</answer>"
)

OSS_FORMAT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation, citing evidence in [docid] brackets}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100%}"
)

REACT_FORMAT = (
    "Provide your answer in the following format:\n"
    "<think>your final thinking</think>\n"
    '<action>finish(answer="your answer")</action>'
)

WEBWEAVER_FORMAT = (
    "Based on the evidence gathered so far, provide a concise answer.\n"
    "IMPORTANT: The <answer> tag must contain ONLY the short factual answer,no explanations, no reasoning, no bullet points. Put all reasoning inside <think>.\n\n"
    "<think>your reasoning over the collected evidence</think>\n"
    "<answer>your short answer</answer>"
)

CPM_EXPLORE_FORMAT = (
    "## How to Respond\n\n"
    "1. **Think**: Provide a SHORT and CONCISE final thinking.\n"
    "2. **Answer**: Provide your SHORT most-likely answer or 'no candidate' inside <answer></answer> tags."
)

BOXED_FORMAT = "Provide your final answer in \\boxed{} format."

DRTULU_FORMAT = "Provide your answer using <answer>...</answer> tags."

SELFASK_FORMAT = "So the final answer is: "

# ======================================================================
# Agent-to-format mapping
# ======================================================================

AGENT_AC_FORMAT = {
    "searcho1":    BOXED_FORMAT,
    "react":       REACT_FORMAT,
    "drtulu":      DRTULU_FORMAT,
    "selfask":     SELFASK_FORMAT,
    "oss":         OSS_FORMAT,
    "glm":         OSS_FORMAT,
    "cpm_explore": CPM_EXPLORE_FORMAT,
    "tongyi":      TAG_FORMAT,
    "webweaver":   WEBWEAVER_FORMAT,
}


def get_candidate_format(agentic_model: str) -> str:
    """Return the answer-candidate format instructions for the given agent."""
    return AGENT_AC_FORMAT.get(agentic_model, TAG_FORMAT)


# ======================================================================
# Structured output + extraction
# ======================================================================

@dataclass
class AnswerCandidateOutput:
    """Structured output from the answer-candidate LLM call."""
    candidate: str
    reasoning: str = ""
    confidence: Optional[float] = None

def _parse_candidate_list(text: str) -> List[str]:
    """Split a candidate value into individual candidates.

    Handles three forms:
    - ``"no candidate"`` -> empty list
    - ``"[c1, c2, c3]"`` -> ``["c1", "c2", "c3"]``
    - ``"single answer"`` -> ``["single answer"]``
    """
    stripped = text.strip()
    if stripped.lower() == "no candidate":
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
        inner = stripped[1:-1]
        return [c.strip().strip('"').strip("'") for c in inner.split(",") if c.strip()]
    return [stripped]

_AGENT_TO_PATTERNS = {
    "searcho1":    ["boxed"],
    "react":       ["finish_action"],
    "drtulu":      ["answer_tag"],
    "selfask":     ["so_final"],
    "oss":         ["exact_answer"],
    "glm":         ["exact_answer"],
    "cpm_explore": ["answer_tag"],
    "tongyi":      ["answer_tag"],
    "webweaver":   ["answer_tag"],
}

_ALL_PATTERN_NAMES = ["answer_tag", "finish_action", "exact_answer", "boxed", "so_final"]


def extract_answer_candidates(
    raw: str,
    expected_format: Optional[str] = None,
) -> Tuple[List[AnswerCandidateOutput], bool]:
    """Parse answer candidates and optional ``<think>`` tags from raw LLM output.

    Returns ``(candidates, format_matched)`` where *format_matched* is
    ``True`` when at least one known format pattern was found in the text
    (even if the value was ``"no candidate"``).

    When *expected_format* is an agent name (e.g. ``"react"``, ``"oss"``),
    the expected pattern is tried first; remaining patterns run only as
    fallback.  When ``None``, all patterns run (backward-compatible).
    """
    thinking = ""
    think_m = re.search(r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", raw, re.DOTALL)
    if think_m:
        thinking = think_m.group(1).strip()

    if not thinking:
        expl_m = re.search(r"Explanation:\s*(.+?)(?=\n(?:Exact Answer|Confidence):|$)", raw, re.DOTALL)
        if expl_m:
            _expl = expl_m.group(1).strip()
            if not re.search(r"\{your\b", _expl, re.IGNORECASE):
                thinking = _expl

    confidence: Optional[float] = None
    conf_m = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)\s*%", raw)
    if conf_m:
        confidence = max(0.0, min(100.0, float(conf_m.group(1))))

    candidates: List[AnswerCandidateOutput] = []
    seen: set = set()
    format_matched = False

    def _is_template_placeholder(text: str) -> bool:
        return bool(re.search(r"\{your\b", text, re.IGNORECASE))

    def _add(text: str) -> None:
        normed = text.lower()
        if normed and normed != "no candidate" and normed not in seen:
            if _is_template_placeholder(text):
                return
            seen.add(normed)
            candidates.append(AnswerCandidateOutput(
                candidate=text, reasoning=thinking, confidence=confidence,
            ))

    def _add_raw(value: str) -> None:
        for c in _parse_candidate_list(value):
            _add(c)

    def _try_answer_tag() -> bool:
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    def _try_finish_action() -> bool:
        m = re.search(r'finish\(answer="(.*?)"\)', raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    def _try_exact_answer() -> bool:
        m = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", raw)
        if m:
            val = re.sub(r"\s*Confidence:\s*\d+(?:\.\d+)?\s*%\s*$", "", m.group(1).strip())
            _add_raw(val)
            return True
        return False

    def _try_boxed() -> bool:
        m = re.search(r"\\boxed\{(.*?)}", raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    def _try_so_final() -> bool:
        m = re.search(r"(?:So the candidate answer is|So the final answer is):\s*(.+?)(?:\n\n|$)", raw, re.DOTALL)
        if m:
            _add_raw(m.group(1).strip())
            return True
        return False

    _PATTERN_FNS = {
        "answer_tag": _try_answer_tag,
        "finish_action": _try_finish_action,
        "exact_answer": _try_exact_answer,
        "boxed": _try_boxed,
        "so_final": _try_so_final,
    }

    primary = _AGENT_TO_PATTERNS.get(expected_format, _ALL_PATTERN_NAMES)
    fallback = [p for p in _ALL_PATTERN_NAMES if p not in primary]

    for name in primary:
        if _PATTERN_FNS[name]():
            format_matched = True

    if not candidates:
        for name in fallback:
            if _PATTERN_FNS[name]():
                format_matched = True

    if format_matched and not candidates:
        candidates.append(AnswerCandidateOutput(
            candidate="no candidate",
            reasoning=thinking or "format matched but no usable candidate extracted",
            confidence=confidence,
        ))

    return candidates, format_matched
