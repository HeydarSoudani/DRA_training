"""Criteria coverage prompts: system instructions, user templates, and extraction.

Provides the prompts and parsing logic for the CriteriaCoverageSignal of the
controller component. The signal decomposes a query into information-need
criteria, tracks their coverage status across iterations, and reports a structured
summary for logging and (later) controller consumption.

Two modes:
- **static**: criteria are extracted verbatim from the query (e.g. BrowseCompPlus
  where the query already enumerates required criteria).
- **dynamic**: an LLM decomposes the query into criteria and updates the list
  each iteration as new evidence arrives.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Dataclasses
# ======================================================================

VALID_STATUSES = {"not_covered", "partial", "covered"}
_MAX_EVIDENCE_LENGTH = 500


@dataclass
class Criterion:
    """A single information-need criterion of a query."""
    name: str
    status: str = "not_covered"  # "not_covered" | "partial" | "covered"
    evidence: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "evidence": self.evidence}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Criterion":
        status = d.get("status", "not_covered")
        if status not in VALID_STATUSES:
            status = "not_covered"
        evidence = d.get("evidence", "")
        if len(evidence) > _MAX_EVIDENCE_LENGTH:
            evidence = evidence[:_MAX_EVIDENCE_LENGTH] + "..."
        return cls(
            name=d.get("name", ""),
            status=status,
            evidence=evidence,
        )


@dataclass
class CriteriaCoverageSummary:
    """Structured output from one criteria-coverage evaluation step."""
    criteria: List[Criterion] = field(default_factory=list)
    num_covered: int = 0
    num_partial: int = 0
    num_not_covered: int = 0
    total: int = 0
    critical_gaps: List[str] = field(default_factory=list)
    minor_gaps: List[str] = field(default_factory=list)
    new_criteria_this_iter: List[str] = field(default_factory=list)
    removed_criteria_this_iter: List[str] = field(default_factory=list)
    stable_since: Optional[int] = None
    frozen: bool = False
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criteria": [a.to_dict() for a in self.criteria],
            "num_covered": self.num_covered,
            "num_partial": self.num_partial,
            "num_not_covered": self.num_not_covered,
            "total": self.total,
            "critical_gaps": self.critical_gaps,
            "minor_gaps": self.minor_gaps,
            "new_criteria_this_iter": self.new_criteria_this_iter,
            "removed_criteria_this_iter": self.removed_criteria_this_iter,
            "stable_since": self.stable_since,
            "frozen": self.frozen,
            "reasoning": self.reasoning,
        }


# ======================================================================
# System prompts (loaded from separate .txt files)
# ======================================================================
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

CRITERIA_INIT_SYSTEM = (_PROMPTS_DIR / "init_system.txt").read_text(encoding="utf-8").strip()
CRITERIA_UPDATE_SYSTEM = (_PROMPTS_DIR / "update_system.txt").read_text(encoding="utf-8").strip()

CRITERIA_INIT_STATIC_SYSTEM = (_PROMPTS_DIR / "init_static_system.txt").read_text(encoding="utf-8").strip()
CRITERIA_UPDATE_STATIC_SYSTEM = (_PROMPTS_DIR / "update_static_system.txt").read_text(encoding="utf-8").strip()

FROZEN_INSTRUCTION = "The criterion list is FROZEN: you may only change statuses (tick). Do NOT add or remove criteria."
UNFROZEN_INSTRUCTION = "The criterion list is open for modification: you may add, remove, or tick criteria."


# ======================================================================
# User templates
# ======================================================================

CRITERIA_INIT_USER_TEMPLATE = """\
Query: {query}

Decompose this query into its key information-need criteria. \
Identify {min_criteria} to {max_criteria} distinct criteria that a comprehensive answer must address.

All criteria start with status "not_covered" and empty evidence.\
"""

CRITERIA_INIT_STATIC_USER_TEMPLATE = """\
Query: {query}

Identify each criterion, clue, or condition stated in the query and list them as criteria. \
This list will be FIXED for the entire search.

All criteria start with status "not_covered" and empty evidence.\
"""

CRITERIA_UPDATE_USER_TEMPLATE = """\
Query: {query}

Current criteria:
{current_criteria_formatted}

New search query: {subqueries}

New retrieved documents:
{doc_snippets}

Review the criterion list given the new evidence. \
Return only the actions for criteria that changed.\
"""

CRITERIA_UPDATE_STATIC_USER_TEMPLATE = """\
Query: {query}

Current criteria (FIXED — do not add or remove):
{current_criteria_formatted}

New search query: {subqueries}

New retrieved documents:
{doc_snippets}

Review the evidence and update statuses for criteria that changed. \
Return only tick actions for criteria whose status or evidence changed.\
"""


# ======================================================================
# Extraction
# ======================================================================

@dataclass
class CriterionAction:
    """A single action returned by the LLM for the delta-based update."""
    action: str  # "tick" | "add" | "remove"
    name: str
    status: str = ""
    evidence: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["CriterionAction"]:
        action = d.get("action", "").strip().lower()
        name = d.get("name", "").strip()
        if not action or not name:
            return None
        if action not in ("tick", "add", "remove"):
            return None
        return cls(
            action=action,
            name=name,
            status=d.get("status", "not_covered"),
            evidence=d.get("evidence", ""),
        )


@dataclass
class CriterionActionResult:
    """Parsed output from a delta-based criteria coverage update."""
    reasoning: str = ""
    actions: List[CriterionAction] = field(default_factory=list)
    critical_gaps: List[str] = field(default_factory=list)
    minor_gaps: List[str] = field(default_factory=list)


def _strip_code_fences(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def extract_criteria_coverage(raw: str) -> Optional[CriteriaCoverageSummary]:
    """Parse a full criterion list from LLM output (used for initialization).

    Expects a raw JSON object with keys: reasoning, criteria, critical_gaps.

    Returns None if parsing fails entirely.
    """
    stripped = _strip_code_fences(raw)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("CriteriaCoverage: JSON parse failed for init output")
        return None

    if not isinstance(data, dict) or "criteria" not in data:
        logger.warning("CriteriaCoverage: no 'criteria' key found in JSON")
        return None

    reasoning = data.get("reasoning", "")
    criteria_list = data.get("criteria", [])
    if not isinstance(criteria_list, list):
        return None

    criteria = []
    for item in criteria_list:
        if isinstance(item, dict) and item.get("name"):
            criteria.append(Criterion.from_dict(item))

    if not criteria:
        logger.warning("CriteriaCoverage: parsed JSON but no valid criteria found")
        return None

    critical_gaps = data.get("critical_gaps", [])
    if not isinstance(critical_gaps, list):
        critical_gaps = []

    num_covered = sum(1 for a in criteria if a.status == "covered")
    num_partial = sum(1 for a in criteria if a.status == "partial")
    num_not_covered = sum(1 for a in criteria if a.status == "not_covered")

    if not critical_gaps:
        critical_gaps = [
            a.name for a in criteria if a.status == "not_covered"
        ]

    minor_gaps = [
        a.name for a in criteria if a.status == "partial"
    ]

    return CriteriaCoverageSummary(
        criteria=criteria,
        num_covered=num_covered,
        num_partial=num_partial,
        num_not_covered=num_not_covered,
        total=len(criteria),
        critical_gaps=critical_gaps,
        minor_gaps=minor_gaps,
        reasoning=reasoning,
    )


def extract_criterion_actions(raw: str) -> Optional[CriterionActionResult]:
    """Parse a delta-based action list from LLM output (used for updates).

    Expects a raw JSON object with keys: reasoning, actions, critical_gaps.

    Returns None if parsing fails entirely.
    """
    stripped = _strip_code_fences(raw)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("CriteriaCoverage: JSON parse failed for update output")
        return None

    if not isinstance(data, dict) or "actions" not in data:
        logger.warning("CriteriaCoverage: no 'actions' key found in JSON")
        return None

    reasoning = data.get("reasoning", "")
    actions_list = data.get("actions", [])
    if not isinstance(actions_list, list):
        return None

    actions = []
    for item in actions_list:
        if isinstance(item, dict):
            action = CriterionAction.from_dict(item)
            if action is not None:
                actions.append(action)

    critical_gaps = data.get("critical_gaps", [])
    if not isinstance(critical_gaps, list):
        critical_gaps = []

    minor_gaps = data.get("minor_gaps", [])
    if not isinstance(minor_gaps, list):
        minor_gaps = []

    return CriterionActionResult(
        reasoning=reasoning,
        actions=actions,
        critical_gaps=critical_gaps,
        minor_gaps=minor_gaps,
    )


def _resolve_criterion_name(
    name: str, criteria_by_name: Dict[str, "Criterion"],
) -> Optional[str]:
    """Resolve *name* to an existing criterion key, case-insensitively.

    Returns the canonical key if found, else ``None``.
    """
    if name in criteria_by_name:
        return name
    name_lower = name.strip().lower()
    for key in criteria_by_name:
        if key.strip().lower() == name_lower:
            return key
    return None


_STATUS_ALIASES = {
    "partially_covered": "partial",
    "partially covered": "partial",
    "mostly covered": "partial",
    "mostly_covered": "partial",
    "partly covered": "partial",
    "partly_covered": "partial",
    "incomplete": "partial",
    "fully_covered": "covered",
    "fully covered": "covered",
    "complete": "covered",
    "missing": "not_covered",
    "uncovered": "not_covered",
    "not covered": "not_covered",
    "none": "not_covered",
}


def _normalise_status(raw_status: str) -> str:
    """Map an LLM-returned status string to a canonical value."""
    normalised = raw_status.strip().lower()
    if normalised in VALID_STATUSES:
        return normalised
    if normalised in _STATUS_ALIASES:
        logger.debug("CriteriaCoverage: status '%s' mapped to '%s'", raw_status, _STATUS_ALIASES[normalised])
        return _STATUS_ALIASES[normalised]
    logger.warning("CriteriaCoverage: unrecognized status '%s', defaulting to 'not_covered'", raw_status)
    return "not_covered"


def apply_criterion_actions(
    current_criteria: List[Criterion],
    action_result: CriterionActionResult,
    max_criteria: int,
    frozen: bool = False,
) -> tuple:
    """Apply parsed actions to the current criterion list.

    Returns (updated_criteria, added_names, removed_names).
    """
    criteria_by_name = {a.name: a for a in current_criteria}
    added = set()
    removed = set()

    for act in action_result.actions:
        status = _normalise_status(act.status)

        if act.action == "tick":
            resolved = _resolve_criterion_name(act.name, criteria_by_name)
            if resolved is not None:
                if status in VALID_STATUSES:
                    criteria_by_name[resolved].status = status
                if act.evidence:
                    criteria_by_name[resolved].evidence = act.evidence[:_MAX_EVIDENCE_LENGTH]
            else:
                logger.warning("CriteriaCoverage: tick for unknown criterion '%s', skipping", act.name)

        elif act.action == "add":
            if frozen:
                logger.info("CriteriaCoverage: ignoring 'add' action while frozen")
                continue
            resolved = _resolve_criterion_name(act.name, criteria_by_name)
            if resolved is None and len(criteria_by_name) < max_criteria:
                add_status = status if status in VALID_STATUSES else "not_covered"
                new_criterion = Criterion(name=act.name, status=add_status, evidence=act.evidence[:_MAX_EVIDENCE_LENGTH])
                criteria_by_name[act.name] = new_criterion
                added.add(act.name)
            elif resolved is not None:
                logger.info("CriteriaCoverage: 'add' for existing criterion '%s', treating as tick", act.name)
                if status in VALID_STATUSES:
                    criteria_by_name[resolved].status = status
                if act.evidence:
                    criteria_by_name[resolved].evidence = act.evidence[:_MAX_EVIDENCE_LENGTH]

        elif act.action == "remove":
            if frozen:
                logger.info("CriteriaCoverage: ignoring 'remove' action while frozen")
                continue
            resolved = _resolve_criterion_name(act.name, criteria_by_name)
            if resolved is not None:
                del criteria_by_name[resolved]
                removed.add(resolved)

    updated_criteria = list(criteria_by_name.values())
    return updated_criteria, added, removed


# ======================================================================
# Formatting helpers
# ======================================================================

def format_criteria_for_prompt(criteria: List[Criterion]) -> str:
    """Format the current criterion list for inclusion in the user prompt."""
    if not criteria:
        return "(no criteria yet)"
    lines = []
    for i, a in enumerate(criteria, 1):
        ev = f" | evidence: {a.evidence}" if a.evidence else ""
        lines.append(f"  {i}. {a.name} [{a.status}]{ev}")
    return "\n".join(lines)


def format_doc_snippets(docs: List[Dict[str, Any]], top_k: int = 10, max_text_length: int = 200) -> str:
    """Format retrieved docs as compact snippets for the criterion prompt."""
    if not docs:
        return "(no documents retrieved)"

    from utils.text_utils import _doc_title, _doc_text

    parts = []
    for i, doc in enumerate(docs[:top_k], 1):
        title = _doc_title(doc)
        text = _doc_text(doc, max_length=max_text_length)
        if title and text:
            parts.append(f"  [{i}] {title}: {text}")
        elif text:
            parts.append(f"  [{i}] {text}")
        elif title:
            parts.append(f"  [{i}] {title}")
    return "\n".join(parts) if parts else "(no documents retrieved)"


def format_summary_for_log(summary: CriteriaCoverageSummary) -> str:
    """Format a compact one-line summary for verbose logging."""
    return (
        f"criteria_coverage: {summary.num_covered}/{summary.total} covered, "
        f"{summary.num_partial}/{summary.total} partial, "
        f"{summary.num_not_covered}/{summary.total} not_covered"
        + (f" | critical_gaps: [{', '.join(summary.critical_gaps)}]" if summary.critical_gaps else "")
        + (f" | minor_gaps: [{', '.join(summary.minor_gaps)}]" if summary.minor_gaps else "")
        + (f" | new: [{', '.join(summary.new_criteria_this_iter)}]" if summary.new_criteria_this_iter else "")
        + (f" | removed: [{', '.join(summary.removed_criteria_this_iter)}]" if summary.removed_criteria_this_iter else "")
        + (f" | stable_since: iter {summary.stable_since}" if summary.stable_since is not None else "")
        + (" | FROZEN" if summary.frozen else "")
    )


def format_summary_for_controller(summary: CriteriaCoverageSummary) -> str:
    """Format the structured summary string for the controller template."""
    lines = [
        f"Criteria coverage: {summary.num_covered}/{summary.total} covered, "
        f"{summary.num_partial}/{summary.total} partial, "
        f"{summary.num_not_covered}/{summary.total} not covered",
    ]
    if summary.critical_gaps:
        lines.append(f"  Critical gaps (no evidence): [{', '.join(summary.critical_gaps)}]")
    if summary.minor_gaps:
        lines.append(f"  Minor gaps (partial evidence): [{', '.join(summary.minor_gaps)}]")
    if summary.new_criteria_this_iter:
        lines.append(f"  New criteria this iter: [{', '.join(summary.new_criteria_this_iter)}]")
    if summary.stable_since is not None:
        lines.append(f"  Criterion list stable since iter: {summary.stable_since}")
    else:
        lines.append("  Criterion list stable since iter: none (still evolving)")
    if summary.frozen:
        lines.append("  Criterion list: FROZEN")
    return "\n".join(lines)
