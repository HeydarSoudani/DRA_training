"""Shared evidence-summary utilities.

Canonical implementations of ``reduce_reasoning_path`` and
``build_evidence_summary``, used by answer-candidate generation,
force-answer prompts, and windowed-prompt rebuilding.
"""

from typing import Any, Dict, List

from utils.doc_formatting import passages2string


def reduce_reasoning_path(
    reasoning_path: List[Dict[str, Any]],
    keep_last: int = 3,
    middle_strategy: str = "strip_docs",
) -> List[Dict[str, Any]]:
    """Prune reasoning_path for context-limited answer generation.

    Returns a new list where recent search steps retain full docs and
    older search steps have docs removed or are dropped entirely.

    Args:
        keep_last: Number of most-recent search steps to keep with full docs.
        middle_strategy: ``"strip_docs"`` removes doc content from older
            steps but keeps think/query; ``"drop"`` removes them entirely.
    """
    search_indices = [
        i for i, step in enumerate(reasoning_path)
        if step.get("docs") or step.get("all_docs")
    ]

    if len(search_indices) <= keep_last:
        return list(reasoning_path)

    strip_indices = set(search_indices[:-keep_last])

    result: List[Dict[str, Any]] = []
    for i, step in enumerate(reasoning_path):
        if i in strip_indices:
            if middle_strategy == "drop":
                continue
            stripped = {k: v for k, v in step.items() if k not in ("docs", "all_docs")}
            stripped["docs"] = []
            stripped["_docs_stripped"] = True
            result.append(stripped)
        else:
            result.append(step)

    return result


def build_evidence_summary(
    reasoning_path: List[Dict[str, Any]],
    seen_top_k: int = 5,
    keep_last: int = 3,
) -> str:
    """Build windowed evidence summary from reasoning_path.

    Recent search steps (last ``keep_last``) include full document text.
    Older steps include only the reasoning and search query, keeping the
    prompt compact while preserving richer evidence than a flat summary.
    """
    pruned = reduce_reasoning_path(reasoning_path, keep_last=keep_last)
    parts: List[str] = []
    for step in pruned:
        action = step.get("action_type", "")
        has_search = bool(step.get("search_query") or step.get("docs"))
        if not has_search and action not in ("search", "critical_search"):
            continue
        sq = step.get("search_query", step.get("query", ""))
        think = step.get("think", "")
        is_stripped = step.get("_docs_stripped", False)
        if is_stripped:
            entry = f"- Search: {sq}"
            if think:
                entry += f"\n  Reasoning: {think}"
        else:
            docs = step.get("docs", [])
            doc_text = passages2string(docs[:seen_top_k])
            entry = f"- Search: {sq}"
            if think:
                entry += f"\n  Reasoning: {think}"
            entry += f"\n  Results:\n  {doc_text}"
        parts.append(entry)
    return "\n\n".join(parts) if parts else "(no evidence collected)"
