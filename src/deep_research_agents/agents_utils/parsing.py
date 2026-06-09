"""Shared parsing utilities for deep research agents.

Consolidates XML tag extraction, boxed answer parsing, and tool-call parsing
functions that were duplicated across multiple agent implementations.
"""

import json
import re
from typing import Any, Dict, List, Optional

try:
    import json5 as _json5
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False


# ---------------------------------------------------------------------------
# Generic XML tag extraction
# ---------------------------------------------------------------------------

def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract the content of the first ``<tag>…</tag>`` block in *text*.

    Args:
        text: Input string that may contain XML-style tags.
        tag:  Tag name (without angle brackets), e.g. ``"think"``.

    Returns:
        Stripped content of the first matching tag, or ``None`` if not found.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_all_tag_content(text: str, tag: str) -> List[str]:
    """Extract content from *all* ``<tag>…</tag>`` blocks in *text*.

    Returns:
        List of stripped content strings, or empty list if none found.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    return [m.strip() for m in pattern.findall(text)]


def extract_last_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract content from the *last* ``<tag>…</tag>`` block in *text*.

    Useful for agents like DrTulu where the last <answer> is the final answer.
    """
    matches = extract_all_tag_content(text, tag)
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Convenience wrappers (backward compatible with general_utils.py)
# ---------------------------------------------------------------------------

def get_think(text: str) -> Optional[str]:
    """Extract the thinking process from the first ``<think>…</think>`` block."""
    return extract_tag_content(text, "think")


def get_query(text: str) -> Optional[str]:
    """Extract a search query from the first ``<search>…</search>`` block."""
    return extract_tag_content(text, "search")


def get_answer(text: str) -> Optional[str]:
    """Extract a final answer from the first ``<answer>…</answer>`` block."""
    return extract_tag_content(text, "answer")


def get_action(text: str) -> Optional[str]:
    """Extract action content from the first ``<action>…</action>`` block."""
    return extract_tag_content(text, "action")


def parse_action_call(text: str) -> Optional[tuple]:
    """Parse a text-based function call like ``search(query="...")`` or ``finish(answer="...")``.

    Handles both ``search(query="...")`` and ``finish(answer="...")`` syntax,
    with or without named parameters, and with single or double quotes.

    Returns:
        ``(action_type, argument_value)`` or ``None`` if not parseable.
    """
    if not text:
        return None
    text = text.strip()

    # Try search(query="...") and finish(answer="...") with named params
    for func, param in [("search", "query"), ("finish", "answer")]:
        for quote in ['"', "'"]:
            pattern = rf'{func}\s*\(\s*{param}\s*=\s*{quote}(.*){quote}\s*\)'
            m = re.match(pattern, text, re.DOTALL)
            if m:
                return (func, m.group(1).strip())

    # Fallback: try without named parameter  e.g. search("...")
    for func in ["search", "finish"]:
        for quote in ['"', "'"]:
            pattern = rf'{func}\s*\(\s*{quote}(.*){quote}\s*\)'
            m = re.match(pattern, text, re.DOTALL)
            if m:
                return (func, m.group(1).strip())

    return None


# ---------------------------------------------------------------------------
# Boxed answer extraction (LaTeX \boxed{})
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the answer from the first ``\boxed{…}`` expression in *text*.

    Used by agents that format their final answers in LaTeX boxed notation
    (e.g. SearchR1, SearchO1, ReSearch).
    """
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Tool call parsing (JSON/XML)
# ---------------------------------------------------------------------------

def parse_tool_call_xml(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from ``<tool_call>…</tool_call>``.

    Tries json5 (if available) then standard json.
    Used by WebWeaver, Tongyi, and similar agents.
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    if _HAS_JSON5:
        try:
            return _json5.loads(raw)
        except Exception:
            pass
    try:
        return json.loads(raw)
    except Exception:
        return None


def parse_tool_calls_xml_list(text: str) -> List[Dict[str, Any]]:
    """Extract and parse *all* ``<tool_call>…</tool_call>`` blocks.

    Returns list of parsed dicts. Unparseable blocks are silently skipped.
    """
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    calls = []
    for raw in matches:
        if _HAS_JSON5:
            try:
                calls.append(_json5.loads(raw))
                continue
            except Exception:
                pass
        try:
            calls.append(json.loads(raw))
        except Exception:
            continue
    return calls


# ---------------------------------------------------------------------------
# Think tag extraction with text cleaning
# ---------------------------------------------------------------------------

def extract_think_and_clean(text: str) -> tuple:
    """Extract ``<think>`` or ``<thought>`` content AND return the cleaned text.

    Returns:
        (reasoning_text, cleaned_text_without_tags)
    """
    for tag in ("think", "thought"):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        m = pattern.search(text)
        if m:
            reasoning = m.group(1).strip()
            cleaned = text[:m.start()] + text[m.end():]
            return reasoning, cleaned.strip()
    return "", text


# ---------------------------------------------------------------------------
# Ensure closing tags (common fixup for server-side stop sequences)
# ---------------------------------------------------------------------------

def ensure_closing_tag(text: str, tag: str) -> str:
    """If *text* contains ``<tag>`` but not ``</tag>``, append the closing tag.

    Many vLLM agents use server-side stop sequences that strip closing tags.
    This helper re-appends them so downstream parsing works correctly.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag in text and close_tag not in text:
        return text + close_tag
    return text
