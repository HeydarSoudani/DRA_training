"""WebWeaver utility functions: memory bank, citation extraction, and text helpers."""

import re
from typing import Dict, List, Set, Tuple


# --- Memory bank ---

def memory_bank_add(bank: Dict[str, dict], entry_id: str, summary: str, evidence: str, url: str = "", title: str = "", **kwargs) -> None:
    """Add or overwrite one evidence entry."""
    bank[entry_id] = {
        "summary": summary,
        "evidence": evidence,
        "url": url,
        "title": title,
        **kwargs,
    }


def memory_bank_retrieve(bank: Dict[str, dict], ids: List[str]) -> Dict[str, str]:
    """Return dict of id -> evidence text for given IDs. Missing IDs are skipped."""
    out = {}
    for eid in ids:
        eid = eid.strip()
        if eid in bank:
            entry = bank[eid]
            out[eid] = entry.get("evidence") or entry.get("summary") or ""
    return out


def memory_bank_format_for_context(bank: Dict[str, dict], max_summary_len: int = 500, ids_only: bool = False) -> str:
    """Format memory bank for LLM context (planner / writer)."""
    lines = []
    for eid, entry in bank.items():
        if ids_only:
            lines.append(f"<{eid}>")
            continue
        summary = (entry.get("summary") or entry.get("evidence") or "")[:max_summary_len]
        lines.append(f"<{eid}>\nSummary: {summary}\n")
    return "\n".join(lines) if lines else "(No evidence yet.)"


# --- Citation extraction from outline ---

CITATION_PATTERN = re.compile(r"<citation>\s*([^<]+?)\s*</citation>")


def extract_citation_ids_from_text(text: str) -> List[str]:
    """Extract citation IDs from text containing <citation>id_1, id_2</citation> or similar."""
    ids: Set[str] = set()
    for m in CITATION_PATTERN.finditer(text):
        inner = m.group(1).strip()
        for part in re.split(r"[\s,;]+", inner):
            part = part.strip()
            if part and (part.startswith("id_") or "_" in part):
                ids.add(part)
    return list(ids)


def extract_citation_ids_from_outline(outline: str) -> List[str]:
    """Extract all unique citation IDs from the full outline (order preserved by first occurrence)."""
    seen: Set[str] = set()
    order: List[str] = []
    for m in CITATION_PATTERN.finditer(outline):
        inner = m.group(1).strip()
        for part in re.split(r"[\s,;]+", inner):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                order.append(part)
    return order


def outline_section_citations(outline: str) -> List[Tuple[str, List[str]]]:
    """
    Split outline into sections (by ## or ###) and return list of (section_heading, citation_ids).
    Useful for the writer to know which IDs to retrieve per section.
    """
    sections: List[Tuple[str, List[str]]] = []
    parts = re.split(r"\n(?=#{1,3}\s)", outline)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        ids = extract_citation_ids_from_text(part)
        heading = part.split("\n")[0][:200] if part else ""
        sections.append((heading, ids))
    if not sections:
        sections.append((outline[:200], extract_citation_ids_from_text(outline)))
    return sections


# --- ID normalisation ---

def normalize_id(raw: str) -> str:
    """Normalise evidence ID for lookup (e.g. id_1, id-1 -> id_1)."""
    s = raw.strip().replace("-", "_")
    if s and not s.startswith("id_"):
        s = f"id_{s}" if s.replace("_", "").isdigit() else s
    return s


def normalize_ids(ids: List[str]) -> List[str]:
    return [normalize_id(i) for i in ids if i]


# --- Report assembly ---

def assemble_report(sections: List[str], sep: str = "\n\n") -> str:
    """Join written sections into the final report."""
    return sep.join(s for s in sections if s and s.strip())


def strip_xml_tags(text: str, keep_content: bool = True) -> str:
    """Remove XML-like tags from text. If keep_content=True, keep text between tags."""
    if keep_content:
        return re.sub(r"<[^>]+>", "", text)
    return re.sub(r"<[^>]*>", "", text)
