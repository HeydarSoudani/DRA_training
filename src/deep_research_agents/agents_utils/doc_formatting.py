"""Document formatting utilities for deep research agents.

Consolidates the various document-to-string conversion functions scattered
across agent implementations into a single module with named strategies.
"""

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers for reference building
# ---------------------------------------------------------------------------

def _doc_id(doc) -> str:
    """Return canonical doc_id from a doc dict or string."""
    if isinstance(doc, str):
        return doc
    return doc.get("doc_id") or doc.get("id") or ""


def _doc_title(doc: Dict[str, Any]) -> str:
    """Extract document title, trying metadata then top-level."""
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            return str(title)
    return doc.get("title") or "Untitled Document"


def _doc_text(doc: Dict[str, Any], max_length: int = 300) -> str:
    """Extract document text snippet, truncated to max_length."""
    text = (
        doc.get("relevant_text")
        or doc.get("text")
        or doc.get("contents")
        or doc.get("snippet")
        or ""
    )
    text = str(text).strip()
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def _collect_cited_docs(result: dict) -> List[Dict[str, Any]]:
    """Collect cited/seen documents from a result dict, deduplicated by doc_id.

    Walks trajectory steps and collects the actual doc objects for documents
    listed in ``component_doc_ids`` (reasoning agents) or
    ``output.doc_ids`` (AgentCPM).  Falls back to ``memory_bank`` entries
    for WebWeaver-style agents.

    Returns:
        List of doc dicts in first-appearance order, deduplicated by doc_id.
    """
    trajectory = result.get("trajectory") or []
    seen_ids: set = set()
    docs: List[Dict[str, Any]] = []

    for step in trajectory:
        # Determine which doc IDs were cited/seen in this step
        cited_ids: Optional[List[str]] = step.get("component_doc_ids")

        # AgentCPM fallback: output.doc_ids
        if not cited_ids:
            output = step.get("output")
            if isinstance(output, dict):
                cited_ids = output.get("doc_ids")

        if not cited_ids:
            continue

        cited_set = set(cited_ids)

        # Build a quick lookup of full doc objects from this step
        step_docs: List[Dict[str, Any]] = []
        if step.get("docs"):
            step_docs = step["docs"]
        elif step.get("all_docs"):
            all_docs = step["all_docs"]
            if all_docs and isinstance(all_docs[0], dict):
                step_docs = all_docs
            else:
                step_docs = [d for sublist in all_docs for d in sublist if isinstance(d, dict)]

        doc_by_id: Dict[str, Dict[str, Any]] = {}
        for d in step_docs:
            did = _doc_id(d)
            if did:
                doc_by_id[did] = d

        # Collect cited docs in the order they appear in cited_ids
        for cid in cited_ids:
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                if cid in doc_by_id:
                    docs.append(doc_by_id[cid])
                else:
                    # Have the ID but not the full doc object — create a minimal entry
                    docs.append({"doc_id": cid})

    # WebWeaver fallback: memory_bank entries
    if not docs:
        memory_bank = result.get("memory_bank")
        if memory_bank and isinstance(memory_bank, dict):
            for doc_id, entry in memory_bank.items():
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append({
                        "doc_id": doc_id,
                        "title": entry.get("title", ""),
                        "relevant_text": entry.get("evidence") or entry.get("summary") or "",
                    })

    return docs


# ---------------------------------------------------------------------------
# Public API: build_references_section
# ---------------------------------------------------------------------------

def build_references_section(result: dict, max_context_length: int = 300) -> str:
    """Build a formatted References section from an agent result's trajectory.

    Collects all cited/seen documents (via ``component_doc_ids`` or
    ``memory_bank``) and formats them as a numbered reference list appended
    below the generation text.

    Skips if the generation already contains a ``## References`` section
    (e.g. AgentCPM handles this internally).

    Args:
        result:             Unified agent result dict.
        max_context_length: Max chars for the context snippet per reference.

    Returns:
        Formatted references section string (starts with ``\\n\\n## References``),
        or ``""`` when no cited documents are found or references already exist.
    """
    generation = result.get("generation", "")

    # Skip if references already present (e.g. AgentCPM)
    if "## References" in generation:
        return ""

    # Skip if no generation text
    if not generation.strip():
        return ""

    cited_docs = _collect_cited_docs(result)
    if not cited_docs:
        return ""

    # Build doc_id → numeric reference mapping and replace UUID citations
    # in the generation body (e.g. [5140886c-a507-...] → [1])
    id_to_ref: Dict[str, str] = {}
    for idx, doc in enumerate(cited_docs, 1):
        did = _doc_id(doc)
        if did:
            id_to_ref[did] = str(idx)

    if id_to_ref:
        updated_gen = result.get("generation", "")
        for did, ref_num in id_to_ref.items():
            updated_gen = updated_gen.replace(f"[{did}]", f"[{ref_num}]")
        result["generation"] = updated_gen

    lines = ["\n\n## References\n"]

    for idx, doc in enumerate(cited_docs, 1):
        did = _doc_id(doc)
        title = _doc_title(doc)
        text = _doc_text(doc, max_length=max_context_length)

        lines.append(f"[{idx}] {title}")
        if text:
            lines.append(f"    {text}")
        if did:
            lines.append(f"    Document ID: {did}")
        lines.append("")  # blank line between entries

    return "\n".join(lines)


def passages2string(passages: List[Dict[str, Any]], max_text_length: int = 300) -> str:
    """Convert passage dicts to a numbered, human-readable string.

    Format: ``[1] Title\\ntext``

    Used by: ReSearch, SearchR1, SearchO1, and as the default format.

    Args:
        passages: List of passage dicts with keys ``doc_id``, ``relevant_text``,
            and optional ``metadata`` (containing ``title``).
        max_text_length: Maximum characters shown per passage before truncation.

    Returns:
        Formatted string, or ``"No documents retrieved."`` when empty.
    """
    if not passages:
        return "No documents retrieved."

    parts = []
    for idx, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata", {})
        title = metadata.get("title", "Untitled Document")
        text = passage.get("relevant_text", "")
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        parts.append(f"[{idx}] {title}\n{text}")

    return "\n\n".join(parts)


def format_as_context(docs: List[Dict[str, Any]], top_k: int = 5) -> str:
    """Convert docs to ``ContextN: text`` format.

    Used by: SelfAsk agent.
    """
    parts = []
    for idx, doc in enumerate(docs[:top_k], 1):
        text = doc.get("relevant_text", "")
        parts.append(f"Context{idx}: {text}")
    return "\n".join(parts)


def format_as_markdown(docs: List[Dict[str, Any]], top_k: int = 5, max_text_length: int = 1000) -> str:
    """Convert docs to markdown format with ``## Web Results`` header.

    Used by: Tongyi-DR agent.

    Args:
        docs:            List of normalised document dicts.
        top_k:           Number of documents to include.
        max_text_length: Maximum characters per document text (avoids flooding
                         the context window with very long documents).
    """
    if not docs:
        return "No results found."

    parts = []
    for i, doc in enumerate(docs[:top_k], 1):
        title = _doc_title(doc)
        text = (
            doc.get("relevant_text")
            or doc.get("text")
            or doc.get("contents")
            or doc.get("snippet")
            or ""
        )
        text = str(text).strip()
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        if title and text:
            parts.append(f"{i}. [{title}]\n{text}")
        elif text:
            parts.append(f"{i}. {text}")
        elif title:
            parts.append(f"{i}. [{title}]\n(no content available)")
        else:
            parts.append(f"{i}. (empty document)")

    header = "## Web Results\n"
    return header + "\n\n".join(parts)


def format_as_json(docs: List[Dict[str, Any]], top_k: int = 5, max_text_length: int = 500) -> str:
    """Convert docs to JSON format with docid/title/snippet.

    Used by: GLM and OSS agents.
    """
    import json

    formatted = []
    for doc in docs[:top_k]:
        text = doc.get(
            "relevant_text", doc.get("text", doc.get("contents", doc.get("snippet", "")))
        )
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        formatted.append({
            "docid": doc.get("doc_id", ""),
            "title": _doc_title(doc),
            "snippet": text,
        })
    return json.dumps(formatted)


def format_as_snippets( docs: List[Dict[str, Any]], top_k: int = 5, snippet_offset: int = 0, ) -> tuple:
    """Convert docs to DR-Tulu ``<snippet id="SN">…</snippet>`` XML format.

    Used by: DrTulu agent.

    Args:
        docs:           Normalised document list from the retriever.
        top_k:          Number of docs to format.
        snippet_offset: Running counter; IDs start at offset+1.

    Returns:
        (xml_string, list_of_snippet_ids_assigned)
    """
    parts: List[str] = []
    ids: List[str] = []

    for i, doc in enumerate(docs[:top_k]):
        sid = f"S{snippet_offset + i + 1}"
        ids.append(sid)

        title = doc.get("title", "")
        url = doc.get("url", doc.get("doc_id", ""))
        body = doc.get("text", doc.get("snippet", "")).strip()
        content = f"Title: {title}\nURL: {url}\n{body}".strip()

        parts.append(f'<snippet id="{sid}">{content}</snippet>')

    return "\n".join(parts), ids
