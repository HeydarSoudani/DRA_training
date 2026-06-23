"""Text and document utilities for LLM I/O.

All pure-Python with no heavy dependencies:

  Parsing    — extract content from XML tags, boxed answers, and tool calls
  Printing   — format and display agent loop progress to the terminal
  Formatting — convert doc dicts to strings for the LLM prompt
  Citations  — extract which docs were actually cited from an agent result
  Evidence   — prune and summarise reasoning-path evidence for answer gen
"""

import json
import re
from typing import Any, Dict, List, Optional

try:
    import json5 as _json5
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False


# ===========================================================================
# Parsing: extract content from LLM output
# ===========================================================================

def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract the content of the first ``<tag>…</tag>`` block in *text*."""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_all_tag_content(text: str, tag: str) -> List[str]:
    """Extract content from *all* ``<tag>…</tag>`` blocks in *text*."""
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    return [m.strip() for m in pattern.findall(text)]


def extract_last_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract content from the *last* ``<tag>…</tag>`` block in *text*."""
    matches = extract_all_tag_content(text, tag)
    return matches[-1] if matches else None


def get_think(text: str) -> Optional[str]:
    return extract_tag_content(text, "think")


def get_query(text: str) -> Optional[str]:
    return extract_tag_content(text, "search")


def get_answer(text: str) -> Optional[str]:
    return extract_tag_content(text, "answer")


def get_action(text: str) -> Optional[str]:
    return extract_tag_content(text, "action")


def parse_action_call(text: str) -> Optional[tuple]:
    """Parse ``search(query="...")`` or ``finish(answer="...")`` function-call syntax."""
    if not text:
        return None
    text = text.strip()
    for func, param in [("search", "query"), ("finish", "answer")]:
        for quote in ['"', "'"]:
            m = re.match(rf'{func}\s*\(\s*{param}\s*=\s*{quote}(.*){quote}\s*\)', text, re.DOTALL)
            if m:
                return (func, m.group(1).strip())
    for func in ["search", "finish"]:
        for quote in ['"', "'"]:
            m = re.match(rf'{func}\s*\(\s*{quote}(.*){quote}\s*\)', text, re.DOTALL)
            if m:
                return (func, m.group(1).strip())
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the answer from the first ``\boxed{…}`` LaTeX expression."""
    match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_tool_call_xml(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from ``<tool_call>…</tool_call>``."""
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
    """Extract and parse *all* ``<tool_call>…</tool_call>`` blocks."""
    calls = []
    for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
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


def extract_think_and_clean(text: str) -> tuple:
    """Extract ``<think>`` or ``<thought>`` content and return (reasoning, cleaned_text)."""
    for tag in ("think", "thought"):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        m = pattern.search(text)
        if m:
            return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    return "", text


def ensure_closing_tag(text: str, tag: str) -> str:
    """Append ``</tag>`` if ``<tag>`` is present but ``</tag>`` is not."""
    if f"<{tag}>" in text and f"</{tag}>" not in text:
        return text + f"</{tag}>"
    return text


# ===========================================================================
# Printing: verbose terminal output for agent loops
# ===========================================================================

def verbose_print(
    iter_num: int,
    component: str,
    message: str,
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
) -> None:
    """Print a consistently formatted verbose line.

    Format: ``[AgentName] [Iter N] [component]: message``
    """
    prefix = f"[{agent_name}] " if agent_name else ""
    iter_label = f"[Iter {iter_num}.{sub_iter}]" if sub_iter is not None else f"[Iter {iter_num}]"
    if len(message) > 2000:
        message = message[:2000] + "..."
    print(f"{prefix}{iter_label} [{component}]: {message}")


def verbose_print_search_results(
    iter_num: int,
    docs: List[Dict[str, Any]],
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
    top_n: int = 5,
    max_context_len: int = 100,
) -> None:
    """Print top search results in a compact format."""
    prefix = f"[{agent_name}] " if agent_name else ""
    iter_label = f"[Iter {iter_num}.{sub_iter}]" if sub_iter is not None else f"[Iter {iter_num}]"

    if not docs:
        print(f"{prefix}{iter_label} [ret result]: (no docs retrieved)")
        return

    for rank, doc in enumerate(docs[:top_n], 1):
        doc_id = doc.get("doc_id") or doc.get("id", "?")
        title = doc.get("title") or doc.get("metadata", {}).get("title", "") or "Untitled"
        context = doc.get("relevant_text") or doc.get("text") or doc.get("contents", "")
        if len(context) > max_context_len:
            context = context[:max_context_len] + "..."
        print(f"{prefix}{iter_label} [ret result]: [{rank}] {doc_id}, {title}, {context.replace(chr(10), ' ').strip()}")

    remaining = len(docs) - top_n
    if remaining > 0:
        print(f"{prefix}{iter_label} [ret result]: ... and {remaining} more docs")


def verbose_print_controller(
    iter_num: int,
    scores: Dict[str, Any],
    action: str,
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
) -> None:
    """Print controller diagnostics for the current search step."""
    prefix = f"[{agent_name}] " if agent_name else ""
    iter_label = f"[Iter {iter_num}.{sub_iter}]" if sub_iter is not None else f"[Iter {iter_num}]"

    def _fmt(val, fmt=".3f"):
        return f"{val:{fmt}}" if val is not None else "—"

    recall = scores.get("marginal_recall")
    n_new = scores.get("num_new_relevant", 0)
    n_total = scores.get("num_docs_this_step", 0)
    recall_str = f"marginal_recall={_fmt(recall)} ({n_new}/{n_total})" if recall is not None else "marginal_recall=— (no qrels)"
    signals_line = (
        f"{recall_str} | novelty={_fmt(scores.get('doc_novelty'))} | "
        f"consec_sim={_fmt(scores.get('consec_query_sim'))} | orig_sim={_fmt(scores.get('orig_query_sim'))}"
    )

    print(f"{prefix}{iter_label} [controller]: {signals_line}")
    padding = " " * len(f"{prefix}{iter_label} [controller]: ")

    ac = scores.get("criteria_coverage")
    if ac is not None and isinstance(ac, dict) and ac.get("total", 0) > 0:
        ac_total = ac["total"]
        ac_line = (
            f"criteria_coverage={ac.get('num_covered', 0)}/{ac_total} covered, "
            f"{ac.get('num_partial', 0)}/{ac_total} partial, "
            f"{ac.get('num_not_covered', 0)}/{ac_total} not_covered"
        )
        if ac.get("critical_gaps"):
            ac_line += f" | critical_gaps=[{', '.join(ac['critical_gaps'])}]"
        if ac.get("minor_gaps"):
            ac_line += f" | minor_gaps=[{', '.join(ac['minor_gaps'])}]"
        if ac.get("frozen"):
            ac_line += " | FROZEN"
        print(f"{padding}{ac_line}")

    answer_candidates = scores.get("answer_candidates", [])
    if answer_candidates:
        for i, cand in enumerate(answer_candidates):
            label = f"candidate_answer[{i}]" if len(answer_candidates) > 1 else "candidate_answer"
            parts = [f"{label}={cand['candidate'].replace(chr(10), ' ')}"]
            if cand.get("confidence") is not None:
                parts.append(f"confidence={cand['confidence']}%")
            if cand.get("reasoning"):
                r = cand["reasoning"].replace("\n", " ")
                parts.append(f"reasoning={r[:150]}{'...' if len(r) > 150 else ''}")
            print(f"{padding}{' | '.join(parts)}")
    else:
        print(f"{padding}candidates=[] | reasoning=—")

    controller_line = f"controller_action={scores.get('controller_action', action)}"
    if scores.get("controller_reasoning"):
        controller_line += f" | controller_reasoning={scores['controller_reasoning']}"
    print(f"{padding}{controller_line}")


# ===========================================================================
# Documents: internal helpers
# ===========================================================================

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
        cited_ids: Optional[List[str]] = step.get("component_doc_ids")

        # AgentCPM fallback: output.doc_ids
        if not cited_ids:
            output = step.get("output")
            if isinstance(output, dict):
                cited_ids = output.get("doc_ids")

        if not cited_ids:
            continue

        cited_set = set(cited_ids)

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

        for cid in cited_ids:
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                if cid in doc_by_id:
                    docs.append(doc_by_id[cid])
                else:
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


# ===========================================================================
# Documents: formatting — convert doc dicts to strings for the LLM prompt
# ===========================================================================

def passages2string(passages: List[Dict[str, Any]], max_text_length: int = 300) -> str:
    """Convert passage dicts to a numbered, human-readable string.

    Format: ``[1] Title\\ntext``

    Used by: ReSearch, SearchR1, SearchO1, and as the default format.
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
    """Convert docs to ``ContextN: text`` format. Used by: SelfAsk agent."""
    parts = []
    for idx, doc in enumerate(docs[:top_k], 1):
        text = doc.get("relevant_text", "")
        parts.append(f"Context{idx}: {text}")
    return "\n".join(parts)


def format_as_markdown(docs: List[Dict[str, Any]], top_k: int = 5, max_text_length: int = 1000) -> str:
    """Convert docs to markdown format with ``## Web Results`` header.

    Used by: Tongyi-DR agent.
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

    return "## Web Results\n" + "\n\n".join(parts)


def format_as_json(docs: List[Dict[str, Any]], top_k: int = 5, max_text_length: int = 500) -> str:
    """Convert docs to JSON format with docid/title/snippet.

    Used by: GLM and OSS agents.
    """
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


def format_as_snippets(
    docs: List[Dict[str, Any]],
    top_k: int = 5,
    snippet_offset: int = 0,
) -> tuple:
    """Convert docs to DR-Tulu ``<snippet id="SN">…</snippet>`` XML format.

    Used by: DrTulu agent.

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


def build_references_section(result: dict, max_context_length: int = 300) -> str:
    """Build a formatted References section from an agent result's trajectory.

    Skips if the generation already contains a ``## References`` section.

    Returns:
        Formatted references section string (starts with ``\\n\\n## References``),
        or ``""`` when no cited documents are found or references already exist.
    """
    generation = result.get("generation", "")

    if "## References" in generation:
        return ""

    if not generation.strip():
        return ""

    cited_docs = _collect_cited_docs(result)
    if not cited_docs:
        return ""

    # Build doc_id → numeric reference mapping and replace UUID citations
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
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Documents: citations — extract which docs were cited from an agent result
# ===========================================================================

def extract_citations_from_text(text: str) -> set[int]:
    """Extract citation IDs from text (e.g., [1], [22], [6]).

    Args:
        text: Text containing citation markers like [1], [22], etc.

    Returns:
        Set of citation IDs found in the text
    """
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return set(int(m) for m in matches)


def _build_cited_docs_ranked_list(result: dict) -> list:
    """Build a deduplicated, ranked cited-doc list from an agent result.

    Extraction strategies, applied in priority order:

    1. **CPMReport / WebWeaver** — uses ``result["citation_to_doc_id"]``.
    2. **Reasoning agents** — walk trajectory steps for ``component_doc_ids``.
    3. **CPMReport trajectory fallback** — ``step["output"]["doc_ids"]``.
    4. **WebWeaver memory_bank fallback** — ``result["memory_bank"]`` keys.

    Returns:
        List of ``{"doc_id": str, "iter": int}`` dicts, ordered by first citation.
        Empty list when no cited-doc information is available.
    """
    citation_to_doc_id = result.get("citation_to_doc_id")
    if citation_to_doc_id:
        final_report = result.get("final_report") or result.get("generation") or ""
        if final_report:
            cited_ids_in_report = extract_citations_from_text(final_report)
        else:
            cited_ids_in_report = set()

        if not cited_ids_in_report:
            cited_ids_in_report = set(
                int(k) if isinstance(k, str) and k.isdigit() else k
                for k in citation_to_doc_id.keys()
            )

        seen: set = set()
        doc_ids: list = []
        for cit_num, doc_id in sorted(citation_to_doc_id.items(), key=lambda x: x[0]):
            cit_int = int(cit_num) if isinstance(cit_num, str) and cit_num.isdigit() else cit_num
            if cit_int not in cited_ids_in_report:
                continue
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
        return [{"doc_id": d, "iter": 1} for d in doc_ids]

    trajectory = result.get("trajectory") or []
    use_component = any(step.get("component_doc_ids") for step in trajectory)

    seen_ids: set = set()
    ranked: list = []
    for step_idx, step in enumerate(trajectory, 1):
        if use_component:
            for doc_id in step.get("component_doc_ids") or []:
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    ranked.append({"doc_id": doc_id, "iter": step_idx})
        else:
            output = step.get("output")
            if isinstance(output, dict):
                for doc_id in output.get("doc_ids") or []:
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        ranked.append({"doc_id": doc_id, "iter": step_idx})

    if ranked:
        return ranked

    memory_bank = result.get("memory_bank")
    if memory_bank and isinstance(memory_bank, dict):
        return [{"doc_id": doc_id, "iter": 1} for doc_id in memory_bank.keys() if doc_id]

    return []


# ===========================================================================
# Documents: evidence — prune/summarise reasoning-path evidence for answer gen
# ===========================================================================

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
    Older steps include only the reasoning and search query.
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
