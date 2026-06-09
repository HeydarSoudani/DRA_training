"""Cited-doc extraction helpers for building ranked doc lists from agent results."""


def _build_cited_docs_ranked_list(result: dict) -> list:
    """Build a deduplicated, ranked cited-doc list from an agent result.

    Extraction strategies, applied in priority order:

    1. **CPMReport / WebWeaver** — uses ``result["citation_to_doc_id"]``.
       CPMReport: maps citation number → doc_id; filtered by ``[N]`` markers
       found in the report text (only docs actually cited).
       WebWeaver: maps sequential index → doc_id for memory bank entries the
       writer retrieved; all entries are used (no ``[N]`` markers in prose).

    2. **Reasoning agents** (ReAct, SearchR1, ReSearch, StepSearch, …) — each
       trajectory step records ``component_doc_ids``: the subset of retrieved
       docs actually formatted into the LLM prompt.  We walk the trajectory in
       step order and collect unique doc_ids by first occurrence.

    3. **CPMReport trajectory fallback** — if ``citation_to_doc_id`` is absent,
       CPM trajectory steps store visible doc IDs at ``step["output"]["doc_ids"]``.

    4. **WebWeaver memory_bank fallback** — if no ``component_doc_ids`` are found,
       falls back to ``result["memory_bank"]`` keys (insertion-ordered cited docs).

    Returns:
        List of ``{"doc_id": str, "iter": int}`` dicts, ordered by first citation.
        ``iter`` is the 1-based trajectory step where the doc first appeared.
        Empty list when no cited-doc information is available.
    """
    # ── citation_to_doc_id path (CPMReport + WebWeaver) ────────────────
    citation_to_doc_id = result.get("citation_to_doc_id")
    if citation_to_doc_id:
        final_report = result.get("final_report") or result.get("generation") or ""
        if final_report:
            from agentic_retrieval_research.evaluation.citation_evaluator import extract_citations_from_text
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

    # ── Walk trajectory steps for cited doc IDs ──────────────────────────────
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

    # ── WebWeaver fallback: memory_bank keys ─────────────────────────────────
    memory_bank = result.get("memory_bank")
    if memory_bank and isinstance(memory_bank, dict):
        return [{"doc_id": doc_id, "iter": 1} for doc_id in memory_bank.keys() if doc_id]

    return []
