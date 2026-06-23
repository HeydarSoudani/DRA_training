"""WebWeaver agent: adapts the WebWeaver planner-writer algorithm to the pipeline.

Algorithm (WebWeaver ICLR 2026 – Section 2-3):
  Planner loop : [search | write_outline | terminate]*
  Writer loop  : [retrieve | write | terminate]*

Both loops use ReAct-style thought-action-observation turns with the pipeline's
LiteLLMClient for generation and local retriever for retrieval.

LLM calls  : pipeline LiteLLMClient  (generator.complete)
Retrieval  : pipeline local retriever  (retriever.retrieve)
"""

import logging
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ── Pipeline imports ──────────────────────────────────────────────────────────
from utils.llm_client import LiteLLMClient

from .base_agent import BasicAgent
from controller_component import TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult
from utils.config import InferenceConfig
from utils.text_utils import extract_tag_content as _extract_tag, parse_tool_call_xml
from deep_research_agents.prompts.webweaver.user_prompts import (
    PLANNER_USER_TEMPLATE,
    WRITER_USER_TEMPLATE,
)
from searcher_component.fusion import interleaving_fusion
from controller_component.prompts.answer_prompts import (
    CANDIDATE_GENERATION_INSTRUCTION,
    AnswerCandidateOutput,
    extract_answer_candidates,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "webweaver"
PLANNER_SYSTEM: str = (_PROMPTS_DIR / "planner_system.txt").read_text()
WRITER_SYSTEM: str  = (_PROMPTS_DIR / "writer_system.txt").read_text()


# ── Memory bank helpers (inlined from webweaver_utils) ───────────────────────

_CITATION_PATTERN = re.compile(r"<citation>\s*([^<]+?)\s*</citation>")


def memory_bank_add(bank: Dict[str, dict], entry_id: str, summary: str, evidence: str, url: str = "", title: str = "", **kwargs) -> None:
    """Add or overwrite one evidence entry."""
    bank[entry_id] = {"summary": summary, "evidence": evidence, "url": url, "title": title, **kwargs}


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


def extract_citation_ids_from_text(text: str) -> List[str]:
    """Extract citation IDs from text containing <citation>id_1, id_2</citation>."""
    from typing import Set
    ids: Set[str] = set()
    for m in _CITATION_PATTERN.finditer(text):
        inner = m.group(1).strip()
        for part in re.split(r"[\s,;]+", inner):
            part = part.strip()
            if part and (part.startswith("id_") or "_" in part):
                ids.add(part)
    return list(ids)


def extract_citation_ids_from_outline(outline: str) -> List[str]:
    """Extract all unique citation IDs from the full outline (order preserved by first occurrence)."""
    from typing import Set
    seen: Set[str] = set()
    order: List[str] = []
    for m in _CITATION_PATTERN.finditer(outline):
        inner = m.group(1).strip()
        for part in re.split(r"[\s,;]+", inner):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                order.append(part)
    return order


def outline_section_citations(outline: str) -> List[Tuple[str, List[str]]]:
    """Split outline into sections and return list of (section_heading, citation_ids)."""
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


def normalize_id(raw: str) -> str:
    """Normalise evidence ID for lookup (e.g. id_1, id-1 -> id_1)."""
    s = raw.strip().replace("-", "_")
    if s and not s.startswith("id_"):
        s = f"id_{s}" if s.replace("_", "").isdigit() else s
    return s


def normalize_ids(ids: List[str]) -> List[str]:
    return [normalize_id(i) for i in ids if i]


def assemble_report(sections: List[str], sep: str = "\n\n") -> str:
    """Join written sections into the final report."""
    return sep.join(s for s in sections if s and s.strip())


def strip_xml_tags(text: str, keep_content: bool = True) -> str:
    """Remove XML-like tags from text."""
    if keep_content:
        return re.sub(r"<[^>]+>", "", text)
    return re.sub(r"<[^>]*>", "", text)


# ── Output parsing helpers ────────────────────────────────────────────────────

def _extract_thought(content: str) -> str:
    """Extract content inside <think>...</think>."""
    return _extract_tag(content, "think") or ""


def _parse_tool_call(content: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from <tool_call>...</tool_call>."""
    return parse_tool_call_xml(content)


def _extract_write_outline(content: str) -> Optional[str]:
    """Extract outline from <write_outline>...</write_outline>."""
    return _extract_tag(content, "write_outline")


def _extract_write_block(content: str) -> Optional[str]:
    """Extract section content from <write>...</write>."""
    return _extract_tag(content, "write")


def _has_terminate(content: str) -> bool:
    return "<terminate>" in content


# ── Planner / writer state ────────────────────────────────────────────────────

@dataclass
class _PlannerState:
    question: str
    memory_bank: Dict[str, Dict[str, str]] = field(default_factory=dict)
    outline: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    terminated: bool = False


@dataclass
class _WriterState:
    question: str
    outline: str
    memory_bank: Dict[str, Dict[str, str]]
    report_parts: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    terminated: bool = False
    retrieved_entry_ids: List[str] = field(default_factory=list)


# ── Agent ─────────────────────────────────────────────────────────────────────

class WebWeaver_Agent(BasicAgent):
    """WebWeaver planner-writer deep research agent.

    Constructor signature matches other reasoning agents so that
    run_pipeline.py can instantiate it transparently::

        agent = WebWeaver_Agent(
            llm_client=llm_client,
            retriever=retriever,
            max_iteration=100,
            seen_top_k=5,
        )

    ``max_iteration`` is split evenly between the planner and writer loops by
    default (planner_max_steps = max_iteration, writer_max_steps = max_iteration * 2).
    """

    def __init__(self, llm_client: LiteLLMClient, retriever: Optional[Any] = None, max_iteration: int = 100, seen_top_k: int = 5, planner_max_steps: Optional[int] = None, writer_max_steps: Optional[int] = None, verbose: bool = True) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.planner_max_steps: int = planner_max_steps if planner_max_steps is not None else max_iteration
        self.writer_max_steps: int = writer_max_steps if writer_max_steps is not None else max_iteration * 2
        self._entry_counter: int = 0
        self.verbose: bool = verbose

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            system_prompt=PLANNER_SYSTEM,
            format_instructions="",
        )

    AGENT_NAME = "WebWeaver"

    def _notify_progress(self, stage: str = None, iteration: int = None) -> None:
        """Bump the step counter and notify the progress bar with the current phase."""
        self._search_iter = getattr(self, "_search_iter", 0) + 1
        cb = getattr(self, "_status_callback", None)
        if cb is not None:
            phase = stage or getattr(self, "_current_phase", "search")
            cb(phase, iteration if iteration is not None else self._search_iter)

    # ── run_single override ──────────────────────────────────────────────────

    def run_single(self, query_id: str, query_text: str, temperature: float = 0.7, status_callback=None) -> Any:
        """Override to unpack the 3-tuple from inference and include citation_to_doc_id."""

        self._status_callback = status_callback
        self._search_iter     = 0
        # Reset trajectory tracker for this query (loads per-query qrels)
        self._reset_tracker(query_id=query_id)
        _meter = self._token_meter()
        _tok_start = _meter.snapshot() if _meter is not None else None
        if _meter is not None:
            _meter.since_last_step()
        try:
            reasoning_path, prediction, citation_to_doc_id = self.inference(query_text, generation_temp=temperature)

            num_searches = sum(
                1 for step in reasoning_path
                if ("docs" in step and step["docs"]) or ("all_docs" in step and step["all_docs"])
            )

            result = {
                "query": query_text,
                "generation": str(prediction) if prediction else "",
                "num_steps": len(reasoning_path),
                "num_searches": num_searches,
                "trajectory": reasoning_path,
                "citation_to_doc_id": citation_to_doc_id,
            }
            # Attach controller stats when available
            self._attach_controller_stats(result)

            # Attach per-query token usage when a meter is available.
            if _meter is not None and _tok_start is not None:
                _tok_end = _meter.snapshot()
                result["token_usage"] = {
                    "input_tokens": _tok_end["input_tokens"] - _tok_start["input_tokens"],
                    "output_tokens": _tok_end["output_tokens"] - _tok_start["output_tokens"],
                    "total_tokens": _tok_end["total_tokens"] - _tok_start["total_tokens"],
                    "num_calls": _tok_end["num_calls"] - _tok_start["num_calls"],
                }

            logger.info(f"  ✓ {num_searches} searches, {len(reasoning_path)} steps, {len(citation_to_doc_id)} cited docs")
            return result

        except Exception as e:
            logger.error(f"  ✗ Error processing {query_id}: {e}")
            traceback.print_exc()
            return None
        finally:
            self._status_callback = None
            self._search_iter     = 0

    # ── Answer candidate (planner-consistent) ────────────────────────────────

    def generate_answer_candidate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
        """Generate answer candidate using the same context the planner sees."""
        memory_bank = getattr(self, "_ac_memory_bank", None)
        planner_history = getattr(self, "_ac_planner_history", None)
        if memory_bank is None or planner_history is None:
            return [AnswerCandidateOutput(
                candidate="no candidate",
                reasoning="no planner state available",
            )]

        memory_text = memory_bank_format_for_context(memory_bank)
        history_text = "\n\n".join(
            f"Thought: {h.get('thought', '')}\n"
            f"Action: {h.get('action', '')}\n"
            f"Observation: {h.get('observation', '')}"
            for h in planner_history[-10:]
        ) or "(No previous steps yet.)"

        cfg = self.inference_config
        user_content = (
            f"## Open-ended research question\n{original_query}\n\n"
            f"## Memory bank (evidence so far)\n{memory_text}\n\n"
            f"## Previous steps (thought, action, observation)\n{history_text}\n\n"
            f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
            f"{cfg.format_instructions}"
        )

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": user_content},
        ]

        llm = self.get_answer_candidate_llm()
        if llm is None:
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning="no LLM client available",
            )]

        try:
            raw = llm.complete(
                messages,
                strip_think=False,
                return_reasoning_fallback=True,
                max_tokens=cfg.max_output_tokens or 1024,
            )
        except Exception:
            logger.warning("WebWeaver answer candidate LLM call failed", exc_info=True)
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning="LLM call failed",
            )]

        if not raw or not raw.strip():
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning="LLM returned empty response",
            )]

        is_reasoning_fallback = raw.startswith("[reasoning_fallback]")
        if is_reasoning_fallback:
            raw = raw[len("[reasoning_fallback]"):]

        candidates, format_matched = extract_answer_candidates(raw)

        if is_reasoning_fallback and not format_matched:
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning=raw.strip(),
            )]

        if not candidates and not format_matched:
            logger.warning(
                "WebWeaver answer candidate: no candidates extracted. "
                "Raw (first 300 chars): %s", raw[:300],
            )
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning=raw.strip(),
            )]

        return candidates

    # ── Planner step ─────────────────────────────────────────────────────────

    def _run_planner_step(self, state: _PlannerState, search_fn, temperature: float) -> _PlannerState:
        memory_text = memory_bank_format_for_context(state.memory_bank)
        history_text = "\n\n".join(
            f"Thought: {h.get('thought', '')}\nAction: {h.get('action', '')}\nObservation: {h.get('observation', '')}"
            for h in state.history[-10:]
        )
        user_content = PLANNER_USER_TEMPLATE.format(
            question=state.question,
            memory_bank=memory_text,
            history=history_text or "(No previous steps yet.)",
        )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        raw_response = self.generator.complete(messages, temperature=temperature, strip_think=False)

        step_num = len(state.history)

        thought       = _extract_thought(raw_response)
        response      = raw_response.split("</think>")[-1].strip() if "</think>" in raw_response else raw_response
        outline_block = _extract_write_outline(response)
        tool_call     = _parse_tool_call(response)
        term          = _has_terminate(response)

        new_state = _PlannerState(
            question=state.question,
            memory_bank=dict(state.memory_bank),
            outline=state.outline,
            history=list(state.history),
            terminated=term,
        )

        if term:
            if thought:
                self._vprint(step_num, "plan-think", thought[:200].replace('\n', ' '))
            self._vprint(step_num, "plan-terminate", f"outline {len(new_state.outline or '')} chars")
            new_state.history.append({"thought": thought, "action": "<terminate>", "observation": ""})
            return new_state

        if outline_block:
            if thought:
                self._vprint(step_num, "plan-think", thought[:200].replace('\n', ' '))
            sections = [line.strip().lstrip('#').strip() for line in outline_block.splitlines() if line.strip().startswith('#')]
            section_summary = ', '.join(s[:40] for s in sections[:8])
            if len(sections) > 8:
                section_summary += ', ...'
            self._vprint(step_num, "plan-outline", f"({len(outline_block)} chars, {len(sections)} sections) [{section_summary}]")
            new_state.outline = outline_block
            new_state.history.append({
                "thought":     thought,
                "action":      f"<write_outline>\n{outline_block[:200]}...\n</write_outline>",
                "observation": "(Outline updated.)",
            })
            return new_state

        if tool_call and tool_call.get("name") == "search":
            args    = tool_call.get("arguments") or {}
            queries = args.get("query") or args.get("queries") or []
            goal    = args.get("goal") or "Relevant information for the research question."
            if isinstance(queries, str):
                queries = [queries]
            if thought:
                self._vprint(step_num, "plan-think", thought[:200].replace('\n', ' '))
            self._vprint(step_num, "plan-search", f"[{', '.join(queries)}]")
            if not queries:
                obs = "Error: search requires a non-empty 'query' array."
                self._vprint(step_num, "plan-error", "Empty query list")
            else:
                try:
                    new_entries, tracker_result = search_fn(queries, goal, thought=thought)
                    for eid, entry in new_entries.items():
                        new_state.memory_bank[eid] = entry
                    obs = f"Added {len(new_entries)} evidence entries to memory bank."
                    if isinstance(tracker_result, TrackerEarlyStopResult):
                        new_state.terminated = True
                    elif isinstance(tracker_result, TrackerCriticalThinkResult):
                        obs += (
                            f"\n[Critical Redirect — {tracker_result.critical_search_query}]\n"
                            f"{tracker_result.critical_observation}"
                        )
                    elif tracker_result:
                        obs = tracker_result
                    self._vprint(step_num, "plan-ret", f"{len(new_entries)} entries, memory bank: {len(new_state.memory_bank)}")
                except Exception as e:
                    obs = f"Search error: {e}"
                    self._vprint(step_num, "plan-error", str(e))
            new_state.history.append({
                "thought":     thought,
                "action":      f"<tool_call> search query={queries} goal={goal}",
                "observation": obs,
            })
            return new_state

        # Unknown or malformed action
        self._vprint(step_num, "plan-error", "Malformed planner action, requesting retry")
        new_state.history.append({
            "thought":     thought,
            "action":      response[:500],
            "observation": "Error: expected one of search, write_outline, or <terminate>. Retry.",
        })
        return new_state

    # ── Writer step ───────────────────────────────────────────────────────────

    def _run_writer_step(self, state: _WriterState, temperature: float) -> Tuple["_WriterState", Optional[str]]:
        report_so_far    = "\n\n".join(state.report_parts)
        memory_summary   = memory_bank_format_for_context(state.memory_bank)

        # Build writer history (compact, like planner) so the LLM can track progress.
        # Mask retrieve observations that were already consumed by a write (paper's pruning).
        history_lines = []
        pending_retrieve_obs = None
        for h in state.history[-15:]:
            action = h.get("action", "")
            thought_snippet = (h.get("thought") or "")[:150].replace('\n', ' ')
            if "retrieve" in action:
                pending_retrieve_obs = h.get("observation", "")
                history_lines.append(f"- Thought: {thought_snippet}\n  Action: {action}\n  Observation: (evidence retrieved, see below if not yet used)")
            elif "<write>" in action:
                # After a write, the retrieve observation is consumed → prune it
                pending_retrieve_obs = None
                history_lines.append(f"- Thought: {thought_snippet}\n  Action: wrote section\n  Observation: (Section appended to report.)")
            elif "<terminate>" in action:
                history_lines.append(f"- Action: <terminate>")
            else:
                history_lines.append(f"- Thought: {thought_snippet}\n  Action: {action}\n  Observation: {h.get('observation', '')[:200]}")
        writer_history = "\n".join(history_lines) if history_lines else "(No previous writer steps.)"

        # Find the most recent unconsumed retrieve observation (not just the last step).
        # This handles gaps from malformed actions between retrieve and write.
        last_obs_block = ""
        if pending_retrieve_obs:
            last_obs_block = "\n\n## Evidence retrieved for current section (use this to write)\n" + pending_retrieve_obs

        user_content = WRITER_USER_TEMPLATE.format(
            question=state.question,
            outline=state.outline,
            memory_bank_summary=memory_summary,
            report_so_far=report_so_far or "(Not started.)",
            writer_history=writer_history,
            last_observation_block=last_obs_block,
        )
        messages = [
            {"role": "system", "content": WRITER_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        raw_response = self.generator.complete(messages, temperature=temperature, strip_think=False)

        step_num = len(state.history)

        thought     = _extract_thought(raw_response)
        response    = raw_response.split("</think>")[-1].strip() if "</think>" in raw_response else raw_response
        tool_call   = _parse_tool_call(response)
        write_block = _extract_write_block(response)
        term        = _has_terminate(response)

        new_state = _WriterState(
            question=state.question,
            outline=state.outline,
            memory_bank=state.memory_bank,
            report_parts=list(state.report_parts),
            history=list(state.history),
            terminated=term,
            retrieved_entry_ids=list(state.retrieved_entry_ids),
        )

        if tool_call and tool_call.get("name") == "retrieve":
            args = tool_call.get("arguments") or {}
            ids  = args.get("retrieve_id") or args.get("url_id") or []
            if isinstance(ids, str):
                ids = [ids]
            if thought:
                self._vprint(step_num, "write-think", thought[:200].replace('\n', ' '))
            self._vprint(step_num, "write-retrieve", f"IDs: {ids}")
            try:
                evidence = memory_bank_retrieve(state.memory_bank, ids)
                obs = "\n\n".join(f"<{k}>\n{v}" for k, v in evidence.items())
                self._vprint(step_num, "write-ret result", f"{len(evidence)} entries ({len(obs)} chars)")
            except Exception as e:
                obs = f"Retrieve error: {e}"
                self._vprint(step_num, "write-error", str(e))
            # Track which memory bank entries the writer actually retrieved (for cited doc eval)
            new_state.retrieved_entry_ids.extend(ids)
            new_state.history.append({
                "thought":     thought,
                "action":      f"retrieve {ids}",
                "observation": obs[:2000] + "..." if len(obs) > 2000 else obs,
            })
            return new_state, None

        if write_block:
            new_state.report_parts.append(write_block)
            if thought:
                self._vprint(step_num, "write-think", thought[:200].replace('\n', ' '))
            self._vprint(step_num, "write-section", f"{len(write_block)} chars, {len(new_state.report_parts)} part(s) total")
            new_state.history.append({
                "thought":     thought,
                "action":      "<write> ... </write>",
                "observation": "(Section appended.)",
            })
            return new_state, write_block

        if term:
            self._vprint(step_num, "write-terminate", f"{len(new_state.report_parts)} sections written")
            new_state.history.append({"thought": thought, "action": "<terminate>", "observation": ""})
            return new_state, None

        # Unknown or malformed action
        self._vprint(step_num, "write-error", "Malformed writer action, requesting retry")
        new_state.history.append({
            "thought":     thought,
            "action":      response[:500],
            "observation": "Error: expected retrieve, write, or <terminate>. Retry.",
        })
        return new_state, None

    # ── Verbose helpers ────────────────────────────────────────────────────────

    def _dump_planner_results(self, outline: str, memory_bank: Dict[str, Dict[str, str]]) -> None:
        """Print a summary of the planner phase results (outline + memory bank)."""
        if not self.verbose or not outline:
            return
        print(f"\n{'='*80}")
        print(f"[{self.AGENT_NAME}] PHASE-1 COMPLETE — OUTLINE")
        print(f"{'='*80}")
        print(outline)
        print(f"{'='*80}")

        n_passages = len(memory_bank)
        print(f"\n[{self.AGENT_NAME}] MEMORY BANK — {n_passages} passage(s)")
        print(f"{'-'*80}")
        PREVIEW_MAX = 5
        TRUNC_LEN   = 200
        for i, (eid, entry) in enumerate(memory_bank.items()):
            if i >= PREVIEW_MAX:
                print(f"  ... and {n_passages - PREVIEW_MAX} more passage(s)")
                break
            title   = entry.get("title", "")
            snippet = (entry.get("evidence") or entry.get("summary") or "")[:TRUNC_LEN]
            if len(entry.get("evidence", "")) > TRUNC_LEN:
                snippet += "..."
            print(f"  [{eid}] {title}")
            print(f"          {snippet}")
        print(f"{'-'*80}\n")

    # ── Main inference ────────────────────────────────────────────────────────

    def inference(self, question: str, generation_temp: float = 0.7) -> Tuple[List[Dict[str, Any]], str, Dict[int, str]]:
        """Run the full WebWeaver planner→writer pipeline for one query.

        Returns:
            reasoning_path : list of per-step dicts compatible with the pipeline
                             evaluators.  Search steps carry:
                               phase             – "planner"
                               search_query      – list of query strings
                               docs              – all retrieved docs (for TREC)
                               component_doc_ids – first seen_top_k doc ids
            prediction     : final report text (concatenated writer sections).
        """
        self._entry_counter = 0
        reasoning_path: List[Dict[str, Any]] = []

        self._print(f"Query: {question}")

        # ── search_fn: bridges planner's tool calls → pipeline retriever ─────
        def search_fn(queries: List[str], goal: str, thought: str = "") -> Tuple[Dict[str, Dict], Optional[str]]:
            new_entries: Dict[str, Dict] = {}
            n_queries = len(queries)
            _first_tracker_action = None
            _stop_tracking = False

            for q_idx, query in enumerate(queries):
                sub_iter = q_idx if n_queries > 1 else None
                self._vprint(self._search_step + 1, "search", query, sub_iter=sub_iter)

                if self.search_tool is not None:
                    docs = self.search_tool.execute(
                        query,
                        original_query=question,
                        reasoning=thought if thought else None,
                    )
                else:
                    docs = self.retrieve_documents(query, original_query=question)

                for doc in docs[:self.seen_top_k]:
                    entry_id = f"id_{self._entry_counter}"
                    self._entry_counter += 1
                    text = (
                        doc.get("relevant_text")
                        or doc.get("text")
                        or doc.get("contents")
                        or doc.get("snippet")
                        or ""
                    ).strip()
                    title = doc.get("title") or (doc.get("metadata", {}).get("title") if isinstance(doc.get("metadata"), dict) else None) or ""
                    new_entries[entry_id] = {
                        "summary":  text[:500],
                        "evidence": text,
                        "url":      doc.get("doc_id", ""),
                        "title":    title,
                    }

                self._vprint_docs(self._search_step + 1, docs[:self.seen_top_k], sub_iter=sub_iter)

                reasoning_path.append({
                    "action_type":       "search",
                    "phase":             "planner",
                    "search_query":      query,
                    "think":             thought if q_idx == 0 else "",
                    "docs":              docs,
                    "all_docs":          docs,
                    "component_doc_ids": [d.get("doc_id", "") for d in docs[:self.seen_top_k]],
                    "sub_iter":          sub_iter,
                    "tracker_observation": None,
                })

                if not _stop_tracking:
                    self._ac_memory_bank = dict(self._ac_memory_bank_base)
                    self._ac_memory_bank.update(new_entries)
                    _result, _stop_tracking = self._track_query(
                        query, docs[:self.seen_top_k],
                        question, thought, [], reasoning_path,
                    )
                    if _result is not None:
                        _first_tracker_action = _result

            tracker_result = _first_tracker_action

            if isinstance(tracker_result, TrackerCriticalThinkDeferred):
                tracker_result = self._execute_deferred_critical_search(
                    tracker_result, question,
                    trajectory=[], reasoning_path=reasoning_path,
                )
                self._search_step += 1

            if isinstance(tracker_result, TrackerCriticalThinkResult):
                for doc in tracker_result.critical_docs[:self.seen_top_k]:
                    entry_id = doc.get("doc_id") or doc.get("id") or ""
                    if not entry_id:
                        continue
                    text = (
                        doc.get("text") or doc.get("contents") or doc.get("snippet") or ""
                    ).strip()
                    title = doc.get("title") or ""
                    new_entries[entry_id] = {
                        "summary":  text[:500],
                        "evidence": text,
                        "url":      entry_id,
                        "title":    title,
                    }
                reasoning_path.append({
                    "action_type":       "critical_search",
                    "phase":             "planner",
                    "search_query":      [tracker_result.critical_search_query],
                    "think":             tracker_result.critical_think,
                    "docs":              tracker_result.critical_docs,
                    "component_doc_ids": [
                        d.get("doc_id", "") for d in tracker_result.critical_docs[:self.seen_top_k]
                    ],
                    "is_critical_think":       True,
                })

            return new_entries, tracker_result

        # ── Planner loop ──────────────────────────────────────────────────────
        self._current_phase = "plan"
        self._print(f"Planner phase (max {self.planner_max_steps} steps)")
        planner_state = _PlannerState(question=question)
        for _ in range(self.planner_max_steps):
            self._notify_progress()
            self._ac_planner_history = list(planner_state.history)
            self._ac_memory_bank_base = dict(planner_state.memory_bank)
            prev_outline = planner_state.outline
            planner_state = self._run_planner_step(planner_state, search_fn, generation_temp)
            # Record write_outline step (search steps are already recorded inside search_fn)
            if planner_state.outline and planner_state.outline != prev_outline:
                last_thought = planner_state.history[-1].get("thought", "") if planner_state.history else ""
                reasoning_path.append({
                    "action_type": "write_outline",
                    "phase":       "planner",
                    "think":       last_thought,
                    "outline":     planner_state.outline,
                })
            if planner_state.terminated:
                break

        self._ac_planner_history = None
        self._ac_memory_bank = None
        self._ac_memory_bank_base = None

        outline     = planner_state.outline or ""
        memory_bank = planner_state.memory_bank
        self._print(f"Planner done: {len(planner_state.history)} steps, {len(memory_bank)} memory entries, outline {len(outline)} chars")

        self._dump_planner_results(outline, memory_bank)

        # ── Writer loop ───────────────────────────────────────────────────────
        self._current_phase = "write"
        self._print(f"Writer phase (max {self.writer_max_steps} steps)")
        writer_state = _WriterState(question=question, outline=outline, memory_bank=memory_bank)
        for _ in range(self.writer_max_steps):
            self._notify_progress()
            prev_history_len = len(writer_state.history)
            writer_state, written = self._run_writer_step(writer_state, generation_temp)
            # Record writer step to reasoning_path
            if len(writer_state.history) > prev_history_len:
                last = writer_state.history[-1]
                step_action = last.get("action", "")
                if "retrieve" in step_action:
                    reasoning_path.append({
                        "action_type": "retrieve",
                        "phase":       "writer",
                        "think":       last.get("thought", ""),
                    })
                elif written is not None:
                    reasoning_path.append({
                        "action_type": "write",
                        "phase":       "writer",
                        "think":       last.get("thought", ""),
                    })
                elif writer_state.terminated:
                    reasoning_path.append({
                        "action_type": "terminate",
                        "phase":       "writer",
                        "think":       last.get("thought", ""),
                    })
            if writer_state.terminated:
                break

        report = "\n\n".join(writer_state.report_parts)
        self._print(f"Writer done: {len(writer_state.history)} steps, {len(writer_state.report_parts)} section(s), {len(report)} chars total")

        # Build citation_to_doc_id from writer's retrieved memory bank entries.
        # Maps sequential citation index → corpus doc_id, mirroring AgentCPM's
        # citation tracking for consistent cited-doc evaluation.
        citation_to_doc_id: Dict[int, str] = {}
        seen_doc_ids: set = set()
        cit_counter = 0
        for eid in writer_state.retrieved_entry_ids:
            entry = memory_bank.get(eid)
            if not entry:
                continue
            doc_id = entry.get("url", "")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                cit_counter += 1
                citation_to_doc_id[cit_counter] = doc_id
        self._print(f"Cited docs: {len(citation_to_doc_id)} unique docs from {len(writer_state.retrieved_entry_ids)} writer retrieve calls")

        self._print(f"Done: {len(reasoning_path)} total reasoning steps")
        return reasoning_path, report, citation_to_doc_id
