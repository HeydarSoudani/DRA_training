"""DR-Tulu agent: inference implementation of the DR-Tulu deep research algorithm.

Adapts the DR-Tulu algorithm (from "DR-Tulu: Reinforcement Learning with Evolving
Rubrics for Deep Research") for the pipeline's LLM and retriever infrastructure.

All DR-Tulu artefacts have been migrated locally:
  prompts/drtulu/system.txt  – system prompt  (action format: think/tool/answer/cite)
  src/drtulu_parser.py       – action parser   (UnifiedToolCallParserV20250824)

Action format (from prompts/drtulu/system.txt):
  <think>...</think>                    – internal reasoning (think action)
  <call_tool name="...">...</call_tool> – tool invocation   (tool action)
  <answer>...</answer>                  – final answer       (answer action)
  <cite id="...">...</cite>             – evidence citation  (cite action)

Inference algorithm (adapted from dr-tulu client.py
_generate_with_tools_commercial_api):

  WHILE step < max_iteration:
    1. LLM generates a segment  (think + optionally a tool call)
    2. Parse first <call_tool>  using UnifiedToolCallParserV20250824
    3. No tool call found       → natural stop (answer emitted or budget exhausted)
    4. Execute tool             → retrieve docs, format as <snippet id=...> XML
    5. Append tool output       → <tool_output>...</tool_output>  (user turn)
    6. Repeat
  Extract final prediction from <answer>...</answer>

LLM calls  : pipeline LiteLLMClient  (generator.complete)
Retrieval  : pipeline local retriever  (retriever.retrieve)
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Local DR-Tulu artefacts ───────────────────────────────────────────────────
_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "drtulu"
SYSTEM_PROMPT = (_PROMPT_DIR / "system.txt").read_text()

from src.drtulu_parser import UnifiedToolCallParserV20250824  # noqa: E402

# ── Pipeline imports (LLM + retriever) ───────────────────────────────────────
from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient              # noqa: E402

from .base_agent import BasicAgent                                                           # noqa: E402
from agent_tools.trajectory_tracker import TrackerCriticalThinkResult, TrackerEarlyStopResult  # noqa: E402
from prompts.trajectory_tracker.answer_prompts import FINAL_ANSWER_INSTRUCTION, DRTULU_FORMAT
from utils.doc_formatting import format_as_snippets                                          # noqa: E402

logger = logging.getLogger(__name__)

# ── Real tool implementations ─────────────────────────────────────────────────
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "agent_tools"
sys.path.insert(0, str(_TOOLS_DIR.parent))
from agent_tools import BrowseWebpageTool, SnippetSearchTool  # noqa: E402

# Tool names advertised in the system prompt.
_SEARCH_TOOLS: Tuple[str, ...] = ("snippet_search", "google_search", "browse_webpage")


class DrTulu_Agent(BasicAgent):
    """DR-Tulu deep research inference agent.

    Implements the Think → Tool → Answer → Cite loop from the DR-Tulu paper,
    using the pipeline's LLM client and retriever instead of the paper's
    MCP-backed web-search tools.

    Constructor signature matches other reasoning agents so that
    run_pipeline.py can instantiate it transparently::

        agent = DrTulu_Agent(
            llm_client=llm_client,
            retriever=retriever,
            max_iteration=100,
            seen_top_k=5,
        )
    """

    def __init__(self, llm_client: LiteLLMClient, retriever: Optional[Any] = None, max_iteration: int = 100, seen_top_k: int = 5, max_tokens_per_step: int = 1024, verbose: bool = True, search_backend: str = "serper") -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)

        # DR-Tulu action prompts (think / tool / answer / cite)
        self.system_prompt: str = SYSTEM_PROMPT

        # DR-Tulu action parser  <call_tool name="...">...</call_tool>
        self.parser: UnifiedToolCallParserV20250824 = UnifiedToolCallParserV20250824()

        # Real tool implementations
        self.browse_webpage_tool = BrowseWebpageTool()
        self.snippet_search_tool = SnippetSearchTool()

        self.max_tokens_per_step: int = max_tokens_per_step
        self.verbose: bool = verbose

    AGENT_NAME = "DrTulu"

    # ── Snippet ID helpers ────────────────────────────────────────────────────

    @staticmethod
    def _make_snippet_id(counter: int) -> str:
        """Return a snippet ID in the DR-Tulu format: S1, S2, …"""
        return f"S{counter}"

    # ── Document formatting ───────────────────────────────────────────────────

    def _format_docs_as_snippets(self, docs: List[Dict[str, Any]], snippet_offset: int) -> Tuple[str, List[str]]:
        """Render retrieved documents as DR-Tulu <snippet id="..."> XML."""
        return format_as_snippets(docs, self.seen_top_k, snippet_offset)

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_search(self, tool_name: str, query: str, parameters: Dict[str, Any], snippet_offset: int, reasoning: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], int]:
        """Dispatch a tool call and format results as DR-Tulu tool output.

        Routing:
            snippet_search  → SnippetSearchTool (S2_API_KEY)
            google_search   → pipeline search_tool or local retriever
            browse_webpage  → BrowseWebpageTool (JINA_API_KEY)

        Returns:
            tool_output_xml : <tool_output>...</tool_output> string.
            docs            : normalised document list (for trajectory).
            new_offset      : updated snippet counter.
        """
        docs: List[Dict[str, Any]] = []

        if tool_name == "snippet_search":
            docs = self.snippet_search_tool.execute(query)

        elif tool_name == "google_search":
            if self.search_tool is not None:
                docs = self.search_tool.execute(
                    query,
                    original_query=self._original_query,
                    reasoning=reasoning,
                )
            elif self.retriever:
                docs = self.retrieve_documents(query, original_query=self._original_query)

        elif tool_name == "browse_webpage":
            # For browse_webpage the query IS the URL
            docs = self.browse_webpage_tool.execute(query)

        if not docs:
            xml = "<tool_output>No results available.</tool_output>"
            return xml, [], snippet_offset

        formatted_snippets, ids = self._format_docs_as_snippets(docs, snippet_offset)
        new_offset = snippet_offset + len(ids)

        # Wrap in <tool_output> — mirrors UnifiedToolCallParserV20250824.format_result()
        tool_output_xml = f"<tool_output>\n{formatted_snippets}\n</tool_output>"
        return tool_output_xml, docs, new_offset

    # ── Action parser ─────────────────────────────────────────────────────────

    def _find_first_tool_call(self, text: str) -> Tuple[Optional[Any], Optional[str]]:
        """Scan *text* for the first <call_tool> action.

        Uses UnifiedToolCallParserV20250824 which handles both
        </call_tool> and </call> closing variants.

        Returns:
            (ToolCallInfo, tool_name)  if a call is found
            (None, None)               otherwise
        """
        for tool_name in _SEARCH_TOOLS:
            if self.parser.has_calls(text, tool_name):
                info = self.parser.parse_call(text, tool_name)
                if info is not None:
                    return info, tool_name
        return None, None

    # ── Main inference loop ───────────────────────────────────────────────────

    def inference(self, query: str, generation_temp: float = 0.7) -> Tuple[List[Dict[str, Any]], str, int]:
        """DR-Tulu inference loop.

        Implements the algorithm from dr-tulu client.py
        ``_generate_with_tools_commercial_api``, adapted to the pipeline's
        synchronous LiteLLMClient.

        Conversation structure (chat-API style):
          system : DR-Tulu system prompt  (think/tool/answer/cite instructions)
          user   : query
          -- loop --
          asst   : <think>...</think>  <call_tool name="...">query</call_tool>
          user   : <tool_output><snippet id="S1">...</snippet></tool_output>
          asst   : <think>...</think>  <answer>...</answer>
          -- end --

        Returns:
            reasoning_path : list of per-step dicts compatible with the pipeline
                             evaluators.  Keys per step:
                               action_type       – "search" | "answer"
                               think             – content of <think> tag
                               search_query      – tool query  (search steps only)
                               tool_name         – tool name   (search steps only)
                               docs              – normalised doc list
                               component_doc_ids – first seen_top_k doc ids
                               generation        – raw assistant text (answer step)
            prediction     : extracted answer string (from <answer> tag, or
                             stripped last assistant turn as fallback).
        """
        # ── Initialise conversation ───────────────────────────────────────────
        self._original_query = query
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": query},
        ]

        reasoning_path: List[Dict[str, Any]] = []
        snippet_offset: int = 0    # running counter for globally-unique snippet IDs
        full_generation: str = ""  # accumulated assistant text for answer extraction
        tracker_result = None

        self._print(f"Query: {query}")

        # ── Iterative Think → Tool → Answer loop ─────────────────────────────
        for _step in range(1, self.max_iteration + 1):
            iter_num = _step
            self._notify_progress("think", iter_num)

            # 1. Generate next segment (think + optional tool call / answer)
            try:
                response_text: str = self.generator.complete(
                    messages,
                    temperature=generation_temp,
                    max_tokens=self.max_tokens_per_step,
                )
            except Exception as e:
                logger.warning(f"Iteration {iter_num} API error: {e}")
                self._print("Context limit hit, forcing final answer from collected evidence")
                prompt = self._build_force_answer_prompt(
                    query, reasoning_path, DRTULU_FORMAT,
                )
                try:
                    forced_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "<answer>"},
                    ]
                    forced_resp = self.generator.complete(
                        forced_messages, temperature=generation_temp,
                    )
                    forced_text = "<answer>" + (forced_resp or "")
                    full_generation += forced_text
                    reasoning_path.append({
                        "action_type": "context_limit",
                        "think": FINAL_ANSWER_INSTRUCTION,
                        "generation": forced_text,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                except Exception as e2:
                    logger.warning(f"Forced final answer also failed: {e2}")
                break
            if not response_text:
                self._vprint(iter_num, "error", "Empty response, stopping")
                break

            # 2. Parse first <call_tool> action (DR-Tulu tool action)
            tool_info, matched_tool = self._find_first_tool_call(response_text)

            # Extract <think> text for the trajectory (DR-Tulu think action)
            think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            think_text = think_match.group(1).strip() if think_match else ""

            if tool_info is None:
                # ── Natural stop: model emitted <answer> or hit token budget ──
                full_generation += response_text
                messages = [*messages, {"role": "assistant", "content": response_text}]

                has_answer = "<answer>" in response_text
                self._notify_progress("answer", iter_num)
                self._vprint(iter_num, "think", think_text or "(no think)")
                self._vprint(iter_num, "answer" if has_answer else "stop", self._extract_answer(response_text) or "(budget exhausted)")

                reasoning_path.append({
                    "action_type":       "answer",
                    "think":             think_text,
                    "generation":        response_text,
                    "docs":              [],
                    "component_doc_ids": [],
                })
                break

            # ── Tool call found ───────────────────────────────────────────────
            # Truncate the assistant text to the end of the </call_tool> tag so
            # we never see hallucinated <tool_output> content.  This mirrors the
            # stop-sequence behaviour in dr-tulu client.py.
            truncated = response_text[: tool_info.end_pos]
            full_generation += truncated
            messages = [*messages, {"role": "assistant", "content": truncated}]

            # 3. Execute tool: retrieve docs → format as <snippet> XML
            tool_query = tool_info.content
            self._notify_progress("search", iter_num)
            self._vprint(iter_num, "think", think_text or "(no think)")
            self._vprint(iter_num, f"search ({matched_tool})", tool_query or "(empty)")
            tool_output_xml, docs, snippet_offset = self._execute_search(
                matched_tool, tool_query, tool_info.parameters, snippet_offset,
                reasoning=think_text if think_text else None,
            )
            # Trajectory tracker: may inject observation, critical_think, or early stop
            seen_docs = docs[:self.seen_top_k]
            tracker_result = self.post_search_evaluate(
                subquery=tool_query or "", docs=seen_docs,
                iter_num=iter_num, original_query=query,
                thinking=think_text,
                seen_docs=seen_docs,
                trajectory=messages,
            )
            _early_stop_triggered = isinstance(tracker_result, TrackerEarlyStopResult)
            if _early_stop_triggered:
                pass  # keep original tool_output_xml
            elif isinstance(tracker_result, TrackerCriticalThinkResult):
                tool_output_xml += (
                    f"\n<tool_output>\n"
                    f"[Critical Redirect — {tracker_result.critical_search_query}]\n"
                    f"{tracker_result.critical_observation}\n"
                    f"</tool_output>"
                )

            # 4. Record step in the trajectory (pipeline-compatible)
            reasoning_path.append({
                "action_type":       "search",
                "think":             think_text,
                "tool_name":         matched_tool,
                "search_query":      tool_query,
                "docs":              docs,
                "component_doc_ids": [
                    d.get("doc_id", "") for d in docs[: self.seen_top_k]
                ],
            })
            if isinstance(tracker_result, TrackerCriticalThinkResult):
                entry = self._critical_think_to_reasoning_entry(tracker_result)
                entry["tool_name"] = matched_tool
                reasoning_path.append(entry)

            # 5. Append tool output as user turn so the model can continue
            #    reasoning with the retrieved evidence (DR-Tulu output handling).
            messages = [*messages, {"role": "user", "content": tool_output_xml}]

            # Early stopping: break loop after appending tool output
            if _early_stop_triggered:
                break

        # ── Force final answer if budget exhausted on a search step ─────────
        # When max_iteration is reached and the last step was a search (not an answer),
        # the model never emitted <answer>.  Append one forced generation with
        # <answer> prefilled so the model must complete the block.
        last_step_was_search = (
            reasoning_path
            and reasoning_path[-1].get("action_type") == "search"
        )
        if last_step_was_search:
            is_early_stop = isinstance(tracker_result, TrackerEarlyStopResult) if tracker_result else False
            if is_early_stop:
                self._vprint(len(reasoning_path) + 1, "force-answer", "Early stopping triggered, forcing final answer")
            else:
                self._vprint(len(reasoning_path) + 1, "force-answer", "Budget exhausted on search, forcing final answer")
            user_content = FINAL_ANSWER_INSTRUCTION
            messages = [
                *messages,
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "<answer>"},
            ]
            forced_response = self.generator.complete(
                messages,
                temperature=generation_temp,
            )
            # Model continues from the prefill; reconstruct the full tag
            forced_text = "<answer>" + (forced_response or "")
            full_generation += forced_text
            action_type = "early_stop" if is_early_stop else "answer"
            reasoning_path.append({
                "action_type":       action_type,
                "think":             FINAL_ANSWER_INSTRUCTION,
                "generation":        forced_text,
                "docs":              [],
                "component_doc_ids": [],
            })

        # ── Extract final answer (DR-Tulu answer action) ─────────────────────
        prediction = self._extract_answer(full_generation)

        if not prediction:
            self._print("No <answer> tag found, using fallback extraction")
            # Fallback: strip all action tags from the last assistant turn
            last_asst = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                "",
            )
            prediction = re.sub(r"<think>.*?</think>", "", last_asst, flags=re.DOTALL)
            prediction = re.sub(
                r"<call_tool[^>]*>.*?(?:</call_tool>|</call>)",
                "",
                prediction,
                flags=re.DOTALL,
            )
            prediction = prediction.strip()

        num_searches = sum(1 for s in reasoning_path if s.get("action_type") == "search")
        self._print(f"Done: {len(reasoning_path)} steps, {num_searches} searches, answer {len(prediction or '')} chars")

        return reasoning_path, prediction, _step

    def _extract_answer(self, text: str) -> Optional[str]:
        """Return the last <answer>...</answer> block (DR-Tulu answer action)."""
        matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return matches[-1].strip() if matches else None
