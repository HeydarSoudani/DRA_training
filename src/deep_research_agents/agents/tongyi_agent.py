"""Tongyi-DeepResearch agent: faithful port of the AgentIR Tongyi-DR ReAct loop.

LLM calls  : VLLMClient (OpenAI-compatible Chat Completions API -> vLLM server)
Retrieval  : pipeline local retriever (self.retrieve_documents)
"""

import copy
import json
import logging
import os
import time
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import json5
except ImportError:
    json5 = None

from .base_agent import BasicAgent
from agent_tools.trajectory_tracker import TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult
from prompts.trajectory_tracker.answer_prompts import (
    FINAL_ANSWER_INSTRUCTION,
    TAG_FORMAT,
    TONGYI_CANDIDATE_ANSWER,
    TONGYI_FORCE_ANSWER,
    AnswerCandidateOutput,
    extract_answer_candidates,
)
from utils.parsing import extract_tag_content, extract_all_tag_content, parse_tool_calls_xml_list
from utils.doc_formatting import format_as_markdown
from utils.inference_config import InferenceConfig
from utils.llm_client import VLLMClient

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "tongyi"
SYSTEM_PROMPT_SEARCH_ONLY = (_PROMPT_DIR / "system_search_only.txt").read_text()


def _today_date() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


class TongyiDR_Agent(BasicAgent):
    """Tongyi-DeepResearch ReAct agent."""

    AGENT_NAME = "TongyiDR"

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        max_iteration: int = 100,
        seen_top_k: int = 5,
        model_url: Optional[str] = None,
        model_name: str = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        temperature: float = 0.6,
        top_p: float = 0.95,
        presence_penalty: float = 1.1,
        max_tokens_per_step: int = 4096,
        max_output_tokens: int = 20000,
        max_context_tokens: int = 108 * 1024,
        verbose: bool = True,
    ) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)

        self.model_url = model_url or os.getenv("TONGYI_API_BASE", "http://127.0.0.1:6008/v1")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens_per_step = max_tokens_per_step
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens
        self.verbose = verbose
        self._cumulative_output_tokens = 0

        self._api_key = os.getenv("TONGYI_API_KEY", "EMPTY")
        self._default_stop_sequences = ["\n<tool_response>", "<tool_response>"]
        self._error_return = "vllm server error!!!"
        self._prepend_reasoning = True

        self._vllm_client = VLLMClient(
            model_url=self.model_url,
            model_name=self.model_name,
            api_key=self._api_key,
        )

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            model_name=model_name,
            api_base=self.model_url,
            api_key=self._api_key,
            max_output_tokens=max_output_tokens,
            system_prompt=SYSTEM_PROMPT_SEARCH_ONLY + " " + _today_date(),
            format_instructions=TAG_FORMAT,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @contextmanager
    def _override_temperature(self, temp: float):
        """Temporarily override the default temperature."""
        old = self.temperature
        self.temperature = temp
        try:
            yield
        finally:
            self.temperature = old

    def _call_server(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Call vLLM server via VLLMClient with agent-specific defaults."""
        return self._vllm_client.call(
            messages,
            stop=self._default_stop_sequences,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens_per_step,
            error_return=self._error_return,
            prepend_reasoning=self._prepend_reasoning,
        )

    @staticmethod
    def estimate_token_count_from_messages(messages: List[Dict[str, str]]) -> int:
        """Rough token count estimate (~4 chars per token)."""
        return sum(len(m.get("content", "")) for m in messages) // 4

    # ── Answer candidate (Chat Completions, same VLLMClient as main loop) ────

    def generate_answer_candidate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
        """Generate answer candidates using the same VLLMClient as the main loop."""
        if not isinstance(trajectory, list):
            logger.warning(
                "generate_answer_candidate requires a message list. "
                "Got %s; skipping.", type(trajectory).__name__,
            )
            return [AnswerCandidateOutput(
                candidate="no candidate",
                reasoning="trajectory is not a message list",
            )]

        messages = copy.deepcopy(trajectory)
        messages[-1] = {"role": "user", "content": TONGYI_CANDIDATE_ANSWER}

        cumulative = getattr(self, "_cumulative_output_tokens", 0)
        remaining_tokens = max(self.max_output_tokens - cumulative, 1024)

        try:
            raw = self._call_server(messages, max_tokens=min(remaining_tokens, self.max_tokens_per_step))
        except Exception:
            logger.warning("Answer candidate API call failed", exc_info=True)
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning="API call failed",
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
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning=raw.strip(),
            )]
        return candidates

    # ── Force answer helpers (Chat Completions, same VLLMClient) ─────────────

    def _force_answer_chat_in_conversation(
        self,
        messages: List[Dict[str, str]],
    ) -> Optional[str]:
        """Replace last message with force-answer instruction (AgentIR style)."""
        messages[-1] = {"role": "user", "content": TONGYI_FORCE_ANSWER}
        try:
            return self._call_server(messages)
        except Exception as e:
            logger.warning("Force answer (in conversation) failed: %s", e)
            return None

    def _force_answer_chat_compressed(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Standalone prompt with compressed evidence — last-resort fallback."""
        cfg = self.inference_config
        prompt = self._build_force_answer_prompt(
            query, reasoning_path, cfg.format_instructions,
        )
        messages: List[Dict[str, str]] = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self._call_server(messages)
        except Exception as e:
            logger.warning("Force answer (compressed) failed: %s", e)
            return None

    # ── Main inference loop ───────────────────────────────────────────────────

    def inference(
        self, query: str, generation_temp: float = 0.7
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        with self._override_temperature(generation_temp):
            system_prompt = SYSTEM_PROMPT_SEARCH_ONLY + " " + _today_date()
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            reasoning_path: List[Dict[str, Any]] = []
            prediction = ""
            start_time = time.time()
            self._cumulative_output_tokens = 0

            self._print(f"Query: {query}")

            _tongyi_early_stop = False
            _tongyi_early_stop_notified = False
            iteration = 1
            while iteration <= self.max_iteration:
                # Wall-clock timeout (AgentIR: 150 min)
                if time.time() - start_time > 150 * 60:
                    self._print("Time limit reached (150 min)")
                    reasoning_path.append({
                        "action_type": "answer",
                        "think": "",
                        "generation": "No answer found after 2h30mins",
                        "docs": [],
                        "component_doc_ids": [],
                    })
                    return reasoning_path, "No answer found after 2h30mins", iteration

                # ── Token budget check ───────────────────────────────────
                remaining_tokens = min(self.max_output_tokens - self._cumulative_output_tokens, self.max_tokens_per_step)
                if remaining_tokens <= 0:
                    logger.info("Global output token budget exhausted, forcing final answer")
                    self._print("Token budget exhausted, forcing final answer in conversation")
                    forced_text = self._force_answer_chat_in_conversation(messages)
                    if not forced_text:
                        self._print("Retrying with compressed evidence fallback")
                        forced_text = self._force_answer_chat_compressed(
                            query, reasoning_path,
                        )
                    if forced_text:
                        prediction = forced_text
                        reasoning_path.append({
                            "action_type": "context_limit",
                            "think": FINAL_ANSWER_INSTRUCTION,
                            "generation": forced_text,
                            "docs": [],
                            "component_doc_ids": [],
                        })
                    break

                # ── LLM call ─────────────────────────────────────────────
                try:
                    content = self._call_server(messages, max_tokens=remaining_tokens)
                except Exception as e:
                    logger.warning(f"Iteration {iteration} API error: {e}")
                    self._print("Context limit hit, forcing final answer in conversation")
                    forced_text = self._force_answer_chat_in_conversation(messages)
                    if not forced_text:
                        self._print("Retrying with compressed evidence fallback")
                        forced_text = self._force_answer_chat_compressed(
                            query, reasoning_path,
                        )
                    if forced_text:
                        prediction = forced_text
                        reasoning_path.append({
                            "action_type": "context_limit",
                            "think": FINAL_ANSWER_INSTRUCTION,
                            "generation": forced_text,
                            "docs": [],
                            "component_doc_ids": [],
                        })
                    break

                # Track cumulative output tokens (estimate ~4 chars per token)
                self._cumulative_output_tokens += len(content) // 4

                # Strip leaked <tool_response> (AgentIR behaviour)
                if "<tool_response>" in content:
                    content = content[: content.find("<tool_response>")]

                messages = [*messages, {"role": "assistant", "content": content.strip()}]

                # Extract reasoning
                think_matches = extract_all_tag_content(content, "think")
                think_text = think_matches[-1] if think_matches else ""

                # ── Tool calls ────────────────────────────────────────────
                if "<tool_call>" in content and "</tool_call>" in content:
                    tool_calls = parse_tool_calls_xml_list(content)
                    if not tool_calls:
                        tool_call_str = content.split("<tool_call>")[1].split("</tool_call>")[0]
                        try:
                            _loads = json5.loads if json5 is not None else json.loads
                            tool_calls = [_loads(tool_call_str)]
                        except Exception as e:
                            result_text = f"Error during tool execution: {e}"
                            reasoning_path.append({
                                "action_type": "search",
                                "think": think_text,
                                "search_query": "",
                                "docs": [],
                                "component_doc_ids": [],
                                "iteration": iteration,
                                "sub_iter": None,
                            })
                            messages = [
                                *messages,
                                {"role": "user", "content": f"<tool_response>\n{result_text}\n</tool_response>"},
                            ]
                            iteration += 1
                            continue

                    # Print think once for this step
                    think_preview = think_text[:200] + "..." if len(think_text) > 200 else think_text
                    self._notify_progress("think", iteration)
                    self._vprint(iteration, "think", think_preview.replace("\n", "\\n") or "(no think)")

                    # Collect search queries from all tool calls
                    search_queries: List[str] = []
                    error_tool_calls: List[str] = []
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})

                        if tool_name == "search":
                            raw_query = (
                                tool_args.get("query", "")
                                if isinstance(tool_args, dict)
                                else tool_args
                            )
                            if isinstance(raw_query, list):
                                search_queries.extend(str(q) for q in raw_query if q)
                            elif raw_query:
                                search_queries.append(str(raw_query))
                        else:
                            error_tool_calls.append(tool_name)

                    # Cap queries per step to avoid excessive retrieval
                    if len(search_queries) > 3:
                        self._print(f"Iteration {iteration}: capping {len(search_queries)} queries to 3")
                        search_queries = search_queries[:3]

                    # Report unavailable tools
                    if error_tool_calls:
                        bad = ", ".join(sorted(set(error_tool_calls)))
                        logger.warning(f"Iteration {iteration}: unavailable tool(s): {bad}")

                    if not search_queries and error_tool_calls:
                        bad = ", ".join(sorted(set(error_tool_calls)))
                        result_text = (
                            f"Error: Tool(s) {bad} not found. "
                            "The only available tool is search. Please use the search tool instead."
                        )
                        all_docs: List[Dict[str, Any]] = []
                    elif not search_queries:
                        result_text = "No valid search query found. Please use the search tool with a single query string."
                        all_docs = []
                    else:
                        # Process each search query individually — evaluate tracker per query
                        n_searches = len(search_queries)
                        all_subqueries: List[str] = []
                        all_docs = []
                        all_seen_docs: List[Dict[str, Any]] = []
                        _first_tracker_action = None
                        _stop_tracking = False

                        for idx, sq in enumerate(search_queries):
                            sub_iter = idx if n_searches > 1 else None

                            if not sq.strip():
                                logger.warning(
                                    f"Iteration {iteration}: empty search query at index {idx}, "
                                    "returning error to model."
                                )
                                self._vprint(iteration, "search", "(empty query – skipped)", sub_iter=sub_iter)
                                continue

                            self._notify_progress("search", iteration)
                            self._vprint(iteration, "search", sq, sub_iter=sub_iter)

                            if self.search_tool is not None:
                                docs = self.search_tool.execute(
                                    sq,
                                    original_query=query,
                                    reasoning=think_text if think_text else None,
                                )
                            else:
                                docs = self.retrieve_documents(sq, original_query=query)

                            self._vprint_docs(iteration, docs[:self.seen_top_k], sub_iter=sub_iter)

                            all_subqueries.append(sq)
                            all_docs.extend(docs)
                            all_seen_docs.extend(docs[:self.seen_top_k])

                            reasoning_path.append({
                                "action_type": "search",
                                "think": think_text if idx == 0 else "",
                                "search_query": sq,
                                "docs": docs,
                                "all_docs": docs,
                                "component_doc_ids": [
                                    d.get("doc_id", "")
                                    for d in docs[: self.seen_top_k]
                                ],
                                "iteration": iteration,
                                "sub_iter": sub_iter,
                            })

                            # Per-query tracker: stop evaluating after first non-continue
                            if not _stop_tracking:
                                _result, _stop_tracking = self._track_query(
                                    sq, docs[:self.seen_top_k],
                                    query, think_text, messages, reasoning_path,
                                )
                                if _result is not None:
                                    _first_tracker_action = _result

                        # Format combined results for model (single <tool_response>)
                        result_text = format_as_markdown(all_docs, self.seen_top_k)

                        # Apply the first non-continue tracker action
                        if _first_tracker_action is not None:
                            if isinstance(_first_tracker_action, TrackerEarlyStopResult):
                                _tongyi_early_stop = True
                            elif isinstance(_first_tracker_action, TrackerCriticalThinkDeferred):
                                _first_tracker_action = self._execute_deferred_critical_search(
                                    _first_tracker_action, query,
                                    trajectory=messages, reasoning_path=reasoning_path,
                                )
                                self._search_step += 1
                            if isinstance(_first_tracker_action, TrackerCriticalThinkResult):
                                result_text += self._format_critical_redirect_text(_first_tracker_action)
                                ct_entry = self._critical_think_to_reasoning_entry(
                                    _first_tracker_action, include_all_docs=True,
                                )
                                ct_entry["iteration"] = _first_tracker_action.critical_think_iter
                                reasoning_path.append(ct_entry)
                                iteration += 1

                    messages = [
                        *messages,
                        {"role": "user", "content": f"<tool_response>\n{result_text}\n</tool_response>"},
                    ]

                    iteration += 1

                    # Early stopping: replace last message with force-answer (AgentIR style)
                    if _tongyi_early_stop and not _tongyi_early_stop_notified:
                        messages[-1] = {"role": "user", "content": TONGYI_FORCE_ANSWER}
                        _tongyi_early_stop_notified = True
                    elif _tongyi_early_stop_notified:
                        break

                # ── Answer detected ───────────────────────────────────────
                elif "<answer>" in content:
                    answer_text = extract_tag_content(content, "answer") or ""
                    answer_preview = answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
                    self._notify_progress("answer", iteration)
                    self._vprint(iteration, "answer", answer_preview.replace("\n", "\\n"))
                    reasoning_path.append({
                        "action_type": "answer",
                        "think": think_text,
                        "generation": content,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                    prediction = answer_text or content.strip()
                    break

                else:
                    # No tool call and no answer — consume an iteration
                    iteration += 1

                # ── Context token limit check ─────────────────────────────
                if self.estimate_token_count_from_messages(messages) > self.max_context_tokens:
                    self._print("Context token limit reached, forcing answer")
                    forced_text = self._force_answer_chat_in_conversation(messages)
                    if not forced_text:
                        self._print("Retrying with compressed evidence fallback")
                        forced_text = self._force_answer_chat_compressed(
                            query, reasoning_path,
                        )
                    if forced_text:
                        prediction = forced_text
                        reasoning_path.append({
                            "action_type": "context_limit",
                            "think": FINAL_ANSWER_INSTRUCTION,
                            "generation": forced_text,
                            "docs": [],
                            "component_doc_ids": [],
                        })
                    break

            # ── Post-loop: force answer if no prediction ──────────────────
            if not prediction:
                self._print("Max iterations reached without answer, forcing final answer in conversation")
                forced_text = self._force_answer_chat_in_conversation(messages)
                if forced_text:
                    prediction = forced_text
                    reasoning_path.append({
                        "action_type": "max_iter_force",
                        "think": FINAL_ANSWER_INSTRUCTION,
                        "generation": forced_text,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                if not prediction:
                    self._print("Retrying with compressed evidence fallback")
                    forced_text = self._force_answer_chat_compressed(
                        query, reasoning_path,
                    )
                    if forced_text:
                        prediction = forced_text
                        reasoning_path.append({
                            "action_type": "max_iter_force",
                            "think": FINAL_ANSWER_INSTRUCTION,
                            "generation": forced_text,
                            "docs": [],
                            "component_doc_ids": [],
                        })
                if not prediction:
                    prediction = "No answer found."

            num_iterations = iteration
            num_searches = sum(
                1 for s in reasoning_path if s.get("action_type") == "search"
            )
            self._print(
                f"Done: {num_iterations} iters, {len(reasoning_path)} steps, "
                f"{num_searches} searches, answer {len(prediction)} chars"
            )
            return reasoning_path, prediction, num_iterations
