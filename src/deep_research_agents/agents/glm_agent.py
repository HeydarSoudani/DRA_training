"""GLM agent: faithful port of the AgentIR GLM client.

Adapted from AgentIR/evaluation/search_agent/glm_client.py
(run_conversation_with_tools) to the pipeline's BasicAgent interface.

LLM calls  : OpenAI Chat Completions API → vLLM server
             (Responses API not used because vLLM's --tool-call-parser glm47
              only works with the Chat Completions endpoint.)
Retrieval  : pipeline local retriever (self.retrieve_documents)
"""

import copy
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openai

from .base_agent import BasicAgent
from deep_research_agents.prompts.glm.user import QUERY_TEMPLATE
from controller_component.prompts.answer_prompts import (
    CANDIDATE_GENERATION_INSTRUCTION,
    FINAL_ANSWER_INSTRUCTION,
    OSS_FORMAT,
    AnswerCandidateOutput,
    extract_answer_candidates,
)
from utils.config import InferenceConfig

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "glm"
GLM_SYSTEM_PROMPT = (_PROMPT_DIR / "system.txt").read_text()


def _parse_tool_calls_from_text(text: str, defined_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse <tool_call> XML tags from reasoning content (GLM-specific quirk).

    GLM sometimes emits tool calls inside reasoning content rather than in the
    standard tool_calls field. This function extracts them.

    Returns tool calls in Chat Completions format (for use in assistant message).
    """

    def _get_argument_type(func_name: str, arg_key: str) -> str:
        for t in defined_tools:
            func = t.get("function", {})
            if func.get("name") == func_name:
                props = func.get("parameters", {}).get("properties", {})
                if arg_key in props:
                    return props[arg_key].get("type", "string")
        return "string"

    tool_calls = []
    tool_call_strs = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    for call in tool_call_strs:
        func_name_match = re.match(r"([^\n<]+)", call.strip())
        func_name = func_name_match.group(1).strip() if func_name_match else None
        if func_name:
            pairs = re.findall(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                call,
                re.DOTALL,
            )
            arguments = {}
            for arg_key, arg_value in pairs:
                arg_key = arg_key.strip()
                arg_value = arg_value.strip()
                arg_type = _get_argument_type(func_name, arg_key)
                if arg_type != "string":
                    try:
                        arg_value = json.loads(arg_value)
                    except Exception:
                        pass
                arguments[arg_key] = arg_value

            call_id = "tool-call-" + str(uuid.uuid4())
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            })
    return tool_calls


class GLM_Agent(BasicAgent):
    """GLM deep research agent using the OpenAI Chat Completions API."""

    AGENT_NAME = "GLM"

    def __init__(self, llm_client=None, retriever=None, max_iteration: int = 100, seen_top_k: int = 5, model_url: Optional[str] = None, model_name: str = "zai-org/GLM-4.7-Flash", max_output_tokens: int = 20000, system_prompt: Optional[str] = None, verbose: bool = True, **kwargs) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)

        self.model_url = model_url or os.getenv(
            "GLM_API_BASE", "http://localhost:6008/v1"
        )
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        self._api_key = os.getenv("GLM_API_KEY", "EMPTY")

        from utils.token_meter import TokenMeter
        self.token_meter = TokenMeter()

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            model_name=model_name,
            api_base=self.model_url,
            api_key=self._api_key,
            max_output_tokens=max_output_tokens,
            reasoning_effort=None,
            system_prompt=GLM_SYSTEM_PROMPT,
            format_instructions=OSS_FORMAT,
        )

    # _get_tool_definitions() and _format_search_results() inherited from BasicAgent

    # ── Answer candidate (Chat Completions, same client as main loop) ───────

    def generate_answer_candidate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
        """Generate answer candidates using Chat Completions API.

        Uses the same direct openai client as the main loop and force answer,
        with the full conversation + instruction appended.
        """
        cfg = self.inference_config

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
        instruction = (
            f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
            f"{cfg.format_instructions}"
        )
        messages.append({"role": "user", "content": instruction})

        cumulative = getattr(self, "_cumulative_output_tokens", 0)
        remaining_tokens = max(cfg.max_output_tokens - cumulative, 1024)

        client = openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        try:
            response = client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                max_tokens=min(remaining_tokens, 1024),
            )
        except Exception:
            logger.warning("Answer candidate API call failed", exc_info=True)
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning="API call failed",
            )]

        raw = response.choices[0].message.content or ""
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if not raw.strip() and reasoning_content:
            raw = "[reasoning_fallback]" + reasoning_content

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

    # _force_answer_chat_in_conversation() and _force_answer_chat_compressed()
    # inherited from BasicAgent

    # run_single() inherited from BasicAgent (handles 3-value inference return)

    # ── Main inference loop ───────────────────────────────────────────────────

    def inference(self, query: str, generation_temp: float = 0.7) -> Tuple[List[Dict[str, Any]], str, int]:
        """GLM Chat Completions inference loop.

        Faithful to AgentIR/evaluation/search_agent/glm_client.py
        (run_conversation_with_tools), using vLLM's Chat Completions API
        with tool calling (--tool-call-parser glm47).

        Returns:
            reasoning_path  : list of per-step dicts (pipeline-compatible)
            prediction      : extracted answer string
            num_iterations  : number of LLM iterations used
        """
        cfg = self.inference_config
        client = openai.OpenAI(
            base_url=cfg.api_base,
            api_key=cfg.api_key,
        )

        formatted_query = QUERY_TEMPLATE.format(Question=query)
        messages: List[Any] = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": formatted_query},
        ]
        tools = self._get_tool_definitions()

        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""
        self._cumulative_output_tokens = 0

        self._print(f"Query: {query}")

        _early_stop = False
        _early_stop_notified = False
        self._reasoning_only_retries = 0
        iteration = 1
        while iteration <= self.max_iteration:
            remaining_tokens = min(self.max_output_tokens - self._cumulative_output_tokens, 4096)
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
            try:
                response = client.chat.completions.create(
                    model=cfg.model_name,
                    messages=messages,
                    tools=tools,
                    max_tokens=remaining_tokens,
                )
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

            if response.usage:
                self._cumulative_output_tokens += response.usage.completion_tokens or 0
            self.token_meter.record_usage(getattr(response, "usage", None))

            message = response.choices[0].message
            cur_reasoning = getattr(message, "reasoning_content", None)
            cur_text = message.content
            official_tool_calls = message.tool_calls or []

            # vLLM bug: reasoning-only response with no content and no tool calls
            if cur_reasoning and not cur_text and not official_tool_calls:
                self._reasoning_only_retries += 1
                if self._reasoning_only_retries > 5:
                    logger.warning("Too many reasoning-only responses, breaking loop")
                    break
                messages.append(message)
                continue

            # Normalise tool calls to a uniform list of dicts
            function_calls: List[Dict[str, Any]] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in official_tool_calls
            ]

            # GLM quirk: parse tool calls embedded in reasoning text
            if cur_reasoning and not function_calls:
                parsed_from_reasoning = _parse_tool_calls_from_text(
                    cur_reasoning, tools,
                )
                if parsed_from_reasoning:
                    function_calls = [
                        {
                            "id": ptc["id"],
                            "name": ptc["function"]["name"],
                            "arguments": ptc["function"]["arguments"],
                        }
                        for ptc in parsed_from_reasoning
                    ]
                    # Create synthetic assistant message with these tool calls
                    messages.append({
                        "role": "assistant",
                        "content": cur_text or "",
                        "tool_calls": parsed_from_reasoning,
                    })
                else:
                    messages.append(message)
            else:
                messages.append(message)

            # Record answer turn if text is present
            if cur_text:
                if not function_calls:
                    self._notify_progress("answer", iteration)
                    self._vprint(iteration, "think", (cur_reasoning or "(no reasoning)")[:100])
                    self._vprint(iteration, "answer", cur_text[:100])
                    reasoning_path.append({
                        "action_type": "answer",
                        "think": cur_reasoning or "",
                        "generation": cur_text,
                        "docs": [],
                        "component_doc_ids": [],
                    })
                prediction = cur_text

            # Terminate if no function calls
            if not function_calls:
                break

            # Process function calls — evaluate tracker per query
            n_searches = sum(1 for fc in function_calls if fc["name"] == "search")
            last_search_msg_idx: Optional[int] = None
            _first_tracker_action = None
            _stop_tracking = False

            for idx, tc in enumerate(function_calls):
                sub_iter = idx if n_searches > 1 else None
                try:
                    arguments = json.loads(tc["arguments"])
                    search_query = arguments.get("query", "").strip()

                    if tc["name"] == "search":
                        if idx == 0 and (cur_reasoning or cur_text):
                            self._notify_progress("think", iteration)
                            think_parts = [p for p in [cur_reasoning, cur_text] if p]
                            self._vprint(iteration, "think", " | ".join(think_parts)[:100])

                        if not search_query:
                            logger.warning(
                                f"Iteration {iteration}: empty search query, returning error to model. "
                                f"Raw arguments: {tc['arguments']!r}"
                            )
                            self._vprint(iteration, "search", "(empty query – asking model to retry)", sub_iter=sub_iter)
                            messages.append(self._build_tool_response_message(
                                tc["id"],
                                "Error: search query is empty. Please provide a non-empty, specific search query.",
                            ))
                            continue

                        self._notify_progress("search", iteration)
                        self._vprint(iteration, "search", search_query, sub_iter=sub_iter)
                        if self.search_tool is not None:
                            docs = self.search_tool.execute(
                                search_query,
                                original_query=query,
                                reasoning=cur_reasoning if cur_reasoning else None,
                            )
                        else:
                            docs = self.retrieve_documents(search_query, original_query=query)
                        result_text = self._format_search_results(docs)
                        self._vprint_docs(iteration, docs[:self.seen_top_k], sub_iter=sub_iter)

                        reasoning_path.append({
                            "action_type": "search",
                            "think": cur_reasoning if idx == 0 else "",
                            "search_query": search_query,
                            "docs": docs,
                            "all_docs": docs,
                            "component_doc_ids": [
                                d.get("doc_id", "")
                                for d in docs[: self.seen_top_k]
                            ],
                            "iteration": iteration,
                            "sub_iter": sub_iter,
                            "tokens": self._step_tokens() if idx == 0 else None,
                        })
                    else:
                        result_text = f"Error: Tool {tc['name']} not found"

                    messages.append(self._build_tool_response_message(tc["id"], result_text))
                    if tc["name"] == "search" and search_query:
                        last_search_msg_idx = len(messages) - 1

                        # Per-query tracker: stop evaluating after first non-continue
                        if not _stop_tracking:
                            _result, _stop_tracking = self._track_query(
                                search_query, docs[:self.seen_top_k],
                                query, cur_reasoning, messages, reasoning_path,
                            )
                            if _result is not None:
                                _first_tracker_action = _result

                except Exception as e:
                    error_msg = f"Error executing {tc.get('name', 'unknown')}: {e}"
                    logger.warning(error_msg)
                    messages.append(self._build_tool_response_message(
                        tc.get("id", ""), error_msg,
                    ))

            # Apply the first non-continue tracker action
            if _first_tracker_action is not None:
                if self._apply_tracker_action(
                    _first_tracker_action, messages, reasoning_path, last_search_msg_idx,
                    original_query=query,
                ):
                    _early_stop = True

            iteration += 1

            # Early stopping: instruct model to answer, then one more iteration
            _should_break, _early_stop_notified = self._handle_early_stop_phase(
                _early_stop, _early_stop_notified, messages,
            )
            if _should_break:
                break

        if not prediction:
            self._print("Max iterations reached without answer, forcing final answer in conversation")
            forced_text = self._force_answer_chat_in_conversation(messages)
            if not forced_text:
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
