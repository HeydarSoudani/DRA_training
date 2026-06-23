"""GPT-OSS agent: faithful port of the AgentIR OSS client.

Adapted from AgentIR/evaluation/search_agent/oss_client.py
(run_conversation_with_tools) to the pipeline's BasicAgent interface.

LLM calls  : OpenAI Responses API → vLLM server
Retrieval  : pipeline local retriever (self.retrieve_documents)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import openai

from pathlib import Path

from .base_agent import BasicAgent
from deep_research_agents.prompts.oss.user import QUERY_TEMPLATE
from controller_component.prompts.answer_prompts import FINAL_ANSWER_INSTRUCTION, OSS_FORMAT
from utils.config import InferenceConfig

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "oss"
OSS_SYSTEM_PROMPT = (_PROMPT_DIR / "system.txt").read_text()


class OSS_Agent(BasicAgent):
    """GPT-OSS deep research agent using the OpenAI Responses API."""

    AGENT_NAME = "OSS"

    def __init__(self, llm_client=None, retriever=None, max_iteration: int = 100, seen_top_k: int = 5, model_url: Optional[str] = None, model_name: str = "openai/gpt-oss-20b", max_output_tokens: int = 20000, reasoning_effort: str = "high", verbose: bool = True) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)

        self.model_url = model_url or os.getenv(
            "OSS_API_BASE", "http://localhost:6008/v1"
        )
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose
        self._api_key = os.getenv("OSS_API_KEY", "EMPTY")

        from utils.token_meter import TokenMeter
        self.token_meter = TokenMeter()

        self.inference_config = InferenceConfig(
            api_type="responses_api",
            model_name=model_name,
            api_base=self.model_url,
            api_key=self._api_key,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            system_prompt=OSS_SYSTEM_PROMPT,
            format_instructions=OSS_FORMAT,
        )

    # _print(), _notify_progress(), _get_tool_definitions(), and
    # _format_search_results() inherited from BasicAgent

    # run_single() inherited from BasicAgent (handles 3-value inference return)

    # ── Main inference loop ───────────────────────────────────────────────────

    def inference( self, query: str, generation_temp: float = 0.7 ) -> Tuple[List[Dict[str, Any]], str, int]:
        """OSS Responses API inference loop.

        Faithful port of run_conversation_with_tools() from oss_client.py.

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

        initial_request = {
            "model": cfg.model_name,
            "max_output_tokens": cfg.max_output_tokens,
            "input": messages,
            "tools": self._get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {
                "effort": cfg.reasoning_effort,
                "summary": "detailed",
            },
        }

        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""

        self._print(f"Query: {query}")

        _early_stop = False
        _early_stop_notified = False
        self._reasoning_only_retries = 0
        iteration = 1
        _last_completed_iter = 0
        while iteration <= self.max_iteration:
            try:
                request = initial_request.copy()
                request["input"] = messages
                response = client.responses.create(**request)
                self.token_meter.record_usage(getattr(response, "usage", None))
            except Exception as e:
                logger.warning(f"Iteration {iteration} API error: {e}")
                self._print("Context limit hit, forcing final answer in conversation")
                forced_text = self._force_answer_responses_api_in_conversation(messages)
                if not forced_text:
                    self._print("Retrying with compressed evidence fallback")
                    forced_text = self._force_answer_responses_api_compressed(
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
                _last_completed_iter = iteration
                break

            response_dict = response.model_dump(mode="python")

            # vLLM bug: reasoning-only response with no content — retry without
            # counting this as an iteration so the displayed counter stays sequential.
            if (
                len(response_dict["output"]) >= 1
                and response_dict["output"][-1]["type"] == "reasoning"
                and not any(
                    item["type"] in ("message", "function_call", "mcp_call")
                    for item in response_dict["output"]
                )
            ):
                self._reasoning_only_retries += 1
                if self._reasoning_only_retries > 5:
                    logger.warning("Too many reasoning-only responses, breaking loop")
                    _last_completed_iter = iteration
                    break
                continue

            # vLLM bug: convert mcp_call to function_call format
            for i in range(len(response_dict["output"])):
                if response_dict["output"][i]["type"] == "mcp_call":
                    response_dict["output"][i] = {
                        "id": response_dict["output"][i]["id"],
                        "call_id": response_dict["output"][i]["id"],
                        "arguments": response_dict["output"][i]["arguments"],
                        "name": response_dict["output"][i]["name"],
                        "type": "function_call",
                        "status": None,
                    }
                messages.append(response_dict["output"][i])

            # Extract reasoning and text from output items
            cur_reasonings = [
                item
                for item in response_dict["output"]
                if item["type"] == "reasoning"
            ]
            cur_texts = [
                item
                for item in response_dict["output"]
                if item["type"] == "message"
            ]

            cur_reasoning = None
            if cur_reasonings:
                turn_reasonings = [
                    " ".join([c["text"] for c in item["content"]])
                    for item in cur_reasonings
                ]
                cur_reasoning = "\n".join(turn_reasonings)

            cur_text = None
            if cur_texts:
                parts = cur_texts[0]["content"]
                cur_text = "\n".join([part["text"] for part in parts])

            function_calls = [
                item
                for item in response_dict["output"]
                if item["type"] in ("function_call", "mcp_call", "custom_tool_call")
            ]

            # Record answer turn if text is present
            if cur_text:
                if not function_calls:
                    self._notify_progress("answer", iteration)
                self._vprint(iteration, "think", (cur_reasoning or "(no reasoning)")[:100])
                self._vprint(iteration, "answer", cur_text[:100])
                reasoning_path.append({
                    "action_type": "answer",
                    "think": cur_reasoning if not function_calls else "",
                    "generation": cur_text,
                    "docs": [],
                    "component_doc_ids": [],
                })
                prediction = cur_text

            # Terminate if no function calls and last output is a message
            if not function_calls:
                if messages and messages[-1].get("type") == "message":
                    _last_completed_iter = iteration
                    break
                continue

            # Process function calls — evaluate controller per query
            new_messages = messages.copy()
            n_searches = sum(1 for fc in function_calls if fc["name"] == "search")
            last_search_msg_idx: Optional[int] = None
            _first_controller_action = None
            _stop_tracking = False

            for idx, tool_call in enumerate(function_calls):
                sub_iter = idx if n_searches > 1 else None
                try:
                    arguments = json.loads(tool_call["arguments"])
                    search_query = arguments.get("query", "").strip()

                    if tool_call["name"] == "search":
                        if cur_reasoning and idx == 0:
                            self._notify_progress("think", iteration)
                            self._vprint(iteration, "think", cur_reasoning[:100])

                        if not search_query:
                            logger.warning(
                                f"Iteration {iteration}: empty search query, returning error to model. "
                                f"Raw arguments: {tool_call['arguments']!r}"
                            )
                            self._vprint(iteration, "search", "(empty query – asking model to retry)", sub_iter=sub_iter)
                            new_messages.append(self._build_tool_response_message(
                                tool_call["call_id"],
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
                        result_text = f"Error: Tool {tool_call['name']} not found"

                    new_messages.append(self._build_tool_response_message(
                        tool_call["call_id"], result_text,
                    ))
                    if tool_call["name"] == "search" and search_query:
                        last_search_msg_idx = len(new_messages) - 1

                        # Per-query controller: stop evaluating after first non-continue
                        if not _stop_tracking:
                            _result, _stop_tracking = self._track_query(
                                search_query, docs[:self.seen_top_k],
                                query, cur_reasoning, new_messages, reasoning_path,
                            )
                            if _result is not None:
                                _first_controller_action = _result

                except Exception as e:
                    error_msg = f"Error executing {tool_call.get('name', 'unknown')}: {e}"
                    logger.warning(error_msg)
                    new_messages.append(self._build_tool_response_message(
                        tool_call.get("call_id", ""), error_msg,
                    ))

            # Apply the first non-continue controller action
            if _first_controller_action is not None:
                if self._apply_controller_action(
                    _first_controller_action, new_messages, reasoning_path, last_search_msg_idx,
                    original_query=query,
                ):
                    _early_stop = True

            messages = new_messages
            _last_completed_iter = iteration
            iteration += 1

            # Early stopping: instruct model to answer, then one more iteration
            _should_break, _early_stop_notified = self._handle_early_stop_phase(
                _early_stop, _early_stop_notified, messages,
            )
            if _should_break:
                break

        if not prediction:
            self._print("Max iterations reached without answer, forcing final answer in conversation")
            forced_text = self._force_answer_responses_api_in_conversation(messages)
            if not forced_text:
                self._print("Retrying with compressed evidence fallback")
                forced_text = self._force_answer_responses_api_compressed(
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

        num_iterations = _last_completed_iter
        num_searches = sum(
            1 for s in reasoning_path if s.get("action_type") == "search"
        )
        self._print(
            f"Done: {num_iterations} iters, {len(reasoning_path)} steps, "
            f"{num_searches} searches, answer {len(prediction)} chars"
        )

        return reasoning_path, prediction, num_iterations
