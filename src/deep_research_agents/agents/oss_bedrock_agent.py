"""GPT-OSS Bedrock agent: calls gpt-oss models via Amazon Bedrock (LiteLLM).

Same behaviour as OSS_Agent (vLLM runner) but uses LiteLLM chat completions
with tool calling instead of the OpenAI Responses API.

LLM calls  : litellm.completion → Bedrock Converse API
Retrieval  : pipeline local retriever (self.retrieve_documents)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

from .base_agent import BasicAgent, _strip_tool_messages
from prompts.oss.user import QUERY_TEMPLATE
from controller_component.prompts.answer_prompts import FINAL_ANSWER_INSTRUCTION, OSS_FORMAT
from utils.config import InferenceConfig

logger = logging.getLogger(__name__)

# ── CLI model name → Bedrock model ID mapping ────────────────────────────────
BEDROCK_OSS_MODELS: Dict[str, str] = {
    "gpt-oss-120b": "bedrock/openai.gpt-oss-120b-1:0",
}
_DEFAULT_BEDROCK_MODEL = BEDROCK_OSS_MODELS["gpt-oss-120b"]

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "oss"
OSS_SYSTEM_PROMPT = (_PROMPT_DIR / "system.txt").read_text()


class OSS_BedrockAgent(BasicAgent):
    """GPT-OSS deep research agent using Bedrock via LiteLLM chat completions."""

    AGENT_NAME = "OSS-Bedrock"

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        max_iteration: int = 100,
        seen_top_k: int = 5,
        model_name: str = _DEFAULT_BEDROCK_MODEL,
        max_output_tokens: int = 20000,
        reasoning_effort: str = "high",
        verbose: bool = True,
    ) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.verbose = verbose
        self._aws_region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            system_prompt=OSS_SYSTEM_PROMPT,
            format_instructions=OSS_FORMAT,
        )

    def get_answer_candidate_llm(self):
        from utils.llm_client import LiteLLMClient
        return LiteLLMClient(model=self.model_name, aws_region_name=self._aws_region)

    # _get_tool_definitions() and _format_search_results() inherited from BasicAgent

    # ── Force answer helpers (LiteLLM / Bedrock overrides) ──────────────────

    def _force_answer_chat_in_conversation(self, messages):
        cfg = self.inference_config
        trimmed = self._trim_first_iteration(messages)
        trimmed = _strip_tool_messages(trimmed)
        trimmed.append({
            "role": "user",
            "content": f"{FINAL_ANSWER_INSTRUCTION}\n\n{cfg.format_instructions}",
        })
        try:
            resp = litellm.completion(
                model=cfg.model_name,
                messages=trimmed,
                max_tokens=cfg.max_output_tokens,
                reasoning_effort=cfg.reasoning_effort,
                aws_region_name=self._aws_region,
                modify_params=True,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Force answer (litellm, in conversation) failed: %s", e)
            return None

    def _force_answer_chat_compressed(self, query, reasoning_path):
        cfg = self.inference_config
        prompt = self._build_force_answer_prompt(
            query, reasoning_path, cfg.format_instructions,
        )
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = litellm.completion(
                model=cfg.model_name,
                messages=messages,
                max_tokens=cfg.max_output_tokens,
                reasoning_effort=cfg.reasoning_effort,
                aws_region_name=self._aws_region,
                modify_params=True,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Force answer (litellm, compressed) failed: %s", e)
            return None

    # run_single() inherited from BasicAgent

    # ── Main inference loop ──────────────────────────────────────────────────

    def inference(
        self, query: str, generation_temp: float = 0.7
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        """Bedrock chat-completions inference loop.

        Same think → search → observe loop as the vLLM runner, but uses
        litellm.completion() with tool definitions instead of the OpenAI
        Responses API.

        Returns:
            reasoning_path  : list of per-step dicts (pipeline-compatible)
            prediction      : extracted answer string
            num_iterations  : number of LLM iterations used
        """
        cfg = self.inference_config

        formatted_query = QUERY_TEMPLATE.format(Question=query)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": formatted_query},
        ]

        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""

        self._print(f"Query: {query}")

        _early_stop = False
        _early_stop_notified = False
        for iteration in range(1, self.max_iteration + 1):
            try:
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    tools=self._get_tool_definitions(),
                    temperature=generation_temp,
                    max_tokens=self.max_output_tokens,
                    reasoning_effort=self.reasoning_effort,
                    aws_region_name=self._aws_region,
                    modify_params=True,
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

            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""
            tool_calls = msg.tool_calls

            # ── Build assistant message for history ──────────────────────
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            # ── Extract reasoning from litellm response ──────────────────
            cur_reasoning = getattr(msg, "reasoning_content", None) or None
            visible_content = content

            # ── Record answer turn if visible content is present ─────────
            if visible_content:
                if not tool_calls:
                    self._notify_progress("answer", iteration)
                self._vprint(iteration, "think", (cur_reasoning or "(no reasoning)")[:100])
                self._vprint(iteration, "answer", visible_content[:100])
                reasoning_path.append({
                    "action_type": "answer",
                    "think": cur_reasoning if not tool_calls else "",
                    "generation": visible_content,
                    "docs": [],
                    "component_doc_ids": [],
                })
                prediction = visible_content

            # ── Terminate if no tool calls ───────────────────────────────
            if not tool_calls:
                if choice.finish_reason == "stop":
                    break
                continue

            # ── Process tool calls — evaluate tracker per query
            n_searches = sum(1 for tc in tool_calls if tc.function.name == "search")
            last_search_msg_idx: Optional[int] = None
            _first_tracker_action = None
            _stop_tracking = False

            for idx, tc in enumerate(tool_calls):
                sub_iter = idx if n_searches > 1 else None
                try:
                    arguments = json.loads(tc.function.arguments)
                    search_query = arguments.get("query", "").strip()

                    if tc.function.name == "search":
                        if cur_reasoning and idx == 0:
                            self._notify_progress("think", iteration)
                            self._vprint(iteration, "think", cur_reasoning[:100])

                        if not search_query:
                            logger.warning(
                                f"Iteration {iteration}: empty search query, "
                                "returning error to model. "
                                f"Raw arguments string: {tc.function.arguments!r} | "
                                f"Parsed keys: {list(arguments.keys())} | "
                                f"Parsed values: {arguments} | "
                                f"Content snippet: {content[:200]!r}"
                            )
                            self._vprint(
                                iteration, "search",
                                "(empty query – asking model to retry)",
                                sub_iter=sub_iter,
                            )
                            messages.append(self._build_tool_response_message(
                                tc.id,
                                "Error: search query is empty. Please provide "
                                "a non-empty, specific search query.",
                            ))
                            continue

                        self._notify_progress("search", iteration)
                        self._vprint(iteration, "search", search_query, sub_iter=sub_iter)

                        if self.search_tool is not None:
                            docs = self.search_tool.execute(
                                search_query,
                                original_query=query,
                                reasoning=cur_reasoning,
                            )
                        else:
                            docs = self.retrieve_documents(
                                search_query, original_query=query,
                            )
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
                        })
                    else:
                        result_text = f"Error: Tool {tc.function.name} not found"

                    messages.append(self._build_tool_response_message(tc.id, result_text))
                    if tc.function.name == "search" and search_query:
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
                    error_msg = f"Error executing {tc.function.name}: {e}"
                    logger.warning(error_msg)
                    messages.append(self._build_tool_response_message(tc.id, error_msg))

            # Apply the first non-continue tracker action
            if _first_tracker_action is not None:
                if self._apply_tracker_action(
                    _first_tracker_action, messages, reasoning_path, last_search_msg_idx,
                    original_query=query,
                ):
                    _early_stop = True

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
