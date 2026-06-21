"""CPM-Explore agent: single-agent tool-calling loop for deep exploration.

Adapted from the AgentCPM-Explore architecture (OpenBMB/AgentCPM) to the
pipeline's BasicAgent interface.

Key features ported from the original:
- Multi-turn reasoning with <think> tags
- Tool calling via <tool_call> tags (parsed from text output)
- Relaxed <answer> tag extraction
- 3-level force-answer escalation (mild → aggressive → terminal)
- Repetition detection with history compression
- No-op counter with escalating prompts

LLM calls  : OpenAI-compatible Chat Completions API → vLLM server
Retrieval  : pipeline local retriever (self.retrieve_documents)
"""

import copy
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openai

from .base_agent import BasicAgent
from controller_component.prompts.answer_prompts import (
    CANDIDATE_GENERATION_INSTRUCTION,
    CPM_EXPLORE_FORMAT,
    FINAL_ANSWER_INSTRUCTION,
    AnswerCandidateOutput,
    extract_answer_candidates,
)
from utils.config import InferenceConfig

logger = logging.getLogger(__name__)

# ── Prompt loading ───────────────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "cpm_explore"


def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


# ── Respond instruction variants ────────────────────────────────────────────

_RESPOND_WITH_TOOLS = (
    "## How to Respond\n\n"
    "1. **Think**: Reason about the problem and plan your next action.\n"
    "2. **Search**: Call the search tool to find relevant information. "
    "Use different queries to explore multiple angles.\n"
    "3. **Answer**: When you have gathered sufficient evidence, "
    "provide your final answer inside <answer></answer> tags.\n\n"
    "To call a tool, output:\n"
    '<tool_call>{"name": "search", "arguments": {"query": "your search query"}}</tool_call>'
)

_RESPOND_NO_TOOLS = (
    "## How to Respond\n\n"
    "1. **Think**: Reason about the problem and plan your next action.\n"
    "2. **Answer**: When you have gathered sufficient evidence, "
    "provide your final answer inside <answer></answer> tags."
)

# ── Query template ───────────────────────────────────────────────────────────

QUERY_TEMPLATE = "Your task is to answer the user's question: {Question}"

# ── Answer extraction (relaxed regex from original CPM-Explore) ──────────────

_ANSWER_PATTERN = re.compile(
    r"(?is)(?:<answer>|<answer|answer>)(.*?)(?:</answer>|</answer|/answer>)"
)


def _extract_answer(text: str) -> Optional[str]:
    """Extract the last <answer>...</answer> from text (relaxed matching)."""
    matches = _ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


# ── Tool call parsing (multiple strategies from original CPM-Explore) ────────


def _try_parse_json(text: str) -> Optional[Dict]:
    """Try to parse JSON with fallback repairs."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text + "}")
        except json.JSONDecodeError:
            pass
    return None


def _normalize_tool_call(parsed: Dict) -> Dict[str, Any]:
    """Normalize parsed JSON into an OpenAI-style tool call dict."""
    name = parsed.get("name", "")
    arguments = parsed.get("arguments", parsed.get("parameters", {}))
    if isinstance(arguments, str):
        arguments = _try_parse_json(arguments) or {"query": arguments}
    return {
        "id": "tool-call-" + str(uuid.uuid4()),
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
        },
    }


def _parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM text output.

    Strategies tried in order:
    1. <tool_call>JSON</tool_call> XML tags (also handles unclosed tags
       from truncated output, matching the original AgentCPM parser)
    2. ```json ... ``` markdown code blocks
    3. Raw JSON object with a ``name`` key
    """
    tool_calls: List[Dict[str, Any]] = []

    # Strategy 1: <tool_call>...</tool_call> XML tags + unclosed fallback
    tc_matches_raw = re.findall(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", text, re.DOTALL,
    )
    tc_matches = [
        m[0] if m[0] else m[1]
        for m in tc_matches_raw
        if m[0] or m[1]
    ]
    for match in tc_matches:
        parsed = _try_parse_json(match.strip())
        if parsed and "name" in parsed:
            tool_calls.append(_normalize_tool_call(parsed))
    if tool_calls:
        return tool_calls

    # Strategy 2: markdown code blocks
    md_matches = re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    for match in md_matches:
        parsed = _try_parse_json(match.strip())
        if parsed and "name" in parsed:
            tool_calls.append(_normalize_tool_call(parsed))
    if tool_calls:
        return tool_calls

    # Strategy 3: raw JSON object
    cleaned = text.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        parsed = _try_parse_json(cleaned)
        if parsed and "name" in parsed:
            tool_calls.append(_normalize_tool_call(parsed))

    return tool_calls


# ═════════════════════════════════════════════════════════════════════════════
# Agent class
# ═════════════════════════════════════════════════════════════════════════════


class CPMExplore(BasicAgent):
    """AgentCPM-Explore: single-agent tool-calling loop for deep search."""

    AGENT_NAME = "CPM-Explore"

    NO_OP_THRESHOLD = 20
    REPETITION_THRESHOLD = 4
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        max_iteration: int = 100,
        seen_top_k: int = 5,
        model_url: Optional[str] = None,
        model_name: str = "openbmb/AgentCPM-Explore",
        max_output_tokens: int = 16384,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)

        self.model_url = model_url or os.getenv(
            "CPM_EXPLORE_API_BASE", "http://localhost:6008/v1"
        )
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self._system_prompt_template = system_prompt or _load_prompt("system.txt")
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.top_p = top_p
        self.verbose = verbose
        self._api_key = os.getenv("CPM_EXPLORE_API_KEY", "EMPTY")

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            model_name=model_name,
            api_base=self.model_url,
            api_key=self._api_key,
            max_output_tokens=max_output_tokens,
            reasoning_effort=None,
            system_prompt=self._build_system_prompt(),
            format_instructions=CPM_EXPLORE_FORMAT,
        )

    # _get_tool_definitions() and _format_search_results() inherited from BasicAgent

    def _build_system_prompt(self) -> str:
        tools = self._get_tool_definitions()
        tools_json = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tools_section = (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        return self._system_prompt_template.format(
            tools_section=tools_section,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _build_system_prompt_no_tools(self) -> str:
        """Build the system prompt with the search tool removed.

        Used during force-stop / early-stop so the model can no longer
        see or invoke the search tool — matching the original AgentCPM
        behaviour of filtering tools when forcing convergence.
        """
        tools_section = (
            "No tools are available. Based on all information gathered so far, "
            "provide your final answer inside <answer></answer> tags."
        )
        return self._system_prompt_template.format(
            tools_section=tools_section,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

    def _swap_system_prompt_no_tools(self, messages: List[Dict[str, Any]]) -> None:
        """Replace the system prompt in messages[0] with the no-tools variant."""
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = self._build_system_prompt_no_tools()

    # ── Repetition detection ─────────────────────────────────────────────────

    @staticmethod
    def _response_signature(content: str, tool_calls: List) -> str:
        tc_keys = sorted(
            tc["function"]["name"] + tc["function"]["arguments"]
            for tc in tool_calls
        ) if tool_calls else []
        payload = json.dumps(
            {"c": (content or "")[:200], "t": tc_keys}, sort_keys=True,
        )
        return hashlib.md5(payload.encode()).hexdigest()

    # ── Force-answer prompts (3-level escalation) ────────────────────────────

    _NO_OP_MILD = (
        "You didn't provide a tool call or a final answer. "
        "Please reassess your plan, take the next step, "
        "or provide your final answer."
    )

    _NO_OP_AGGRESSIVE = (
        "You have repeatedly failed to produce a tool call or a final answer. "
        "Do NOT make any more tool calls. "
        "Based on all information gathered so far, provide your best answer NOW."
    )

    # _force_answer_chat_in_conversation() and _force_answer_chat_compressed()
    # inherited from BasicAgent

    # ── Answer candidate (Chat Completions, same client as main loop) ───────

    def generate_answer_candidate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
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

        # Swap system prompt to no-tools variant with "no candidate" option
        if messages and messages[0].get("role") == "system":
            tools_section = (
                "No tools are available. Based on all information gathered so far, "
                "provide your final answer inside <answer></answer> tags.\n"
                "If the evidence is insufficient to answer, respond with 'no candidate'."
            )
            messages[0]["content"] = self._system_prompt_template.format(
                tools_section=tools_section,
                current_date=datetime.now().strftime("%Y-%m-%d"),
            )

        instruction = (
            f"{FINAL_ANSWER_INSTRUCTION}\n\n"
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
                max_tokens=min(remaining_tokens, 4096),
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

    # ── History compression ──────────────────────────────────────────────────

    def _compress_history(
        self,
        messages: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Compress conversation history via LLM summarization.

        Keeps system + original user message, replaces the middle with a
        summary.  Mirrors the Discard & Summarize strategy from the original
        AgentCPM-Explore implementation.
        """
        if len(messages) <= 3:
            return messages

        cfg = self.inference_config
        client = openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

        assistant_parts = []
        for msg in messages[2:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant" and content:
                assistant_parts.append(content[:500])
            elif role == "user" and content:
                assistant_parts.append(f"[Tool result]: {content[:200]}")

        if not assistant_parts:
            return messages

        history_text = "\n---\n".join(assistant_parts)

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarization assistant. Summarize the following "
                    "research conversation concisely, preserving key findings "
                    "and evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this research conversation for the task: "
                    f"'{query}'\n\n{history_text[:8000]}"
                ),
            },
        ]

        try:
            completion = client.chat.completions.create(
                model=cfg.model_name,
                messages=summary_messages,
                max_tokens=1024,
            )
            summary = completion.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"History compression failed: {e}")
            return messages

        compressed = [messages[0], messages[1]]
        compressed.append({
            "role": "assistant",
            "content": (
                "## Research Progress Summary\n\n"
                f"{summary}\n\n"
                "I will continue the task based on this summary."
            ),
        })

        self._print(
            f"Compressed history: {len(messages)} → {len(compressed)} messages"
        )
        return compressed

    # ── Main inference loop ──────────────────────────────────────────────────

    def inference(
        self, query: str, generation_temp: float = 0.7,
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        """CPM-Explore inference loop via OpenAI-compatible vLLM server.

        Single-agent tool-calling loop: the model reasons, searches, and
        eventually produces a final answer.  Tool calls are parsed from the
        model's text output (``<tool_call>`` XML tags, markdown JSON blocks,
        or raw JSON).  Search results are fed back as user messages wrapped
        in ``<tool_response>`` tags.

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

        system_prompt = self._build_system_prompt()
        formatted_query = QUERY_TEMPLATE.format(Question=query)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_query},
        ]

        reasoning_path: List[Dict[str, Any]] = []
        prediction = ""
        self._cumulative_output_tokens = 0

        no_op_count = 0
        last_signature: Optional[str] = None
        consecutive_repetitions = 0

        self._print(f"Query: {query}")

        _early_stop = False
        _early_stop_notified = False

        iteration = 1
        while iteration <= self.max_iteration:
            remaining_tokens = min(cfg.max_output_tokens - self._cumulative_output_tokens, 2048)
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

            create_kwargs: Dict[str, Any] = {
                "model": cfg.model_name,
                "messages": messages,
                "max_tokens": remaining_tokens,
            }
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature
            elif generation_temp is not None:
                create_kwargs["temperature"] = generation_temp
            if self.top_p is not None:
                create_kwargs["top_p"] = self.top_p

            try:
                completion = client.chat.completions.create(**create_kwargs)
            except Exception as e:
                logger.warning(f"Iteration {iteration} API error: {e}")
                self._print("Context limit hit, forcing final answer in conversation")
                self._swap_system_prompt_no_tools(messages)
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

            usage = getattr(completion, "usage", None)
            if usage is not None:
                self._cumulative_output_tokens += getattr(
                    usage, "completion_tokens", 0
                )

            choice = completion.choices[0]
            assistant_msg = {
                k: v for k, v in choice.message.model_dump().items()
                if v is not None
            }
            if "content" not in assistant_msg:
                assistant_msg["content"] = ""
            content = assistant_msg.get("content") or ""
            raw_content = content  # preserve for tool-call parsing

            # ── Extract thinking ─────────────────────────────────────────
            current_thinking = None
            reasoning_content = (
                getattr(choice.message, "reasoning_content", None)
                or assistant_msg.get("reasoning_content")
            )
            if reasoning_content:
                current_thinking = reasoning_content

            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                extracted_think = think_match.group(1).strip()
                if current_thinking:
                    current_thinking = current_thinking + "\n" + extracted_think
                else:
                    current_thinking = extracted_think
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL,
                ).strip()

            # Re-inject thinking into message content so the model sees
            # its own reasoning in subsequent turns (matches original
            # AgentCPM return_thought=True behaviour).
            if current_thinking and not re.search(r"<think>", assistant_msg.get("content", "")):
                parts = [f"<think>{current_thinking}</think>"]
                if content:
                    parts.append(content)
                assistant_msg["content"] = "\n".join(parts)

            for key in ("reasoning_content", "reasoning"):
                assistant_msg.pop(key, None)

            # ── Parse tool calls from text ───────────────────────────────
            # Parse from raw_content (before <think> stripping) so tool
            # calls adjacent to or inside thinking blocks are not lost.
            text_tool_calls = _parse_tool_calls_from_text(raw_content)
            if not text_tool_calls and current_thinking:
                text_tool_calls = _parse_tool_calls_from_text(current_thinking)
            native_tool_calls = assistant_msg.get("tool_calls") or []
            all_tool_calls = native_tool_calls or text_tool_calls

            # ── Check for answer ─────────────────────────────────────────
            answer = _extract_answer(content)
            if not answer and current_thinking:
                answer = _extract_answer(current_thinking)
            if not answer and not all_tool_calls:
                if "exact answer:" in content.lower():
                    answer = content

            # ── Repetition detection ─────────────────────────────────────
            current_sig = self._response_signature(content, all_tool_calls)
            if current_sig == last_signature:
                consecutive_repetitions += 1
            else:
                consecutive_repetitions = 1
            last_signature = current_sig

            if consecutive_repetitions >= self.REPETITION_THRESHOLD:
                self._print(
                    f"Repetition detected ({consecutive_repetitions}x), "
                    "compressing history"
                )
                messages = self._compress_history(messages, query)
                consecutive_repetitions = 0
                continue

            messages.append(assistant_msg)

            # ── Case 1: answer found ─────────────────────────────────────
            if answer and not all_tool_calls:
                self._notify_progress("answer", iteration)
                self._vprint(iteration, "think", current_thinking or "(no thinking)")
                self._vprint(iteration, "answer", (answer or content)[:300])
                reasoning_path.append({
                    "action_type": "answer",
                    "think": current_thinking or "",
                    "generation": content,
                    "docs": [],
                    "component_doc_ids": [],
                })
                prediction = content
                break

            # ── Case 2: no tool calls and no answer → no-op ──────────────
            if not all_tool_calls:
                no_op_count += 1
                self._vprint(iteration, "no-op", f"count={no_op_count}")
                logger.debug(
                    "no-op raw_content (first 500 chars): %s",
                    raw_content[:500],
                )
                self._print(
                    f"  [debug] raw output ({len(raw_content)} chars): "
                    f"{raw_content[:300]!r}"
                )

                if no_op_count >= self.NO_OP_THRESHOLD:
                    messages.append({
                        "role": "system",
                        "content": self._NO_OP_AGGRESSIVE,
                    })
                    no_op_count = 0
                    self._print("Injected aggressive force-answer prompt")
                else:
                    messages.append({
                        "role": "user",
                        "content": self._NO_OP_MILD,
                    })
                continue

            # ── Case 3: tool calls → execute ─────────────────────────────
            if current_thinking:
                self._notify_progress("think", iteration)
                self._vprint(iteration, "think", current_thinking[:300])

            no_op_count = 0
            n_searches = sum(1 for tc in all_tool_calls if tc["function"]["name"] == "search")
            last_search_msg_idx: Optional[int] = None
            _first_tracker_action = None
            _stop_tracking = False

            for idx, tool_call in enumerate(all_tool_calls):
                tname = tool_call["function"]["name"]
                targs_str = tool_call["function"]["arguments"]
                sub_iter = idx if n_searches > 1 else None

                try:
                    targs = (
                        json.loads(targs_str)
                        if isinstance(targs_str, str)
                        else targs_str
                    )

                    if tname == "search":
                        search_query = targs.get("query", "")
                        self._notify_progress("search", iteration)
                        self._vprint(iteration, "search", search_query, sub_iter=sub_iter)

                        if self.search_tool is not None:
                            docs = self.search_tool.execute(
                                search_query,
                                original_query=query,
                                reasoning=current_thinking,
                            )
                        else:
                            docs = self.retrieve_documents(
                                search_query, original_query=query,
                            )
                        result_text = self._format_search_results(docs)
                        self._vprint_docs(iteration, docs[:self.seen_top_k], sub_iter=sub_iter)

                        reasoning_path.append({
                            "action_type": "search",
                            "think": current_thinking if idx == 0 else "",
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
                        result_text = (
                            f"Error: Tool '{tname}' not found. "
                            "Available tools: search"
                        )

                    # CPM-Explore expects <tool_response> tags (text-based tool calling)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"<tool_response>\n{result_text}\n</tool_response>"
                        ),
                    })
                    if tname == "search" and search_query:
                        last_search_msg_idx = len(messages) - 1

                        # Per-query tracker: stop evaluating after first non-continue
                        if not _stop_tracking:
                            _result, _stop_tracking = self._track_query(
                                search_query, docs[:self.seen_top_k],
                                query, current_thinking, messages, reasoning_path,
                            )
                            if _result is not None:
                                _first_tracker_action = _result

                except Exception as e:
                    error_msg = f"Error executing {tname}: {e}"
                    logger.warning(error_msg)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"<tool_response>\n{error_msg}\n</tool_response>"
                        ),
                    })

            # Apply the first non-continue tracker action
            if _first_tracker_action is not None:
                if self._apply_tracker_action(
                    _first_tracker_action, messages, reasoning_path, last_search_msg_idx,
                    original_query=query,
                ):
                    _early_stop = True

            iteration += 1

            # Early stopping: instruct model to answer, then one more iteration
            if _early_stop and not _early_stop_notified:
                self._swap_system_prompt_no_tools(messages)
            _should_break, _early_stop_notified = self._handle_early_stop_phase(
                _early_stop, _early_stop_notified, messages,
            )
            if _should_break:
                break

        # ── No answer after loop → terminal force ────────────────────────
        if not prediction:
            self._print("Max iterations reached without answer, forcing final answer in conversation")
            self._swap_system_prompt_no_tools(messages)
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
