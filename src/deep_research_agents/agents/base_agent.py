"""Base class for retrieval agents."""

import copy
import logging
import traceback
from typing import Callable, Dict, List, Any, Optional, Union

from utils.llm_client import LiteLLMClient
from searcher_component import normalize_retrieval_response
from controller_component import TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult  # noqa: E501
from controller_component.prompts.answer_prompts import (
    CANDIDATE_GENERATION_INSTRUCTION,
    FINAL_ANSWER_INSTRUCTION,
    TAG_FORMAT,
    AnswerCandidateOutput,
    extract_answer_candidates,
)
from utils.text_utils import passages2string, format_as_json  # noqa: F401 – re-exported for back-compat
from utils.text_utils import reduce_reasoning_path, build_evidence_summary
from utils.config import InferenceConfig
from utils.text_utils import verbose_print, verbose_print_search_results, verbose_print_tracker  # noqa: F401
from utils.text_utils import get_think as _get_think, get_query as _get_query, get_answer as _get_answer  # noqa: E501

logger = logging.getLogger(__name__)

_TrackerResult = Union[TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult]


class AgentVerboseMixin:
    """Mixin providing verbose logging and controller helpers."""

    @property
    def _display_name(self) -> str:
        return getattr(self, "name", None) or getattr(self, "AGENT_NAME", "Agent")

    @property
    def _is_verbose(self) -> bool:
        return getattr(self, "verbose", True)

    def _print(self, message: str) -> None:
        if self._is_verbose:
            print(f"[{self._display_name}] {message}")

    def _vprint(self, iter_num: int, component: str, message: str, *, sub_iter: int = None) -> None:
        if self._is_verbose:
            verbose_print(iter_num, component, message, agent_name=self._display_name, sub_iter=sub_iter)

    def _vprint_docs(self, iter_num: int, docs: list, *, sub_iter: int = None) -> None:
        if self._is_verbose:
            verbose_print_search_results(iter_num, docs, agent_name=self._display_name, sub_iter=sub_iter)

    def _vprint_tracker(self, iter_num: int, scores: dict, action: str, *, sub_iter: int = None) -> None:
        if self._is_verbose:
            verbose_print_tracker(iter_num, scores, action, agent_name=self._display_name, sub_iter=sub_iter)

    def _critical_think_to_reasoning_entry(self, ct: TrackerCriticalThinkResult, *, include_all_docs: bool = False) -> dict:
        seen_top_k = getattr(self, "seen_top_k", 5)
        entry = {
            "action_type": "critical_search",
            "think": ct.critical_think,
            "search_query": ct.critical_search_query,
            "docs": ct.critical_docs,
            "component_doc_ids": [d.get("doc_id", "") for d in ct.critical_docs[:seen_top_k]],
            "is_critical_think": True,
        }
        if include_all_docs:
            entry["all_docs"] = ct.critical_docs
        return entry

    @staticmethod
    def _format_critical_redirect_text(ct: TrackerCriticalThinkResult) -> str:
        return (
            f"\n\n[Critical Redirect — {ct.critical_search_query}]\n"
            f"{ct.critical_observation}"
        )

    def post_search_evaluate(
        self,
        subquery: Union[str, List[str]],
        docs: List[Dict[str, Any]],
        iter_num: int,
        original_query: Optional[str] = None,
        thinking: str = "",
        seen_docs: Optional[List[Dict[str, Any]]] = None,
        sub_iter: Optional[int] = None,
        trajectory: Any = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        defer_critical_search: bool = False,
        **kwargs,
    ) -> Optional[_TrackerResult]:
        """Evaluate a search step via the controller.

        Returns None (continue), TrackerCriticalThinkResult (inject notice),
        TrackerCriticalThinkDeferred (deferred intervene), or TrackerEarlyStopResult.
        """
        tracker = getattr(self, "controller", None)
        if tracker is None:
            if seen_docs is not None:
                self._vprint_docs(iter_num, seen_docs, sub_iter=sub_iter)
            return None

        decision = tracker.evaluate(
            subquery=subquery, docs=docs, original_query=original_query or "",
            iter_num=iter_num, thinking=thinking, trajectory=trajectory,
            reasoning_path=reasoning_path, **kwargs,
        )

        critical_think_triggered = (
            decision.critical_thinking_output is not None
            and decision.critical_thinking_output.search_query.strip()
        )

        if self._is_verbose:
            if seen_docs is not None:
                self._vprint_docs(iter_num, seen_docs, sub_iter=sub_iter)
            self._vprint_tracker(iter_num, decision.scores, decision.action, sub_iter=sub_iter)
            if critical_think_triggered:
                self._vprint(iter_num, "notice",
                    f"[{decision.action}] critical_think" + (" (deferred)" if defer_critical_search else ""),
                    sub_iter=sub_iter)

        if critical_think_triggered:
            pno = decision.critical_thinking_output
            if defer_critical_search:
                return TrackerCriticalThinkDeferred(
                    critical_think=pno.reasoning, critical_search_query=pno.search_query,
                    scores=decision.scores, iter_num=iter_num,
                )
            retrieve_fn = getattr(self, "retrieve_documents", None)
            if retrieve_fn is None:
                logger.warning("Tracker critical_think requested but no retriever available")
                return None
            critical_think_iter = iter_num + 1
            self._notify_progress("critical_think", critical_think_iter)
            critical_docs = retrieve_fn(pno.search_query, original_query=original_query)
            self._notify_progress("critical_search", critical_think_iter)
            seen_top_k = getattr(self, "seen_top_k", 5)
            critical_observation = passages2string(critical_docs[:seen_top_k])
            if self._is_verbose:
                self._vprint(critical_think_iter, "critical think", pno.reasoning or "(no reasoning)", sub_iter=sub_iter)
                self._vprint(critical_think_iter, "critical search", pno.search_query, sub_iter=sub_iter)
                self._vprint_docs(critical_think_iter, critical_docs[:seen_top_k], sub_iter=sub_iter)
            critical_think_decision = tracker.evaluate(
                subquery=pno.search_query, docs=critical_docs[:seen_top_k],
                original_query=original_query or "", iter_num=critical_think_iter,
                thinking=pno.reasoning, trajectory=trajectory, reasoning_path=reasoning_path,
            )
            if self._is_verbose:
                self._vprint_tracker(critical_think_iter, critical_think_decision.scores, critical_think_decision.action, sub_iter=sub_iter)
            return TrackerCriticalThinkResult(
                critical_think=pno.reasoning, critical_search_query=pno.search_query,
                critical_docs=critical_docs, critical_observation=critical_observation,
                critical_think_scores=critical_think_decision.scores, critical_think_iter=critical_think_iter,
            )

        early_stop = getattr(decision, "early_stopping_output", None)
        if early_stop is not None:
            if self._is_verbose:
                self._vprint(iter_num, "notice", f"[{decision.action}] early_stopping", sub_iter=sub_iter)
            return TrackerEarlyStopResult(reasoning=early_stop.reasoning, scores=decision.scores)

        return None

    def _execute_deferred_critical_search(
        self,
        deferred: TrackerCriticalThinkDeferred,
        original_query: str,
        trajectory: Any = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[TrackerCriticalThinkResult]:
        """Execute a deferred critical search and return the full result."""
        retrieve_fn = getattr(self, "retrieve_documents", None)
        if retrieve_fn is None:
            logger.warning("Tracker critical_think requested but no retriever available")
            return None
        tracker = getattr(self, "controller", None)
        critical_think_iter = deferred.iter_num + 1
        self._notify_progress("critical_think", critical_think_iter)
        critical_docs = retrieve_fn(deferred.critical_search_query, original_query=original_query)
        self._notify_progress("critical_search", critical_think_iter)
        seen_top_k = getattr(self, "seen_top_k", 5)
        critical_observation = passages2string(critical_docs[:seen_top_k])
        if self._is_verbose:
            self._vprint(critical_think_iter, "critical think", deferred.critical_think or "(no reasoning)")
            self._vprint(critical_think_iter, "critical search", deferred.critical_search_query)
            self._vprint_docs(critical_think_iter, critical_docs[:seen_top_k])
        critical_think_scores = {}
        if tracker is not None:
            critical_think_decision = tracker.evaluate(
                subquery=deferred.critical_search_query, docs=critical_docs[:seen_top_k],
                original_query=original_query, iter_num=critical_think_iter,
                thinking=deferred.critical_think, trajectory=trajectory, reasoning_path=reasoning_path,
            )
            critical_think_scores = critical_think_decision.scores
            if self._is_verbose:
                self._vprint_tracker(critical_think_iter, critical_think_scores, critical_think_decision.action)
        return TrackerCriticalThinkResult(
            critical_think=deferred.critical_think, critical_search_query=deferred.critical_search_query,
            critical_docs=critical_docs, critical_observation=critical_observation,
            critical_think_scores=critical_think_scores, critical_think_iter=critical_think_iter,
        )

    def _reset_tracker(self, query_id: Optional[str] = None) -> None:
        self._search_step = 0
        tracker = getattr(self, "controller", None)
        if tracker is not None:
            tracker.reset(query_id=query_id)

    def _attach_controller_stats(self, result: dict) -> None:
        controller = getattr(self, "controller", None)
        if controller is not None:
            result["controller_score_history"] = list(controller.score_history)
            result["controller_unique_doc_ids"] = sorted(controller.unique_doc_ids)
            result["controller_unique_doc_count"] = controller.unique_doc_count
            result["controller_answer_candidates"] = list(controller.answer_candidates)


def _strip_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove tool-related content from a message list.

    Providers like Bedrock reject messages that contain tool_calls or
    role="tool" entries when no ``tools=`` parameter is supplied. This
    helper converts such messages into plain text so the conversation
    can be sent as a regular completion call (e.g. answer candidate
    generation).
    """
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool":
            sanitized.append({
                "role": "user",
                "content": f"[Search Result]\n{msg.get('content', '')}",
            })
        elif msg.get("tool_calls"):
            new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            if not new_msg.get("content"):
                new_msg["content"] = "(searching...)"
            sanitized.append(new_msg)
        else:
            sanitized.append(msg)
    return sanitized


class BasicAgent(AgentVerboseMixin):
    """Base class for retrieval agents."""

    def __init__(self, llm_client: LiteLLMClient, retriever: Optional[Any] = None, max_iteration: int = 100, seen_top_k: int = 5, search_tool=None, controller=None):
        """Initialize BasicAgent.

        Args:
            llm_client: LiteLLM client for generation
            retriever: Retriever endpoint client (optional for no-retrieval models)
            max_iteration: Maximum iterations for multi-step methods
            seen_top_k: Number of top docs passed to the next component (visible
                to the model). These docs are also considered as cited docs.
                Doc IDs for these are recorded in the trajectory.
            search_tool: Optional RetrievalSearchTool instance. When provided,
                retrieve_documents() delegates to it (supporting fusion and
                reranking). Falls back to raw retriever if not set.
            controller: Optional Controller instance. When
                provided, post_search_evaluate() computes a score after each
                search and may inject an observation into the agent's context.
        """
        self.generator = llm_client
        self.retriever = retriever
        self.search_tool = search_tool
        self.controller = controller
        self.max_iteration = max_iteration
        self.seen_top_k = seen_top_k
        self._search_step = 0

        self.inference_config: InferenceConfig = InferenceConfig()

    # Agent display name used by _display_name property; subclasses can override.
    AGENT_NAME: str = "Agent"

    def _token_meter(self):
        """Return the active :class:`TokenMeter`, or None if unavailable.

        Prefers an agent-owned ``self.token_meter`` (vendor agents that hold
        their own provider client) and falls back to the generator's meter
        (LiteLLM/HF-backed agents using ``self.generator``).
        """
        meter = getattr(self, "token_meter", None)
        if meter is not None:
            return meter
        for client_attr in ("generator", "_vllm_client"):
            meter = getattr(getattr(self, client_attr, None), "token_meter", None)
            if meter is not None:
                return meter
        return None

    def _step_tokens(self):
        """Tokens consumed since the previous trajectory step (or None)."""
        meter = self._token_meter()
        return meter.since_last_step() if meter is not None else None

    def get_answer_candidate_llm(self) -> Optional[LiteLLMClient]:
        """Return a LiteLLM client for answer candidate generation.

        Self-managed agents (those with model_name / model_url attributes)
        automatically get a hosted_vllm/ LiteLLM wrapper pointing at the
        same vLLM server so the answer-candidate generator uses the same
        backbone model.  API-backed agents fall back to self.generator.
        Subclasses with non-standard routing (e.g. Bedrock) can still
        override this method.
        """
        model_name = getattr(self, "model_name", None)
        model_url = getattr(self, "model_url", None)
        if model_name and model_url:
            api_key = getattr(self, "_api_key", "EMPTY")
            return LiteLLMClient(
                model=f"hosted_vllm/{model_name}",
                api_base=model_url,
                api_key=api_key,
            )
        return self.generator

    def get_system_prompt(self) -> Optional[str]:
        """Return the agent's system prompt, or None if not set."""
        return getattr(self, "system_prompt", None) or getattr(self, "_system_prompt", None)

    # ------------------------------------------------------------------
    # Default document formatting and tool definitions
    # ------------------------------------------------------------------

    def _format_search_results(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as JSON for tool response messages."""
        return format_as_json(docs, self.seen_top_k)

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return search tool definition in the appropriate API format.

        Dispatches on ``inference_config.api_type``.  Subclasses with
        additional tools should override this method.
        """
        cfg = self.inference_config
        desc = (
            "Search for information using the search engine. "
            f"Returns top {self.seen_top_k} results."
        )
        if cfg.api_type == "responses_api":
            return [{
                "type": "function",
                "name": "search",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }]
        return [{
            "type": "function",
            "function": {
                "name": "search",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                },
            },
        }]

    # ------------------------------------------------------------------
    # Unified Responses API call (used by force answer + answer candidate)
    # ------------------------------------------------------------------

    def _make_responses_api_call(
        self,
        messages: List[Any],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens_override: Optional[int] = None,
        return_reasoning_fallback: bool = False,
    ) -> Optional[str]:
        """Make a Responses API call using ``self.inference_config``.

        Returns the text content from the first message item.  When
        *return_reasoning_fallback* is True and no message item exists,
        the reasoning summary text is returned (prefixed with
        ``[reasoning_fallback]`` so callers can distinguish it).
        Returns None only on API failure or truly empty output.
        """
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

        request: Dict[str, Any] = {
            "model": cfg.model_name,
            "max_output_tokens": max_output_tokens_override or cfg.max_output_tokens,
            "input": messages,
            "truncation": "auto",
        }
        if cfg.reasoning_effort is not None:
            request["reasoning"] = {
                "effort": cfg.reasoning_effort,
                "summary": "detailed",
            }
        if tools:
            request["tools"] = tools

        try:
            response = client.responses.create(**request)
        except Exception as e:
            logger.warning("Responses API call failed: %s", e)
            return None

        for item in response.output:
            if getattr(item, "type", None) == "message":
                return "\n".join(p.text for p in item.content)

        if return_reasoning_fallback:
            reasoning_parts = []
            for item in response.output:
                if getattr(item, "type", None) == "reasoning":
                    reasoning_parts.extend(
                        p.text for p in item.content if hasattr(p, "text")
                    )
            if reasoning_parts:
                return "[reasoning_fallback]" + "\n".join(reasoning_parts)

        return None

    # ------------------------------------------------------------------
    # Trim first iteration helper
    # ------------------------------------------------------------------

    @staticmethod
    def _msg_attr(msg: Any, key: str, default: Any = None) -> Any:
        """Get an attribute from a message, whether it's a dict or pydantic object."""
        if isinstance(msg, dict):
            return msg.get(key, default)
        return getattr(msg, key, default)

    @classmethod
    def _trim_first_iteration(cls, messages: List[Any]) -> List[Any]:
        """Remove the first search iteration from the message history.

        Returns a new list: ``[prefix] + [iter_2 … iter_L]``.
        Works for both Responses API (type-keyed items) and Chat Completions
        (role-keyed messages).  If fewer than 2 iterations exist, returns a
        copy of *messages* unchanged.
        """
        prefix_end = 0
        for i, msg in enumerate(messages):
            if cls._msg_attr(msg, "role") in ("system", "user"):
                prefix_end = i + 1
            else:
                break

        if prefix_end >= len(messages):
            return list(messages)

        first_item = messages[prefix_end]

        if cls._msg_attr(first_item, "type") is not None:
            seen_tool_output = False
            for i in range(prefix_end, len(messages)):
                if cls._msg_attr(messages[i], "type") == "function_call_output":
                    seen_tool_output = True
                elif seen_tool_output:
                    return messages[:prefix_end] + messages[i:]
        else:
            assistant_count = 0
            for i in range(prefix_end, len(messages)):
                if cls._msg_attr(messages[i], "role") == "assistant":
                    assistant_count += 1
                    if assistant_count == 2:
                        return messages[:prefix_end] + messages[i:]

        return list(messages)

    # ------------------------------------------------------------------
    # Shared force-answer (Responses API agents)
    # ------------------------------------------------------------------

    def _force_answer_responses_api_in_conversation(
        self,
        messages: List[Any],
    ) -> Optional[str]:
        """Append force-answer instruction to conversation and call the API.

        Uses ``inference_config`` for model, max_output_tokens, reasoning_effort.
        Trims the first iteration to free context space for the instruction.
        """
        cfg = self.inference_config
        trimmed = self._trim_first_iteration(messages)
        trimmed.append({
            "role": "user",
            "content": f"{FINAL_ANSWER_INSTRUCTION}\n\n{cfg.format_instructions}",
        })
        return self._make_responses_api_call(trimmed)

    def _force_answer_responses_api_compressed(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Standalone prompt with compressed evidence — last-resort fallback.

        Uses ``inference_config`` for model, max_output_tokens, reasoning_effort.
        """
        cfg = self.inference_config
        prompt = self._build_force_answer_prompt(
            query, reasoning_path, cfg.format_instructions,
        )
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._make_responses_api_call(messages)

    # ------------------------------------------------------------------
    # Shared force-answer (Chat Completions agents)
    # ------------------------------------------------------------------

    def _force_answer_chat_in_conversation(
        self,
        messages: List[Any],
    ) -> Optional[str]:
        """Append force-answer instruction to conversation and call Chat Completions API.

        Uses ``inference_config`` for model, api_base, api_key, max_output_tokens.
        Trims the first iteration to free context space for the instruction.
        """
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        trimmed = self._trim_first_iteration(messages)
        trimmed.append({
            "role": "user",
            "content": f"{FINAL_ANSWER_INSTRUCTION}\n\n{cfg.format_instructions}",
        })
        force_max_tokens = min(cfg.max_output_tokens, 4096)
        try:
            response = client.chat.completions.create(
                model=cfg.model_name,
                messages=trimmed,
                max_tokens=force_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Force answer (chat, in conversation) failed: %s", e)
            return None

    def _force_answer_chat_compressed(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Standalone prompt with compressed evidence — Chat Completions fallback.

        Uses ``inference_config`` for model, api_base, api_key, max_output_tokens.
        """
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        prompt = self._build_force_answer_prompt(
            query, reasoning_path, cfg.format_instructions,
        )
        messages: List[Dict[str, str]] = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        force_max_tokens = min(cfg.max_output_tokens, 4096)
        try:
            response = client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                max_tokens=force_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Force answer (chat, compressed) failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Answer candidate generation (shared across all agents)
    # ------------------------------------------------------------------

    def generate_answer_candidate(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        seen_top_k: int = 5,
    ) -> List[AnswerCandidateOutput]:
        """Generate answer candidates using the same config as the main loop.

        For Responses API agents: appends instruction to the live conversation
        and makes one more API call (full context preserved).

        For chat-completion agents: builds an evidence-summary prompt and calls
        the LLM client.
        """
        cfg = self.inference_config

        if cfg.api_type == "responses_api":
            return self._generate_answer_candidate_responses_api(
                original_query, trajectory, seen_top_k,
            )
        return self._generate_answer_candidate_chat(
            original_query, trajectory, reasoning_path, seen_top_k,
        )

    def _generate_answer_candidate_responses_api(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None],
        seen_top_k: int,
    ) -> List[AnswerCandidateOutput]:
        cfg = self.inference_config

        if not isinstance(trajectory, list):
            logger.warning(
                "generate_answer_candidate (responses_api) requires a message list. "
                "Got %s; skipping.", type(trajectory).__name__,
            )
            return [AnswerCandidateOutput(
                candidate="no candidate",
                reasoning="trajectory is not a Responses API message list",
            )]

        messages = copy.deepcopy(trajectory)
        instruction = (
            f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
            f"{cfg.format_instructions}"
        )
        messages.append({"role": "user", "content": instruction})

        raw = self._make_responses_api_call(messages, return_reasoning_fallback=True)

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
                "Answer candidate (Responses API) returned text but no candidates. "
                "Raw (first 300 chars): %s", raw[:300],
            )
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning=raw.strip(),
            )]
        return candidates

    def _generate_answer_candidate_chat(
        self,
        original_query: str,
        trajectory: Union[str, List[Dict[str, Any]], None],
        reasoning_path: Optional[List[Dict[str, Any]]],
        seen_top_k: int,
    ) -> List[AnswerCandidateOutput]:
        cfg = self.inference_config

        if not isinstance(trajectory, list):
            logger.warning(
                "generate_answer_candidate (chat) requires a message list. "
                "Got %s; skipping.", type(trajectory).__name__,
            )
            return [AnswerCandidateOutput(
                candidate="no candidate",
                reasoning="trajectory is not a message list",
            )]

        messages = _strip_tool_messages(copy.deepcopy(trajectory))
        instruction = (
            f"{CANDIDATE_GENERATION_INSTRUCTION}\n\n"
            f"{cfg.format_instructions}"
        )
        messages.append({"role": "user", "content": instruction})

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
                max_tokens=cfg.max_output_tokens,
            )
        except Exception:
            logger.warning("Answer candidate LLM call failed", exc_info=True)
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
                "Answer candidate LLM returned text but no candidates. "
                "Raw (first 300 chars): %s", raw[:300],
            )
            return [AnswerCandidateOutput(
                candidate="no candidate", reasoning=raw.strip(),
            )]
        return candidates

    def _notify_progress(self, stage: str, iteration: int) -> None:
        """Report current iteration and stage to the progress bar callback."""
        cb = getattr(self, "_status_callback", None)
        if cb is not None:
            cb(stage, iteration)

    # ------------------------------------------------------------------
    # Common run interface (shared by all reasoning agents)
    # ------------------------------------------------------------------

    def run_single(self, query_id: str, query_text: str, temperature: float = 0.7, status_callback=None) -> Optional[Dict[str, Any]]:
        """Process a single query and return its normalised result dict.

        Args:
            query_id:   Identifier for the query.
            query_text: The query string.
            temperature: Sampling temperature passed to inference().
            status_callback: Optional callable(stage: str, iteration: int) invoked at
                             each retrieval step.  Used to stream live progress to the
                             parent process's tqdm bars in multi-GPU runs.

        Returns:
            result_dict with keys: query, generation, num_steps, num_searches, trajectory.
            Returns None on error.

        Note:
            Supports inference() returning either 2 values (reasoning_path, prediction)
            or 3 values (reasoning_path, prediction, num_iterations).
        """
        self._status_callback = status_callback
        self._search_iter     = 0
        # Reset controller for this query (loads per-query qrels)
        self._reset_tracker(query_id=query_id)
        # Snapshot token usage so we can report per-query totals.
        _meter = self._token_meter()
        _tok_start = _meter.snapshot() if _meter is not None else None
        if _meter is not None:
            _meter.since_last_step()  # clear per-step cursor for this query
        try:
            inference_result = self.inference(query_text, generation_temp=temperature)

            # Support both 2-value and 3-value returns from inference()
            if len(inference_result) == 3:
                reasoning_path, prediction, num_iterations = inference_result
            else:
                reasoning_path, prediction = inference_result
                num_iterations = None

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
            }
            if num_iterations is not None:
                result["num_iterations"] = num_iterations

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

            if num_iterations is not None:
                logger.info(
                    f"  ✓ {num_iterations} iters, {num_searches} searches, "
                    f"{len(reasoning_path)} steps"
                )
            else:
                logger.info(f"  ✓ {num_searches} searches, {len(reasoning_path)} steps")
            return result

        except Exception as e:
            logger.error(f"  ✗ Error processing {query_id}: {e}")
            traceback.print_exc()
            return None
        finally:
            self._status_callback = None
            self._search_iter     = 0

    def cleanup(self):
        """Release any resources held by the agent."""
        pass

    # ------------------------------------------------------------------
    # Helpers used by subclasses
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shared tool-call loop helpers (GLM, OSS)
    # ------------------------------------------------------------------

    def _build_tool_response_message(
        self,
        call_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Build a tool response message in the appropriate API format.

        Chat Completions API → ``{"role": "tool", "tool_call_id": ..., "content": ...}``
        Responses API        → ``{"type": "function_call_output", "call_id": ..., "output": ...}``
        """
        cfg = self.inference_config
        if cfg.api_type == "responses_api":
            return {"type": "function_call_output", "call_id": call_id, "output": content}
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def _handle_early_stop_phase(
        self,
        early_stop: bool,
        notified: bool,
        messages: List[Any],
    ) -> tuple:
        """Manage the two-phase early stop state machine.

        Phase 1 (early_stop=True, notified=False): append force-answer
        instruction and set notified=True.
        Phase 2 (notified=True): signal the caller to break.

        Returns:
            (should_break, notified)
        """
        if early_stop and not notified:
            cfg = self.inference_config
            if messages and messages[-1].get("role") == "tool":
                messages.append({
                    "role": "assistant",
                    "content": "I have reviewed the search results. Let me now provide my final answer.",
                })
            messages.append({
                "role": "user",
                "content": f"{FINAL_ANSWER_INSTRUCTION}\n\n{cfg.format_instructions}",
            })
            return False, True
        if notified:
            return True, True
        return False, False

    def _evaluate_and_handle_tracker(
        self,
        all_subqueries: List[str],
        all_seen_docs: List[Dict[str, Any]],
        iteration: int,
        query: str,
        cur_reasoning: Optional[str],
        messages: List[Any],
        reasoning_path: List[Dict[str, Any]],
        last_search_msg_idx: Optional[int],
    ) -> tuple:
        """Post-search tracker evaluation with aggregated data.

        Handles both early-stop and critical-think results. For critical
        think, appends redirect text to the last search result message and
        records the entry in reasoning_path.

        Returns:
            (early_stop_triggered: bool, extra_iterations: int)
        """
        seen_docs_dedup = self.get_unique_docs(all_seen_docs)
        tracker_result = self.post_search_evaluate(
            subquery=all_subqueries,
            docs=seen_docs_dedup,
            iter_num=iteration, original_query=query,
            thinking=cur_reasoning or "",
            seen_docs=None,
            trajectory=messages,
            reasoning_path=reasoning_path,
        )
        if isinstance(tracker_result, TrackerEarlyStopResult):
            return True, 0
        if isinstance(tracker_result, TrackerCriticalThinkResult):
            if last_search_msg_idx is not None:
                msg = messages[last_search_msg_idx]
                content_key = "output" if "output" in msg else "content"
                msg[content_key] += self._format_critical_redirect_text(tracker_result)
            ct_entry = self._critical_think_to_reasoning_entry(
                tracker_result, include_all_docs=True,
            )
            ct_entry["iteration"] = tracker_result.critical_think_iter
            reasoning_path.append(ct_entry)
            return False, 1
        return False, 0

    # ------------------------------------------------------------------
    # Per-query tracker helpers
    # ------------------------------------------------------------------

    def _track_query(
        self,
        search_query: str,
        seen_docs: List[Dict[str, Any]],
        original_query: str,
        cur_reasoning: Optional[str],
        messages: List[Any],
        reasoning_path: List[Dict[str, Any]],
    ) -> tuple:
        """Evaluate the tracker for a single search query.

        Increments ``_search_step`` and calls ``post_search_evaluate`` with
        ``defer_critical_search=True`` so that an "intervene" decision
        returns a ``TrackerCriticalThinkDeferred`` without executing
        retrieval.  The actual critical search is executed later by
        ``_apply_tracker_action`` after all sub-queries have completed.

        Returns:
            (tracker_result, stop_evaluating) where *tracker_result* is the
            actionable result (``TrackerCriticalThinkDeferred``,
            ``TrackerEarlyStopResult``, or ``None``) and *stop_evaluating*
            is ``True`` when the controller issued a non-continue action
            (even if no actionable result was produced).
        """
        self._search_step += 1
        tracker_result = self.post_search_evaluate(
            subquery=search_query,
            docs=seen_docs,
            iter_num=self._search_step,
            original_query=original_query,
            thinking=cur_reasoning or "",
            seen_docs=None,
            trajectory=messages,
            reasoning_path=reasoning_path,
            defer_critical_search=True,
        )

        stop_evaluating = tracker_result is not None
        if not stop_evaluating:
            tracker = getattr(self, "controller", None)
            if tracker is not None and tracker.score_history:
                last_action = tracker.score_history[-1].get("controller_action")
                if last_action and last_action != "continue":
                    stop_evaluating = True

        return tracker_result, stop_evaluating

    def _apply_tracker_action(
        self,
        tracker_result: Union[TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult],
        messages: List[Any],
        reasoning_path: List[Dict[str, Any]],
        last_search_msg_idx: Optional[int],
        original_query: Optional[str] = None,
    ) -> bool:
        """Apply a deferred tracker action after the tool-call loop.

        For deferred critical-think: executes the critical search retrieval,
        then injects redirect text and appends the entry to reasoning_path.
        For critical-think: injects redirect text into the last search
        result message and appends the entry to reasoning_path.
        For early-stop: signals the caller to stop.

        Returns True if early stop was triggered.
        """
        if isinstance(tracker_result, TrackerEarlyStopResult):
            return True
        if isinstance(tracker_result, TrackerCriticalThinkDeferred):
            tracker_result = self._execute_deferred_critical_search(
                tracker_result, original_query or "",
                trajectory=messages, reasoning_path=reasoning_path,
            )
            if tracker_result is None:
                return False
            self._search_step += 1
        if isinstance(tracker_result, TrackerCriticalThinkResult):
            if last_search_msg_idx is not None:
                msg = messages[last_search_msg_idx]
                content_key = "output" if "output" in msg else "content"
                msg[content_key] += self._format_critical_redirect_text(tracker_result)
            ct_entry = self._critical_think_to_reasoning_entry(
                tracker_result, include_all_docs=True,
            )
            ct_entry["iteration"] = tracker_result.critical_think_iter
            reasoning_path.append(ct_entry)
            return False
        return False

    # ------------------------------------------------------------------
    # Other helpers
    # ------------------------------------------------------------------

    def get_unique_docs(self, docs_lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate documents based on doc_id."""
        return list({doc['doc_id']: doc for doc in docs_lst}.values())

    def get_think(self, text: str) -> Optional[str]:
        """Extract thinking process from <think> tags."""
        return _get_think(text)

    def get_query(self, text: str) -> Optional[str]:
        """Extract search query from <search> tags."""
        return _get_query(text)

    def get_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer> tags."""
        return _get_answer(text)

    def _rebuild_step_text(self, step: Dict[str, Any]) -> str:
        """Reconstruct prompt text for one reasoning step from structured data.

        Default uses ``<think>/<search>/<information>`` tags (SearchR1/StepSearch).
        Override in subclasses for agents with different prompt formats.
        """
        think = step.get("think", "")
        sq = step.get("search_query", step.get("query", ""))
        docs = step.get("docs", [])
        is_stripped = step.get("_docs_stripped", False)
        seen_top_k = getattr(self, "seen_top_k", 5)

        output_text = ""
        if think:
            output_text += f"<think>{think}</think>\n"
        if sq:
            output_text += f"<search>{sq}</search>"

        if is_stripped:
            search_results = "(earlier search results omitted for brevity)"
        else:
            search_results = passages2string(docs[:seen_top_k])

        return f"\n\n{output_text}<information>{search_results}</information>\n\n"

    def _rebuild_windowed_prompt(
        self,
        initial_prompt: str,
        reasoning_path: List[Dict[str, Any]],
        answer_nudge: str,
        keep_last: int = 3,
    ) -> str:
        """Rebuild a windowed conversation prompt from reasoning_path.

        Reconstructs the agent's conversation with full document context for
        recent search turns and stripped context for older turns, then appends
        the answer nudge.  Same pattern as early stopping (real conversation
        + nudge) but with reduced context to fit within the limit.
        """
        pruned = reduce_reasoning_path(reasoning_path, keep_last=keep_last)

        prompt = initial_prompt
        for step in pruned:
            if step.get("search_query") or step.get("query"):
                prompt += self._rebuild_step_text(step)

        prompt += answer_nudge
        return prompt

    def _build_force_answer_prompt(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
        format_instructions: str,
    ) -> str:
        """Build a compact prompt for forcing a final answer.

        Reusable across agents that handle their own LLM calls (e.g. GLM,
        OSS).  Agents using ``self.generator`` should prefer
        ``_force_answer_on_context_limit`` which wraps this.
        """
        evidence = build_evidence_summary(reasoning_path, seen_top_k=self.seen_top_k)
        return (
            f"{FINAL_ANSWER_INSTRUCTION}\n\n"
            f"Question: {query}\n\n"
            f"Evidence collected:\n{evidence}\n\n"
            f"{format_instructions}"
        )

    def _force_answer_on_context_limit(
        self,
        query: str,
        reasoning_path: List[Dict[str, Any]],
        format_instructions: str,
        extract_fn: Callable[[Optional[str]], Optional[str]],
        *,
        generation_temp: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
        generator_kwargs: Optional[Dict[str, Any]] = None,
        initial_prompt: Optional[str] = None,
        answer_nudge: Optional[str] = None,
    ) -> Optional[str]:
        """Force a final answer when context limit is reached.

        Two modes depending on whether ``initial_prompt`` is provided:

        **Windowed mode** (``initial_prompt`` set): Rebuilds the conversation
        from ``initial_prompt`` + pruned ``reasoning_path`` using
        ``_rebuild_windowed_prompt``, then appends the answer nudge.

        **Summary mode** (``initial_prompt`` is ``None``): Falls back to
        ``_build_force_answer_prompt`` which builds a standalone prompt with
        a windowed evidence summary.

        Returns the extracted answer string, or ``None`` on failure.
        """
        self._print("Context limit hit, forcing final answer from collected evidence")

        if initial_prompt is not None:
            nudge = answer_nudge if answer_nudge is not None else f"\n{format_instructions}"
            prompt = self._rebuild_windowed_prompt(
                initial_prompt, reasoning_path, answer_nudge=nudge,
            )
        else:
            prompt = self._build_force_answer_prompt(query, reasoning_path, format_instructions)

        messages: List[Dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if assistant_prefill is not None:
            messages.append({"role": "assistant", "content": assistant_prefill})

        try:
            kwargs: Dict[str, Any] = {"temperature": generation_temp, "max_tokens": max_tokens}
            if generator_kwargs:
                kwargs.update(generator_kwargs)
            forced_output = self.generator.complete(messages, **kwargs)
            answer = extract_fn(forced_output)
            self._vprint(-1, "forced_answer", (answer or "(no answer)")[:300])
            reasoning_path.append({
                'action_type': 'context_limit',
                'think': FINAL_ANSWER_INSTRUCTION,
                'prediction': answer,
            })
            return answer
        except Exception as e2:
            logger.warning(f"Forced final answer also failed: {e2}")
            return None

    def retrieve_documents(self, query: str, *, original_query: Optional[str] = None, reasoning: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for a query.

        Delegates to ``self.search_tool`` when available (supports fusion
        and reranking).  Falls back to the raw retriever otherwise.

        Args:
            query: The search query string.
            original_query: The original user query, forwarded to the search
                tool so that ``retrieval_input`` / ``post_fusion_reranker_input``
                composition modes can combine it with sub-queries.
            reasoning: Optional trajectory context forwarded to the search
                tool for use with ``retrieval_input`` / ``post_fusion_reranker_input``
                composition modes.
        """
        self._search_iter = getattr(self, "_search_iter", 0) + 1

        if self.search_tool is not None:
            return self.search_tool.execute(query, original_query=original_query, reasoning=reasoning)

        if not self.retriever:
            return []

        results = self.retriever.retrieve(query)
        return normalize_retrieval_response(results)


class TagReasoningAgent(BasicAgent):
    """Base for reasoning agents using a think/search/answer tag loop.

    Covers the shared inference pattern used by SearchR1, StepSearch, and
    ReSearch.  Subclasses configure prompt formatting and answer detection
    via constructor args and optional method overrides.

    Configuration points:
        _format_initial_prompt(question) — build the initial user prompt.
        _build_messages(input_prompt)    — wrap prompt into message list.
        _has_answer(output_text)         — detect whether output contains a final answer.
        _extract_prediction(output_text) — pull the prediction from the answer output.
        curr_step_template               — format string for appending search steps.
        _system_prompt                   — optional system message (None = omit).
    """

    AGENT_NAME = "TagReasoning"

    def __init__(self, llm_client: LiteLLMClient, retriever: Optional[Any] = None, max_iteration: int = 100, seen_top_k: int = 5, verbose: bool = True):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self._system_prompt: Optional[str] = None
        self.curr_step_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

        self.inference_config = InferenceConfig(
            api_type="chat_completion",
        )

    # -- Configuration points (override in subclasses) --

    def _format_initial_prompt(self, question: str) -> str:
        raise NotImplementedError

    def _build_messages(self, input_prompt: str) -> List[Dict[str, str]]:
        if self._system_prompt:
            return [
                {'role': 'system', 'content': self._system_prompt},
                {'role': 'user', 'content': input_prompt},
            ]
        return [{'role': 'user', 'content': input_prompt}]

    def _has_answer(self, output_text: str) -> bool:
        return '</answer>' in output_text

    def _extract_prediction(self, output_text: str) -> Optional[str]:
        return self.get_answer(output_text)

    # -- Shared inference loop --

    def inference(self, question: str, generation_temp: float = 0.7) -> tuple:
        input_prompt = self._format_initial_prompt(question)
        messages = self._build_messages(input_prompt)

        reasoning_path: List[Dict[str, Any]] = []
        early_stop_result = None
        for iter_idx in range(self.max_iteration):
            iter_num = iter_idx
            self._notify_progress("think", iter_num)
            try:
                output_text = self.generator.complete(messages, temperature=generation_temp)
            except Exception as e:
                logger.warning(f"Iteration {iter_num} API error: {e}")
                self._force_answer_on_context_limit(
                    question, reasoning_path,
                    f"{TAG_FORMAT}\n<answer>",
                    lambda out: (out or "").split("</answer>")[0].strip(),
                    generation_temp=generation_temp,
                    system_prompt=self._system_prompt,
                    initial_prompt=self._format_initial_prompt(question),
                    answer_nudge=f"\n{FINAL_ANSWER_INSTRUCTION}\n<answer>",
                )
                break

            if self._has_answer(output_text):
                one_step_think = self.get_think(output_text)
                prediction = self._extract_prediction(output_text)
                self._vprint(iter_num, "think", one_step_think or "(no think)")
                self._notify_progress("answer", iter_num)
                self._vprint(iter_num, "answer", prediction or "(no answer)")
                reasoning_path.append({'think': one_step_think, 'prediction': prediction, 'tokens': self._step_tokens()})
                break

            tmp_query = self.get_query(output_text)
            think_text = self.get_think(output_text)
            self._vprint(iter_num, "think", think_text or "(no think)")

            if tmp_query:
                self._notify_progress("search", iter_num)
                self._vprint(iter_num, "search", tmp_query)
                search_docs = self.retrieve_documents(
                    tmp_query,
                    original_query=question,
                    reasoning=think_text if think_text else None,
                )
                search_results = passages2string(search_docs[:self.seen_top_k])

                seen_docs = search_docs[:self.seen_top_k]
                tracker_result = self.post_search_evaluate(
                    subquery=tmp_query, docs=seen_docs,
                    iter_num=iter_num, original_query=question,
                    thinking=think_text or "",
                    seen_docs=seen_docs,
                    trajectory=input_prompt,
                    reasoning_path=reasoning_path,
                )
                if isinstance(tracker_result, TrackerEarlyStopResult):
                    early_stop_result = tracker_result
            else:
                search_docs, search_results = [], ''
                tracker_result = None

            reasoning_path.append({
                'think': think_text,
                'search_query': tmp_query,
                'docs': search_docs,
                'component_doc_ids': [d.get('doc_id', '') for d in search_docs[:self.seen_top_k]],
                'tokens': self._step_tokens(),
            })

            search_text = self.curr_step_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text

            if early_stop_result is not None:
                break

            if isinstance(tracker_result, TrackerCriticalThinkResult):
                reasoning_path.append(self._critical_think_to_reasoning_entry(tracker_result))
                critical_think_output = (
                    f"<think>{tracker_result.critical_think}</think>\n"
                    f"<search>{tracker_result.critical_search_query}</search>"
                )
                critical_think_text = self.curr_step_template.format(
                    output_text=critical_think_output,
                    search_results=tracker_result.critical_observation,
                )
                input_prompt += critical_think_text
            messages = self._build_messages(input_prompt)

        prediction = reasoning_path[-1].get('prediction') if reasoning_path else None

        if not prediction:
            action_type = 'early_stop' if early_stop_result is not None else 'max_iter_force'
            force_input = input_prompt + (
                f"\n{FINAL_ANSWER_INSTRUCTION} {TAG_FORMAT}\n<answer>"
            )
            force_messages = self._build_messages(force_input)
            force_output = self.generator.complete(force_messages, temperature=generation_temp)
            prediction = (force_output or "").split("</answer>")[0].strip()
            self._vprint(iter_num, "think", FINAL_ANSWER_INSTRUCTION)
            self._vprint(iter_num, "finish", prediction or "(no answer)")
            reasoning_path.append({
                'action_type': action_type,
                'think': FINAL_ANSWER_INSTRUCTION,
                'prediction': prediction,
                'tokens': self._step_tokens(),
            })

        num_iterations = iter_idx + 1 if reasoning_path else 0
        return reasoning_path, prediction, num_iterations
