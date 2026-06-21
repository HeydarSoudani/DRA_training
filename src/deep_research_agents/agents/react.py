"""Unified ReAct agent with optional planning.

LLM calls  : LiteLLM client (``self.generator.complete()``)
Retrieval  : pipeline local retriever (``self.retrieve_documents``)
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from utils.llm_client import LiteLLMClient

_PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "react"
REACT_SYSTEM_PROMPT = (_PROMPT_DIR / "system.txt").read_text()
REACT_WITH_PLAN_SYSTEM_PROMPT = (_PROMPT_DIR / "system_with_plan.txt").read_text()

logger = logging.getLogger(__name__)

from .base_agent import BasicAgent
from agent_tools.react_tools import PlanTool
from controller_component import TrackerCriticalThinkResult, TrackerEarlyStopResult
from controller_component.prompts.answer_prompts import FINAL_ANSWER_INSTRUCTION, REACT_FORMAT
from utils.text_utils import passages2string
from utils.config import InferenceConfig
from utils.text_utils import get_action, parse_action_call, extract_think_and_clean


class ReActAgent(BasicAgent):
    """Unified ReAct agent. Plan is an optional component.

    use_plan=False: Think -> Search -> Observe -> ... -> Finish
    use_plan=True:  Plan(create) -> Think -> Search -> Observe -> Plan(update) -> ... -> Finish
                    Plan calls are hard-coded (not LLM-chosen actions).

    Search uses search_tool.execute() when available, else retrieve_documents().
    """

    AGENT_NAME = "ReAct"

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        max_iteration: int = 100,
        seen_top_k: int = 5,
        use_plan: bool = False,
        verbose: bool = True,
    ):
        super().__init__(llm_client, retriever, max_iteration, seen_top_k)
        self.verbose = verbose
        self.use_plan = use_plan

        _sys_prompt = REACT_WITH_PLAN_SYSTEM_PROMPT if use_plan else REACT_SYSTEM_PROMPT
        self.inference_config = InferenceConfig(
            api_type="chat_completion",
            system_prompt=_sys_prompt,
            format_instructions=REACT_FORMAT,
        )

        self.current_step_template = (
            "<think>{think}</think>\n"
            "<action>{action_text}</action>\n"
            "<observation>{observation}</observation>\n"
        )

        # Tools
        if use_plan:
            self.plan_tool = PlanTool(llm_client)

        # Per-query state (reset in inference())
        self.query: Optional[str] = None
        self.retrieved_docs: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.last_search_query: Optional[str] = None
        self.last_search_results: Optional[List[Dict[str, Any]]] = None

    def _rebuild_step_text(self, step):
        think = step.get("think", "")
        sq = step.get("search_query", step.get("query", ""))
        docs = step.get("docs", [])
        is_stripped = step.get("_docs_stripped", False)

        action_text = f'search(query="{sq}")'
        if is_stripped:
            observation = "(earlier search results omitted for brevity)"
        else:
            observation = passages2string(docs[:self.seen_top_k])

        return self.current_step_template.format(
            think=think, action_text=action_text, observation=observation,
        ) + "\n"

    # ------------------------------------------------------------------
    # Observation dispatch
    # ------------------------------------------------------------------

    def get_observation(self, action_type: str, action_entity: str) -> tuple:
        """Execute an action and return its observation.

        Tracker evaluation is handled by the main loop, not here.

        Returns:
            (docs, obs, done)
        """
        done, docs, obs = False, None, None

        if action_type == "search":
            search_query = action_entity
            if self.search_tool is not None:
                docs = self.search_tool.execute(
                    search_query,
                    original_query=self.query,
                )
            else:
                docs = self.retrieve_documents(search_query, original_query=self.query)
            self.retrieved_docs.extend(docs)
            obs = passages2string(docs[:self.seen_top_k])

            self.last_search_query = search_query
            self.last_search_results = docs

        elif action_type == "finish":
            done = True
            obs = action_entity

        else:
            obs = f"Invalid action type: {action_type}. Use the search or finish tool."

        return docs, obs, done

    # ------------------------------------------------------------------
    # History context (used when use_plan=True)
    # ------------------------------------------------------------------

    def _build_recent_history_context(self) -> str:
        """Build prompt context from history.

        Strategy (plan mode):
        - Include current plan at the top.
        - Include ALL Search actions (including critical_search) and observations.
        """
        parts = []

        if self.use_plan:
            current_plan = self.plan_tool.get_current_plan()
            if current_plan:
                parts.append(f"Current Research Plan:\n{current_plan}\n")

        for entry in self.history:
            if entry["action"] in ("search", "critical_search"):
                action_text = f'search(query="{entry["value"]}")'
                parts.append(
                    self.current_step_template.format(
                        think=entry["thought"],
                        action_text=action_text,
                        observation=entry["observation"],
                    )
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Force answer helpers
    # ------------------------------------------------------------------

    def _extract_forced_conclusion(self, output: Optional[str]) -> Optional[str]:
        """Extract a conclusion from a forced-answer LLM response."""
        if not output:
            return None
        action_raw = get_action(output)
        if action_raw is None:
            _, action_raw = extract_think_and_clean(output)
        parsed = parse_action_call(action_raw)
        if parsed and parsed[0] == "finish":
            return parsed[1]
        return self.get_think(output) or output.strip()

    # ------------------------------------------------------------------
    # Main inference loop
    # ------------------------------------------------------------------

    def inference(self, query: str, generation_temp: float = 0.7) -> Tuple[List[Dict[str, Any]], str, int]:
        """Run the ReAct loop.

        Returns:
            reasoning_path  : list of per-step dicts (pipeline-compatible)
            prediction      : extracted answer string
            num_iterations  : number of LLM iterations used
        """
        # Reset per-query state
        self.query = query
        self.retrieved_docs = []
        self.history = []
        self.last_search_query = None
        self.last_search_results = None
        if self.use_plan:
            self.plan_tool.reset()

        cfg = self.inference_config

        reasoning_path: List[Dict[str, Any]] = []
        conclusion: Optional[str] = None

        base_prompt = cfg.system_prompt + f"\nUser Query: {query}\n\n"
        input_prompt = base_prompt  # used in non-plan mode (linear accumulation)

        # Hard-coded: create initial plan before the loop
        if self.use_plan:
            self._vprint(0, "plan", "mode=create")
            plan_result = self.plan_tool.execute(mode="create", query=query)
            plan_obs = plan_result["observation"]
            self._vprint(0, "plan-output", (plan_obs or "")[:500])
            reasoning_path.append({
                "think": "",
                "action_type": "plan",
                "plan_mode": "create",
                "observation": plan_obs,
            })

        consecutive_parse_errors = 0
        max_consecutive_parse_errors = 3
        early_stop_result: Optional[TrackerEarlyStopResult] = None

        self._print(f"Query: {query}")

        for iter_num in range(self.max_iteration):
            display_iter = iter_num + 1
            self._current_iter = display_iter
            self._notify_progress("think", display_iter)

            # Build the prompt
            if self.use_plan:
                prompt = base_prompt + self._build_recent_history_context()
            else:
                prompt = input_prompt

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ]

            # Generate thought + action (text-based)
            try:
                output_text = self.generator.complete(
                    messages,
                    temperature=generation_temp,
                    max_tokens=800,
                    strip_think=False,
                )
            except Exception as e:
                logger.warning(f"Iteration {display_iter} API error: {e}")
                conclusion = self._force_answer_on_context_limit(
                    query, reasoning_path,
                    REACT_FORMAT,
                    self._extract_forced_conclusion,
                    generation_temp=generation_temp,
                    system_prompt="",
                    generator_kwargs={"strip_think": False},
                    initial_prompt=base_prompt,
                    answer_nudge=f"\n{FINAL_ANSWER_INSTRUCTION} {REACT_FORMAT}",
                )
                break

            # Extract thought from <think> tags
            think_content = self.get_think(output_text)
            if think_content:
                thought = think_content.replace("\n", " ").strip()
            else:
                thought = output_text.replace("\n", " ").strip()

            # Extract action: try <action> tag first, then look in cleaned text
            action_raw = get_action(output_text)
            if action_raw is None:
                _, cleaned = extract_think_and_clean(output_text)
                action_raw = cleaned

            # Parse the action call (e.g. search(query="...") or finish(answer="..."))
            parsed = parse_action_call(action_raw)

            if parsed is None:
                consecutive_parse_errors += 1
                parse_error_obs = (
                    "Could not parse your action. "
                    "You MUST output <action>search(query=\"...\")</action> or "
                    "<action>finish(answer=\"...\")</action>."
                )
                self._vprint(display_iter, "parse-error", output_text[:200])
                self.history.append({
                    "iter": display_iter,
                    "thought": thought,
                    "action": "parse_error",
                    "value": "",
                    "observation": parse_error_obs,
                })
                if not self.use_plan:
                    input_prompt += self.current_step_template.format(
                        think=thought or "(parse error)",
                        action_text="(parse error)",
                        observation=parse_error_obs,
                    ) + "\n"

                if consecutive_parse_errors >= max_consecutive_parse_errors:
                    self._vprint(display_iter, "force-finish", "too many consecutive parse errors")
                    break

                continue

            consecutive_parse_errors = 0
            action_type, action_entity = parsed

            # Build action_text for history display
            if action_type == "search":
                action_text = f'search(query="{action_entity}")'
            else:
                action_text = f'finish(answer="{action_entity}")'

            # Verbose output: print think/action BEFORE executing
            self._vprint(display_iter, "think", thought or "(no thought)")
            self._current_thought = thought or ""
            if action_type == "search":
                self._notify_progress("search", display_iter)
                self._vprint(display_iter, "search", action_entity)
            elif action_type == "finish":
                self._notify_progress("finish", display_iter)
                self._vprint(display_iter, "finish", action_entity or "(no conclusion)")

            docs, obs, done = self.get_observation(action_type, action_entity)
            obs = obs.replace("\n", " ").strip() if obs else None

            # Build history entry
            history_entry: Dict[str, Any] = {
                "iter": display_iter,
                "thought": thought,
                "action": action_type,
                "value": action_entity,
                "observation": obs,
            }

            # ---- action-specific bookkeeping ----

            if action_type == "search":
                if docs is not None:
                    seen_docs = docs[:self.seen_top_k]
                    component_doc_ids = [d.get("doc_id", "") for d in seen_docs]
                    history_entry["docs"] = docs
                    history_entry["component_doc_ids"] = component_doc_ids
                    history_entry["num_docs"] = len(docs)

                    reasoning_path.append({
                        "think": thought,
                        "action_type": action_type,
                        "query": action_entity,
                        "docs": docs,
                        "component_doc_ids": component_doc_ids,
                    })

                # Trajectory tracker: evaluate after each search
                if docs is not None:
                    tracker_result = self.post_search_evaluate(
                        subquery=action_entity,
                        docs=docs[:self.seen_top_k],
                        iter_num=display_iter,
                        original_query=self.query,
                        thinking=thought or "",
                        seen_docs=docs[:self.seen_top_k],
                        trajectory=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": input_prompt},
                        ],
                        reasoning_path=reasoning_path,
                    )
                    if isinstance(tracker_result, TrackerEarlyStopResult):
                        early_stop_result = tracker_result
                        done = True
                    elif isinstance(tracker_result, TrackerCriticalThinkResult):
                        self.retrieved_docs.extend(tracker_result.critical_docs)
                        self.history.append({
                            "iter": tracker_result.critical_think_iter,
                            "thought": tracker_result.critical_think,
                            "action": "critical_search",
                            "value": tracker_result.critical_search_query,
                            "observation": tracker_result.critical_observation,
                            "is_critical_think": True,
                        })
                        ct_entry = self._critical_think_to_reasoning_entry(
                            tracker_result, include_all_docs=True,
                        )
                        ct_entry["iteration"] = tracker_result.critical_think_iter
                        reasoning_path.append(ct_entry)
                        if not self.use_plan:
                            ct_step = self.current_step_template.format(
                                think=tracker_result.critical_think,
                                action_text=f'search(query="{tracker_result.critical_search_query}")',
                                observation=tracker_result.critical_observation,
                            )
                            input_prompt += ct_step + "\n"

                # Hard-coded: update plan after each search observation
                if self.use_plan:
                    self._vprint(display_iter, "plan", "mode=update")
                    plan_result = self.plan_tool.execute(
                        mode="update",
                        query=self.query,
                        last_search_query=self.last_search_query,
                        search_results=self.last_search_results,
                    )
                    self._vprint(display_iter, "plan-output", (plan_result["observation"] or "")[:500])
                    reasoning_path.append({
                        "think": "",
                        "action_type": "plan",
                        "plan_mode": "update",
                        "observation": plan_result["observation"],
                    })

            elif action_type == "finish":
                reasoning_path.append({
                    "think": thought,
                    "action_type": action_type,
                    "conclusion": action_entity,
                })
                conclusion = action_entity

            self.history.append(history_entry)

            if done or iter_num == self.max_iteration - 1:
                break

            # In non-plan mode, accumulate history linearly in the prompt
            if not self.use_plan:
                current_step_text = self.current_step_template.format(
                    think=thought,
                    action_text=action_text,
                    observation=obs,
                )
                input_prompt += current_step_text + "\n"

        # ---- force Finish if the loop ended without one ----
        if not conclusion and reasoning_path:
            if self.use_plan:
                force_prompt = base_prompt + self._build_recent_history_context()
            else:
                force_prompt = input_prompt

            if early_stop_result:
                force_prompt += f"\n<think>{early_stop_result.reasoning}</think>\n"

            force_prompt += (
                f"\n{FINAL_ANSWER_INSTRUCTION} {REACT_FORMAT}"
            )

            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": force_prompt},
            ]

            force_output = self.generator.complete(
                messages, max_tokens=500, strip_think=False,
            )

            # Try to parse finish action from output
            action_raw = get_action(force_output)
            if action_raw is None:
                _, action_raw = extract_think_and_clean(force_output)
            parsed = parse_action_call(action_raw)
            if parsed and parsed[0] == "finish":
                conclusion = parsed[1]
            else:
                think_content = self.get_think(force_output)
                conclusion = think_content or force_output.strip()

            action_type = "early_stop" if early_stop_result else "finish"
            force_think = self.get_think(force_output) or ""
            final_think = early_stop_result.reasoning if early_stop_result else force_think
            self._vprint(iter_num + 1, "think", final_think or "(no thought)")
            self._vprint(iter_num + 1, "finish", conclusion or "(no conclusion)")
            reasoning_path.append({
                "think": early_stop_result.reasoning if early_stop_result else "",
                "action_type": action_type,
                "conclusion": conclusion,
            })
            self.history.append({
                "iter": iter_num + 1,
                "thought": early_stop_result.reasoning if early_stop_result else "",
                "action": action_type,
                "value": conclusion,
                "observation": conclusion,
            })

        num_iterations = iter_num + 1
        return reasoning_path, conclusion if conclusion else "", num_iterations

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_history(self) -> list:
        """Return a copy of the action history."""
        return self.history.copy()
