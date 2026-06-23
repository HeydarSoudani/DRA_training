"""
CPMReport: Writing as Reasoning Agent

This module implements the CPMReport approach where an agent generates
comprehensive reports through iterative search, planning, and writing.

The agent operates as a state machine with the following states:
- search: Search for relevant information
- analyst-init_plan: Create initial report outline
- write: Write content for a section
- analyst-extend_plan: Extend outline with subsections
- done: Report generation complete
"""

import os
import re
import json
import copy
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Template

from searcher_component.fusion import interleaving_fusion
from deep_research_agents.agents.base_agent import BasicAgent
from deep_research_agents.agent_tools.controller_results import CriticalThinkResult, EarlyStopResult
from utils.config import InferenceConfig

logger = logging.getLogger(__name__)

_VLLM_MODEL_PREFIXES = ("openbmb/",)


def _is_vllm_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _VLLM_MODEL_PREFIXES)


class CPMReport(BasicAgent):
    """
    CPMReport implements a writing-as-reasoning approach for report generation.

    The agent iteratively searches for information, plans the report structure,
    and writes content following a multi-agent workflow:
    Analyst (planning) -> Searcher (retrieval) -> Writer (content generation)
    """

    AGENT_NAME = "CPMReport"

    def __init__(
        self,
        llm_client=None,
        retriever=None,
        max_iteration: int = 140,
        seen_top_k: int = 5,
        max_extend_steps: int = 5,
        max_retries: int = 3,
        hard_mode: bool = True,
        verbose: bool = True,
        search_tool=None,
        oracle_outline_path: str = None,
        max_passage_chars: int = 4000,
        model_name: str = "",
        model_url: str = None,
        **kwargs,
    ):
        super().__init__(
            llm_client=llm_client,
            retriever=retriever,
            max_iteration=max_iteration,
            seen_top_k=seen_top_k,
            search_tool=search_tool,
        )
        self.max_steps = max_iteration
        self.max_extend_steps = max_extend_steps
        self.max_retries = max_retries
        self.hard_mode = hard_mode
        self.verbose = verbose
        self.max_passage_chars = max_passage_chars

        # --- Dual API routing ---------------------------------------------------
        # When model_name looks like a vLLM-served model (e.g. "openbmb/..."),
        # we bypass self.generator and talk directly to the vLLM OpenAI endpoint.
        # Otherwise we delegate to self.generator (LiteLLMClient from the pipeline).
        self._use_vllm = bool(model_name and _is_vllm_model(model_name))
        self.model_name = model_name
        self.model_url = model_url or os.environ.get("CPM_REPORT_API_BASE", "http://localhost:6008/v1")
        self._api_key = os.environ.get("CPM_REPORT_API_KEY", "EMPTY")
        self._default_temperature = 0.7

        if self._use_vllm:
            self.inference_config = InferenceConfig(
                api_type="chat_completion",
                model_name=model_name,
                api_base=self.model_url,
                api_key=self._api_key,
                max_output_tokens=16384,
                reasoning_effort=None,
            )
        else:
            self.inference_config = InferenceConfig(
                api_type="chat_completion",
                model_name=model_name,
            )

        # Load oracle outline data if provided
        self._oracle_data = None
        if oracle_outline_path:
            with open(oracle_outline_path, "r", encoding="utf-8") as f:
                self._oracle_data = json.load(f)

        # Load prompts
        self.prompts = self._load_prompts()

    # ==================== LLM Call Routing ====================

    async def _llm_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._use_vllm:
            return await self._llm_call_vllm(messages, temperature, max_tokens)
        return await self._llm_call_api(messages, temperature, max_tokens)

    async def _llm_call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self.generator.complete(messages, **kwargs)

    async def _llm_call_vllm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        import openai as _openai

        cfg = self.inference_config
        client = _openai.OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)
        create_kwargs: Dict[str, Any] = {
            "model": cfg.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content or ""

    def _load_prompts(self) -> Dict[str, Template]:
        """Load Jinja2 templates for different agent actions."""
        prompt_dir = Path(__file__).parent.parent / "prompts" / "cpm_report"

        prompts = {}
        prompt_names = ["search", "init_plan", "extend_plan", "write", "write_citation"]
        if self._oracle_data is not None:
            prompt_names.append("init_plan_oracle")
        for prompt_name in prompt_names:
            prompt_file = prompt_dir / f"{prompt_name}.txt"
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = Template(f.read())

        return prompts

    # ==================== State Management ====================

    def _init_state(self, query: str) -> Dict[str, Any]:
        """Initialize state for a new query."""
        return {
            "query": query,
            "state": "search",
            "cursor": "outline",
            "survey": {},
            "step": 0,
            "extend_time": 0,
            "extend_result": "",
            "retrieved_info": "",
            "parsed": True,
            "retry_count": 0,
            "citation_registry": {},
            "citation_counter": 0,
            "citation_to_doc_id": {},  # Map citation_id -> doc_id for evaluation
            "keywords_history": [],
            "search_history": [],
            "retrieved_docs_per_search": [],  # Track retrieved docs for each search
            "trajectory": [],  # Track full trajectory step-by-step
        }

    def _update_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update state based on current state and results.

        State transition logic:
        - If step >= max_steps: -> done
        - If parsed is False: -> increment retry_count and handle accordingly
        - If current state is 'search':
            - cursor == 'outline': -> analyst-init_plan
            - cursor is section-X: -> write
            - cursor is None: -> done
        - If current state is 'analyst-init_plan':
            - cursor != 'outline': -> search (init plan succeeded)
            - cursor == 'outline': -> analyst-init_plan (retry)
        - If current state is 'write':
            - cursor is not None: -> search (continue writing)
            - cursor is None and extend_time < max_extend_steps: -> analyst-extend_plan
            - cursor is None and extend_time >= max_extend_steps: -> done
        - If current state is 'analyst-extend_plan':
            - extend_result == 'extended': -> search
            - extend_result == 'nop': -> done
            - extend_result == 'retry': -> analyst-extend_plan
        """
        current_state = state["state"]
        cursor = state["cursor"]
        extend_time = state["extend_time"]
        extend_result = state.get("extend_result", "")
        parsed = state.get("parsed", True)
        step = state["step"]
        retry_count = state.get("retry_count", 0)

        # Increment step
        step += 1
        state["step"] = step

        # Check early stopping
        if state.get("early_stop"):
            state["state"] = "done"
            return state

        # Check max steps
        if step >= self.max_steps:
            state["state"] = "done"
            return state

        # Handle parse failures
        if not parsed:
            retry_count += 1
            state["retry_count"] = retry_count

            # Check if max retries exceeded
            if retry_count >= self.max_retries:
                self._print(f"  ⚠ Max retries ({self.max_retries}) reached, skipping current action")

                # Handle different states
                if current_state == "analyst-init_plan":
                    # Can't proceed without initial plan, so give up
                    state["state"] = "done"
                    return state

                elif current_state == "write":
                    # Skip this section and move to next
                    survey = state["survey"]
                    # Mark section as having empty content so we can move on
                    if cursor:
                        survey = self._update_position(survey, cursor, {"content": "[Content generation failed after retries]"})
                        state["survey"] = survey
                    # Check next cursor position
                    state["cursor"] = self._check_progress_position(survey)
                    state["retry_count"] = 0  # Reset retry count

                    # Transition based on new cursor
                    if state["cursor"] is None:
                        # No more sections
                        if extend_time < self.max_extend_steps:
                            state["state"] = "analyst-extend_plan"
                            state["extend_time"] += 1
                        else:
                            state["state"] = "done"
                    else:
                        # Move to search for next section
                        state["state"] = "search"
                    return state

                elif current_state == "analyst-extend_plan":
                    state["extend_time"] += 1
                    state["retry_count"] = 0  # Reset retry count
                    if state["extend_time"] >= self.max_extend_steps:
                        state["state"] = "done"
                    return state

                elif current_state == "search":
                    # If search fails, try to continue anyway with empty results
                    state["retrieved_info"] = ""
                    state["retry_count"] = 0  # Reset retry count
                    # Transition based on cursor
                    if cursor == "outline":
                        state["state"] = "analyst-init_plan"
                    elif cursor is not None:
                        state["state"] = "write"
                    else:
                        state["state"] = "done"
                    return state

            # Otherwise keep current state for retry
            return state

        # Reset retry count on successful parse
        state["retry_count"] = 0

        # State transitions
        if current_state == "search":
            if cursor == "outline":
                state["state"] = "analyst-init_plan"
            elif cursor is not None:
                state["state"] = "write"
            else:
                state["state"] = "done"

        elif current_state == "analyst-init_plan":
            if cursor != "outline" and cursor is not None:
                state["state"] = "search"
            # else: keep analyst-init_plan for retry

        elif current_state == "write":
            if cursor is not None:
                state["state"] = "search"
            elif extend_time < self.max_extend_steps:
                state["state"] = "analyst-extend_plan"
                state["extend_time"] += 1
            else:
                state["state"] = "done"

        elif current_state == "analyst-extend_plan":
            if extend_result == "extended":
                state["state"] = "search"
            elif extend_result == "nop":
                state["state"] = "done"
            elif extend_time < self.max_extend_steps:
                # keep analyst-extend_plan for retry
                pass
            else:
                state["state"] = "done"

        return state

    # ==================== Search Functions ====================

    async def _search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search action."""
        query = state["query"]
        survey = state["survey"]

        # Format prompt
        current_outline = self._print_survey_outline(survey)
        current_instruction = self._get_current_instruction(state)

        prompt = self.prompts["search"].render(
            user_query=query,
            current_outline=current_outline,
            current_instruction=current_instruction,
        )

        step = state['step']

        # Call LLM with temperature adjustment for retries
        retry_count = state.get("retry_count", 0)
        temperature = min(self._default_temperature + (retry_count * 0.3), 1.0)  # Increase temp on retries
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_call(messages, temperature=temperature)

        # Parse response
        result = self._parse_search_response(response)
        keywords = result.get("keywords", [])
        parsed = result.get("parse_success", False)
        thought = result.get("thought", "")

        state["parsed"] = parsed

        if not parsed:
            self._vprint(step, "search-error", "Failed to parse search response")
            return state

        self._vprint(step, "search", str(keywords) if len(keywords) != 1 else keywords[0])
        state["keywords_history"].append(keywords)

        # Perform retrieval for each keyword, then fuse by interleaving
        if self.search_tool is not None:
            # Unified search tool handles retrieve + fuse + optional rerank
            reasoning = self._print_survey_outline(survey) if survey else None
            fused_docs = self.search_tool.execute(
                keywords,
                original_query=state["query"],
                reasoning=reasoning if reasoning else None,
            )
            for keyword in keywords:
                state["search_history"].append({
                    "keyword": keyword,
                    "num_results": "—",  # per-keyword counts unavailable
                })
        else:
            per_keyword_docs = []  # List of doc lists (one per keyword)
            for keyword in keywords:
                try:
                    result = self.retriever.retrieve(keyword)

                    if isinstance(result, dict) and "results" in result:
                        docs = result["results"]
                    else:
                        docs = result

                    per_keyword_docs.append(docs)
                    state["search_history"].append({
                        "keyword": keyword,
                        "num_results": len(docs)
                    })
                except Exception as e:
                    self._print(f"  ⚠ Retrieval error for '{keyword}': {e}")
                    per_keyword_docs.append([])

            # Fuse results from all keywords by interleaving (like Tongyi agent)
            if not per_keyword_docs:
                fused_docs = []
            elif len(per_keyword_docs) == 1:
                fused_docs = per_keyword_docs[0]
            else:
                fused_docs = interleaving_fusion(per_keyword_docs, window=1)

        # Extract passages from fused docs
        fused_passages = []
        for idx, doc in enumerate(fused_docs):
            text = doc.get("relevant_text", "") or doc.get("text", "")
            if not text:
                text = doc.get("contents", "")
            if not text:
                title = doc.get("title", "")
                contents = doc.get("contents", "")
                text = f"{title}\n{contents}".strip() if title or contents else ""
            if not text or not text.strip():
                self._print(f"  ⚠ Warning: Empty passage for fused doc {idx}: keys={list(doc.keys())}")
            fused_passages.append(text)

        # Process passages with citations (single fused list)
        retrieved_info = self._process_passages_with_citation(
            [fused_passages], survey, state, [fused_docs]
        )

        # Trajectory controller: may inject observation, critical_think, or early stop
        seen_docs = fused_docs[:self.seen_top_k]
        controller_result = self.post_search_evaluate(
            subquery=keywords,
            docs=seen_docs, iter_num=step,
            original_query=state["query"],
            thinking=thought,
            seen_docs=seen_docs,
            trajectory=state["trajectory"],
            reasoning_path=state["trajectory"],
        )
        if isinstance(controller_result, EarlyStopResult):
            state["early_stop"] = True
        elif isinstance(controller_result, CriticalThinkResult):
            retrieved_info += f"\n{controller_result.critical_observation}"

        state["retrieved_info"] = retrieved_info

        # Store fused docs for this search (for evaluation)
        state["retrieved_docs_per_search"].append(fused_docs)

        # Count non-empty passages
        non_empty_passages = sum(1 for p in fused_passages if p and p.strip())
        self._vprint(step, "search-summary", f"{len(keywords)} queries fused, {len(fused_docs)} docs ({non_empty_passages} non-empty), {len(state['citation_to_doc_id'])} seen docs total")

        # Collect doc IDs for trajectory
        doc_ids_retrieved = []
        for doc in fused_docs:
            doc_id = doc.get("doc_id") or doc.get("id")
            if doc_id:
                doc_ids_retrieved.append(doc_id)

        # Add to trajectory
        state["trajectory"].append({
            "step": state["step"],
            "state": "search",
            "action": "search",
            "think": thought,
            "input": {"keywords": keywords},
            "output": {
                "num_results": len(doc_ids_retrieved),
                "doc_ids": doc_ids_retrieved[:self.seen_top_k],
            },
            "all_docs": fused_docs,
        })
        if isinstance(controller_result, CriticalThinkResult):
            critical_think_doc_ids = [
                doc.get("doc_id") or doc.get("id") or ""
                for doc in controller_result.critical_docs
            ]
            state["trajectory"].append({
                "step": controller_result.critical_think_iter,
                "state": "critical_search",
                "action": "critical_search",
                "think": controller_result.critical_think,
                "input": {"keywords": [controller_result.critical_search_query]},
                "output": {
                    "num_results": len(critical_think_doc_ids),
                    "doc_ids": critical_think_doc_ids[:self.seen_top_k],
                },
                "all_docs": controller_result.critical_docs,
                "is_critical_think": True,
            })

        return state

    # ==================== Planning Functions ====================

    async def _init_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial report outline."""
        query = state["query"]
        retrieved_info = state["retrieved_info"]

        prompt = self.prompts["init_plan"].render(
            user_query=query,
            current_information=retrieved_info,
        )

        step = state['step']

        # Call LLM with temperature adjustment for retries
        retry_count = state.get("retry_count", 0)
        temperature = min(self._default_temperature + (retry_count * 0.3), 1.0)  # Increase temp on retries
        messages = [{"role": "user", "content": prompt}]
        # Use lower max_tokens for init_plan since it only needs to generate outline structure
        response = await self._llm_call(messages, temperature=temperature, max_tokens=1000)

        # Parse response
        result = self._parse_init_plan_response(response, query)
        parsed = result.get("parse_success", False)
        action = result.get("action", {})
        thought = result.get("thought", "")

        state["parsed"] = parsed

        if not parsed or action.get("name") != "init-plan":
            self._vprint(step, "plan-error", "Failed to parse init-plan response")
            return state

        # Create survey
        survey = {
            "title": action.get("title", ""),
            "sections": action.get("sections", [])
        }
        state["survey"] = survey
        state["cursor"] = self._check_progress_position(survey)

        section_titles = [s.get("title", "") for s in survey.get("sections", [])]
        self._vprint(step, "init-plan", f"'{survey.get('title', '')}' with {len(section_titles)} sections: {section_titles}")

        # Extract doc IDs from retrieved info for trajectory
        num_docs, doc_ids = self._extract_doc_ids_from_cited_text(state["retrieved_info"], state, limit=20)

        # Add to trajectory
        state["trajectory"].append({
            "step": state["step"],
            "state": "analyst-init_plan",
            "action": "init-plan",
            "think": thought,
            "input": {
                "num_retrieved_docs": num_docs,
                "retrieved_doc_ids": doc_ids,
            },
            "output": {
                "title": survey.get("title", ""),
                "num_sections": len(survey.get("sections", [])),
                "sections": [s.get("title", "") for s in survey.get("sections", [])],
            }
        })

        return state

    async def _init_plan_oracle(self, state: Dict[str, Any], query_id: str) -> Dict[str, Any]:
        """Create initial report outline from oracle aspects instead of retrieved passages.

        Skips the initial search step entirely. Oracle aspects and descriptions
        are fed to the LLM via the init_plan_oracle prompt so it can generate
        the outline structure.
        """
        query = state["query"]

        # Extract unique aspects from oracle data for this query
        entry = self._oracle_data.get(str(query_id))
        if entry is None:
            self._print(f"  ⚠ Query {query_id!r} not found in oracle data, falling back to normal init_plan")
            return state

        aspects = []
        seen_aspects = set()
        for doc in entry.get("documents", []):
            aspect_name = doc.get("aspect", "")
            if aspect_name and aspect_name not in seen_aspects:
                seen_aspects.add(aspect_name)
                aspects.append({
                    "name": aspect_name,
                    "description": doc.get("aspect_description", ""),
                })

        if not aspects:
            self._print(f"  ⚠ No aspects found in oracle data for {query_id!r}, falling back to normal init_plan")
            return state

        prompt = self.prompts["init_plan_oracle"].render(
            user_query=query,
            aspects=aspects,
        )

        step = state['step']
        self._vprint(step, "oracle-init-plan", f"Using {len(aspects)} oracle aspects")

        # Call LLM with temperature adjustment for retries
        retry_count = state.get("retry_count", 0)
        temperature = min(self._default_temperature + (retry_count * 0.3), 1.0)
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_call(messages, temperature=temperature, max_tokens=1000)

        # Parse response (same as _init_plan)
        result = self._parse_init_plan_response(response, query)
        parsed = result.get("parse_success", False)
        action = result.get("action", {})
        thought = result.get("thought", "")

        state["parsed"] = parsed

        if not parsed or action.get("name") != "init-plan":
            self._vprint(step, "plan-error", "Failed to parse oracle init-plan response")
            return state

        # Create survey
        survey = {
            "title": action.get("title", ""),
            "sections": action.get("sections", [])
        }
        state["survey"] = survey
        state["cursor"] = self._check_progress_position(survey)

        section_titles = [s.get("title", "") for s in survey.get("sections", [])]
        self._vprint(step, "oracle-init-plan", f"'{survey.get('title', '')}' with {len(section_titles)} sections: {section_titles}")

        # Add to trajectory
        state["trajectory"].append({
            "step": state["step"],
            "state": "analyst-init_plan_oracle",
            "action": "init-plan-oracle",
            "think": thought,
            "input": {
                "oracle_aspects": [a["name"] for a in aspects],
            },
            "output": {
                "title": survey.get("title", ""),
                "num_sections": len(survey.get("sections", [])),
                "sections": [s.get("title", "") for s in survey.get("sections", [])],
            }
        })

        return state

    async def _extend_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extend outline with subsections."""
        query = state["query"]
        survey = state["survey"]
        cursor = state["cursor"]

        survey_text = self._print_survey_outline(survey, last_detail=True)

        prompt = self.prompts["extend_plan"].render(
            user_query=query,
            current_survey=survey_text,
        )

        step = state['step']

        # Call LLM with temperature adjustment for retries
        retry_count = state.get("retry_count", 0)
        temperature = min(self._default_temperature + (retry_count * 0.3), 1.0)  # Increase temp on retries
        messages = [{"role": "user", "content": prompt}]
        # Use lower max_tokens for extend_plan since it only needs to generate subsection structure
        response = await self._llm_call(messages, temperature=temperature, max_tokens=800)

        # Parse response
        result = self._parse_extend_plan_response(response, survey, cursor, query)
        parsed = result.get("parse_success", False)
        action = result.get("action", {})
        action_name = action.get("name", "")
        thought = result.get("thought", "")

        state["parsed"] = parsed

        if not parsed:
            self._vprint(step, "extend-error", "Failed to parse extend-plan response")
            state["extend_result"] = "retry"
            return state

        if action_name == "extend-plan":
            position = action.get("position", "")
            subsections = action.get("subsections", [])

            if position and subsections:
                survey = self._update_position(survey, position, {"subsections": copy.deepcopy(subsections)})
                state["survey"] = survey
                state["cursor"] = self._check_progress_position(survey)
                state["extend_result"] = "extended"
                sub_titles = [s.get("title", "") for s in subsections]
                self._vprint(step, "extend-plan", f"{position} + {len(subsections)} subsections: {sub_titles}")

                # Add to trajectory
                state["trajectory"].append({
                    "step": state["step"],
                    "state": "analyst-extend_plan",
                    "action": "extend-plan",
                    "think": thought,
                    "input": {"position": position},
                    "output": {
                        "position": position,
                        "num_subsections": len(subsections),
                        "subsections": [s.get("title", "") for s in subsections],
                    }
                })
            else:
                state["extend_result"] = "retry"
                self._vprint(step, "extend-error", "Invalid extend-plan action")

        elif action_name == "nop":
            state["extend_result"] = "nop"
            self._vprint(step, "extend-plan", "No extension needed (nop)")

            # Add to trajectory
            state["trajectory"].append({
                "step": state["step"],
                "state": "analyst-extend_plan",
                "action": "nop",
                "think": thought,
                "input": {},
                "output": {"result": "no extension needed"}
            })
        else:
            state["extend_result"] = "retry"

        return state

    # ==================== Writing Functions ====================

    async def _write(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Write content for current section."""
        query = state["query"]
        survey = state["survey"]
        cursor = state["cursor"]
        retrieved_info = state["retrieved_info"]

        survey_text = self._print_survey_outline(survey, last_detail=True)
        current_instruction = self._get_current_instruction(state)

        step = state['step']

        # Check if we have sufficient retrieved information
        if not retrieved_info or not retrieved_info.strip():
            self._vprint(step, "write-error", f"No retrieved info for section: {cursor}")
            state["parsed"] = False
            return state

        # Count available citations
        num_citations = len(self._extract_bibkeys(retrieved_info))
        if num_citations == 0:
            self._vprint(step, "write-error", f"No valid citations for section: {cursor}")
            state["parsed"] = False
            return state

        self._vprint(step, "write", f"section '{cursor}' with {num_citations} citations")

        # Use citation-based writing prompt
        prompt = self.prompts["write_citation"].render(
            user_query=query,
            current_survey=survey_text,
            current_instruction=current_instruction,
            current_information=retrieved_info,
        )

        # Call LLM with temperature adjustment for retries
        retry_count = state.get("retry_count", 0)
        temperature = min(self._default_temperature + (retry_count * 0.3), 1.0)  # Increase temp on retries
        if retry_count > 0:
            self._vprint(step, "write-retry", f"{retry_count}/{self.max_retries}, temp={temperature:.2f}")
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_call(messages, temperature=temperature)

        # Parse response
        retrieved_bibkeys = self._extract_bibkeys(retrieved_info)
        result = self._parse_write_response(response, survey, cursor, query, retrieved_bibkeys)
        parsed = result.get("parse_success", False)
        action = result.get("action", {})
        thought = result.get("thought", "")

        state["parsed"] = parsed

        if not parsed or action.get("name") != "write":
            self._vprint(step, "write-error", "Failed to parse write response")
            return state

        content = action.get("content", "")
        if content:
            # Post-process content to handle '#' characters
            # Escape '#' at the start of lines to prevent markdown header conflicts
            content = re.sub(r'^(\s*)#', r'\1\\#', content, flags=re.MULTILINE)

            survey = self._update_position(survey, cursor, {"content": content})
            state["survey"] = survey
            state["cursor"] = self._check_progress_position(survey)
            self._vprint(step, "write-done", f"section '{cursor}' ({len(content)} chars), next: {state['cursor']}")

            # Extract doc IDs from retrieved info for trajectory
            num_docs, doc_ids = self._extract_doc_ids_from_cited_text(retrieved_info, state, limit=20)

            # Add to trajectory
            state["trajectory"].append({
                "step": state["step"],
                "state": "write",
                "action": "write",
                "think": thought,
                "input": {
                    "section": cursor,
                    "num_retrieved_docs": num_docs,
                    "retrieved_doc_ids": doc_ids,
                },
                "output": {
                    "section": cursor,
                    "content_length": len(content),
                    "content_preview": content[:200],
                }
            })

        return state

    # ==================== Response Parsing ====================

    def _parse_response(self, response_text: str, is_json: bool = True, valid_actions: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Parse agent response into thought and action."""
        extracted_result = {}

        if valid_actions is None:
            valid_actions = ["search", "init-plan", "extend-plan", "nop", "write"]

        think_pattern = r"<thought>(.*?)</thought>"
        action_pattern = r"<action>(.*?)</action>"

        think_is_valid, action_is_valid = False, False

        think_match = re.search(think_pattern, response_text, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
            think_is_valid = True
        else:
            think = ""
        extracted_result["thought"] = think

        if is_json:
            action_match = re.search(action_pattern, response_text, re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()
                try:
                    action = json.loads(action)
                    action_is_valid = self._validate_action(action, valid_actions=valid_actions, **kwargs)
                except:
                    action_is_valid = False
                    action = {}
            else:
                action_is_valid = False
                action = {}
        else:
            action_match = re.search(action_pattern, response_text, re.DOTALL | re.MULTILINE)
            if action_match:
                action = action_match.group(1).strip()
            else:
                action = ""
            action = {"name": "write", "content": action}
            action_is_valid = self._validate_action(action, valid_actions=valid_actions, **kwargs)

        extracted_result["action"] = action
        extracted_result["parse_success"] = action_is_valid

        return extracted_result

    def _parse_search_response(self, response: str) -> Dict[str, Any]:
        """Parse search response to extract keywords."""
        result = self._parse_response(
            response_text=response,
            is_json=True,
            valid_actions=["search"],
            hard_mode=self.hard_mode
        )
        keywords = result.get("action", {}).get("keywords", [])
        return {"keywords": keywords, "parse_success": result.get("parse_success", False), "thought": result.get("thought", "")}

    def _parse_init_plan_response(self, response: str, user_instruction: str) -> Dict[str, Any]:
        """Parse init-plan response."""
        return self._parse_response(
            response_text=response,
            is_json=True,
            user_instruction=user_instruction,
            valid_actions=["init-plan"],
            hard_mode=self.hard_mode
        )

    def _parse_extend_plan_response(self, response: str, current_survey: Dict[str, Any], cursor: str, user_instruction: str) -> Dict[str, Any]:
        """Parse extend-plan response."""
        return self._parse_response(
            response_text=response,
            is_json=True,
            current_survey=current_survey,
            cursor=cursor,
            user_instruction=user_instruction,
            valid_actions=["extend-plan", "nop"],
            hard_mode=self.hard_mode
        )

    def _parse_write_response(self, response: str, current_survey: Dict[str, Any], cursor: str, user_instruction: str, retrieved_bibkeys: List[str]) -> Dict[str, Any]:
        """Parse write response."""
        return self._parse_response(
            response_text=response,
            is_json=False,
            current_survey=current_survey,
            cursor=cursor,
            user_instruction=user_instruction,
            retrieved_bibkeys=retrieved_bibkeys,
            valid_actions=["write"],
            hard_mode=self.hard_mode
        )

    # ==================== Validation ====================

    def _validate_action(self, action: Dict[str, Any], valid_actions: List[str], current_survey: Optional[Dict[str, Any]] = None, cursor: Optional[str] = None, user_instruction: Optional[str] = None, hard_mode: bool = False, retrieved_bibkeys: Optional[List[str]] = None) -> bool:
        """Validate if an action is properly formatted."""
        if not isinstance(action, dict):
            return False
        if "name" not in action:
            return False

        if action["name"] not in valid_actions:
            return False

        try:
            if action["name"] == "search":
                assert "keywords" in action
                assert isinstance(action["keywords"], list)
                assert len(action["keywords"]) > 0
                assert action.keys() == {"name", "keywords"}
                for kw in action["keywords"]:
                    assert isinstance(kw, str) and len(kw) > 0
                if hard_mode:
                    assert len(action["keywords"]) <= 5

            elif action["name"] == "init-plan":
                assert "title" in action
                assert "sections" in action
                assert isinstance(action["title"], str) and len(action["title"]) > 0
                assert isinstance(action["sections"], list) and len(action["sections"]) > 0
                assert action.keys() == {"name", "title", "sections"}
                for sec in action["sections"]:
                    assert isinstance(sec, dict)
                    assert "title" in sec and "plan" in sec
                    assert isinstance(sec["title"], str) and len(sec["title"]) > 0
                    assert isinstance(sec["plan"], str) and len(sec["plan"]) > 0
                    assert sec.keys() == {"title", "plan"}
                if hard_mode:
                    assert 3 <= len(action["sections"]) <= 12
                    if user_instruction:
                        assert self._check_language_consistency(
                            {"title": action["title"], "sections": action["sections"]},
                            user_instruction
                        )

            elif action["name"] == "extend-plan":
                assert "position" in action
                assert "subsections" in action
                assert isinstance(action["position"], str) and len(action["position"]) > 0
                assert isinstance(action["subsections"], list) and len(action["subsections"]) > 0
                assert action.keys() == {"name", "position", "subsections"}
                if cursor is not None:
                    assert action["position"] == cursor
                assert action["position"].count(".") < 2

                # Check if already extended
                if current_survey:
                    try:
                        section_node = self._get_position(current_survey, action["position"], tag="outline")
                        if "subsections" in section_node:
                            return False
                    except:
                        return False

                for sec in action["subsections"]:
                    assert isinstance(sec, dict)
                    assert "title" in sec and "plan" in sec
                    assert isinstance(sec["title"], str) and len(sec["title"]) > 0
                    assert isinstance(sec["plan"], str) and len(sec["plan"]) > 0
                    assert sec.keys() == {"title", "plan"}
                if hard_mode:
                    assert 2 <= len(action["subsections"]) <= 7
                    if user_instruction:
                        assert self._check_language_consistency(
                            {"subsections": action["subsections"]},
                            user_instruction
                        )

            elif action["name"] == "nop":
                assert action.keys() == {"name"}

            elif action["name"] == "write":
                assert "content" in action
                assert action.keys() == {"name", "content"}
                if hard_mode:
                    # Note: '#' character validation removed - handled through post-processing

                    try:
                        assert "bibkey" not in action["content"].lower(), "Content contains 'bibkey' word"
                    except AssertionError as e:
                        if self.verbose:
                            print(f"[Validation Failed] {e}")
                        raise

                    try:
                        assert len(action["content"].strip()) > 50, f"Content too short: {len(action['content'].strip())} chars"
                    except AssertionError as e:
                        if self.verbose:
                            print(f"[Validation Failed] {e}")
                        raise

                    if user_instruction:
                        try:
                            assert self._check_language_consistency(action["content"], user_instruction), "Language consistency check failed"
                        except AssertionError as e:
                            if self.verbose:
                                print(f"[Validation Failed] {e}")
                            raise

                    # Check citations
                    if "[" in action["content"] and "]" in action["content"]:
                        # Count [id] citations
                        citation_pattern = r"\[(\d+)\]"
                        citations = re.findall(citation_pattern, action["content"])

                        # Removed the max citation limit - allow thorough citation
                        # Old limit of 12 was too restrictive for detailed sections
                        # try:
                        #     assert len(citations) <= 12, f"Too many citations: {len(citations)} > 12"
                        # except AssertionError as e:
                        #     if self.verbose:
                        #         print(f"[Validation Failed] {e}")
                        #     raise

                        try:
                            assert len(citations) > 0, f"Content has brackets but no valid citations [1], [2], etc."
                        except AssertionError as e:
                            if self.verbose:
                                print(f"[Validation Failed] {e}")
                                content_preview = action["content"][:200] if len(action["content"]) > 200 else action["content"]
                                print(f"[Content Preview] {content_preview}...")
                            raise

                        # Validate citation IDs if retrieved_bibkeys provided
                        if retrieved_bibkeys:
                            for cit_id in citations:
                                # Citation IDs should correspond to available passages
                                pass  # Lenient validation for now

        except Exception as e:
            if self.verbose and str(e):
                print(f"[Validation Error] {type(e).__name__}: {e}")
            return False

        return True

    # ==================== Helper Functions ====================

    def _check_progress_position(self, survey: Dict[str, Any]) -> Optional[str]:
        """Check the current progress position in the survey."""
        if not survey or survey == {}:
            return "outline"

        if "sections" in survey:
            for i, section in enumerate(survey["sections"]):
                if "content" not in section:
                    return f"section-{i+1}"
                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        if "content" not in subsection:
                            return f"section-{i+1}.{j+1}"
                        if "subsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsections"]):
                                if "content" not in subsubsection:
                                    return f"section-{i+1}.{j+1}.{k+1}"
        return None

    def _update_position(self, survey: Dict[str, Any], position: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update survey content at a specific position."""
        survey = copy.deepcopy(survey)

        if position == "outline":
            for key, value in update_data.items():
                survey[key] = value
        else:
            parts = position.split('-')[1].split('.')
            indices = [int(part) - 1 for part in parts]
            current = survey

            for i, idx in enumerate(indices):
                if i == 0:
                    current = current['sections'][idx]
                else:
                    current = current['subsections'][idx]

            for key, value in update_data.items():
                current[key] = value

        return survey

    def _get_position(self, survey: Dict[str, Any], position: str, tag: str = "content") -> Any:
        """Get content at a specific position in the survey."""
        parts = position.split('-')[1].split('.')
        indices = [int(part) - 1 for part in parts]
        current = survey

        for i, idx in enumerate(indices):
            if i == 0:
                current = current['sections'][idx]
            else:
                current = current['subsections'][idx]

        if tag == "outline":
            return current
        elif tag == "content":
            return current.get('content', "")
        else:
            raise ValueError(f"Invalid tag: {tag}")

    def _get_current_instruction(self, state: Dict[str, Any]) -> str:
        """Generate instruction for current cursor position."""
        cursor = state.get("cursor")
        survey = state.get("survey", {})

        if not cursor or cursor == "outline":
            return "Please create an initial outline for the report."

        if cursor is None:
            return "The report is complete."

        try:
            section_info = self._get_position(survey, cursor, tag="outline")
            section_title = section_info.get("title", section_info.get("name", ""))
            section_plan = section_info.get("plan", "")

            return f"Please write content for {cursor}: {section_title}\nPlan: {section_plan}"
        except:
            return f"Please write content for {cursor}"

    def _print_survey_outline(self, survey: Dict[str, Any], last_detail: bool = False) -> str:
        """Print survey structure with hierarchical detail."""
        if not survey or survey == {}:
            return "There is no survey."

        lines = []

        # Title
        try:
            title = survey.get("title", "Untitled")
            lines.append(f"# Title: {title}\n")
        except:
            lines.append(f"# Title: None\n")

        # Sections
        now_section = self._check_progress_position(survey)
        now_hire = now_section.count(".") if now_section else 0

        if "sections" in survey:
            for i, section in enumerate(survey["sections"]):
                title_key = "name" if "name" in section else "title"
                name = section.get(title_key, "")

                # Content summary
                if "content" in section and section["content"]:
                    content = "[OK] " + section["content"][:100].replace("\n", " ")
                elif "plan" in section and section["plan"]:
                    content = "[PLAN] " + section["plan"].replace("\n", " ")
                else:
                    content = ""

                lines.append(f"## Section-{i+1} [{name}]\n{content}\n")

                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        sub_name = subsection.get(title_key, "")

                        if "content" in subsection and subsection["content"]:
                            sub_content = "[OK] " + subsection["content"][:100].replace("\n", " ")
                        elif "plan" in subsection and subsection["plan"]:
                            sub_content = "[PLAN] " + subsection["plan"].replace("\n", " ")
                        else:
                            sub_content = ""

                        lines.append(f"### Section-{i+1}.{j+1} [{sub_name}]\n{sub_content}\n")

                        if "subsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsections"]):
                                subsub_name = subsubsection.get(title_key, "")

                                if "content" in subsubsection and subsubsection["content"]:
                                    subsub_content = "[OK] " + subsubsection["content"][:100].replace("\n", " ")
                                elif "plan" in subsubsection and subsubsection["plan"]:
                                    subsub_content = "[PLAN] " + subsubsection["plan"].replace("\n", " ")
                                else:
                                    subsub_content = ""

                                lines.append(f"#### Section-{i+1}.{j+1}.{k+1} [{subsub_name}]\n{subsub_content}\n")

        return "\n".join(lines).strip()

    def _process_passages_with_citation(self, passages_list: List[List[str]], survey: Dict[str, Any], state: Dict[str, Any], docs_list: List[List[Dict[str, Any]]] = None) -> str:
        """Process passages and assign citation IDs.

        Args:
            passages_list: List of passage lists (one per keyword)
            survey: Survey/outline structure
            state: Agent state
            docs_list: List of doc lists (one per keyword), matching passages_list structure

        Returns:
            Formatted string with cited passages
        """
        # Use configured seen_top_k for the number of passages to pass to components
        top_k = self.seen_top_k

        if not passages_list:
            return ""

        num_queries = len(passages_list)
        per_query_limit = max(1, top_k // num_queries)

        seen = set()
        all_passages = []

        # First pass: get unique passages up to per_query_limit from each query
        for query_idx, passages in enumerate(passages_list):
            docs = docs_list[query_idx] if docs_list and query_idx < len(docs_list) else []

            for psg_idx, psg in enumerate(passages[:per_query_limit]):
                # Skip empty passages
                if not psg or not psg.strip():
                    continue

                if psg not in seen:
                    seen.add(psg)
                    doc_id = None
                    if psg_idx < len(docs):
                        doc = docs[psg_idx]
                        doc_id = doc.get("doc_id") or doc.get("id")

                    citation_id = self._assign_citation_id(state, psg, doc_id)
                    truncated_psg = psg[:self.max_passage_chars] + "..." if len(psg) > self.max_passage_chars else psg
                    cited_psg = f"[{citation_id}] {truncated_psg}"
                    all_passages.append(cited_psg)

        # Second pass: fill remaining slots
        remaining_slots = top_k - len(all_passages)
        if remaining_slots > 0:
            for query_idx, passages in enumerate(passages_list):
                docs = docs_list[query_idx] if docs_list and query_idx < len(docs_list) else []

                for psg_idx, psg in enumerate(passages[per_query_limit:], start=per_query_limit):
                    if not psg or not psg.strip():
                        continue

                    if psg not in seen and remaining_slots > 0:
                        seen.add(psg)
                        doc_id = None
                        if psg_idx < len(docs):
                            doc = docs[psg_idx]
                            doc_id = doc.get("doc_id") or doc.get("id")

                        citation_id = self._assign_citation_id(state, psg, doc_id)
                        truncated_psg = psg[:self.max_passage_chars] + "..." if len(psg) > self.max_passage_chars else psg
                        cited_psg = f"[{citation_id}] {truncated_psg}"
                        all_passages.append(cited_psg)
                        remaining_slots -= 1

        return "\n\n".join(all_passages).strip()

    def _assign_citation_id(self, state: Dict[str, Any], doc_text: str, doc_id: str = None) -> int:
        """Assign a unique ID to a document and track its doc_id.

        Args:
            state: Agent state
            doc_text: Document text content
            doc_id: Original document ID from corpus (optional)

        Returns:
            Citation ID assigned to this document
        """
        doc_hash = doc_text.strip()

        if doc_hash in state["citation_registry"]:
            citation_id = state["citation_registry"][doc_hash]
        else:
            state["citation_counter"] += 1
            citation_id = state["citation_counter"]
            state["citation_registry"][doc_hash] = citation_id

        # Store citation_id -> doc_id mapping if doc_id is provided
        if doc_id and citation_id not in state["citation_to_doc_id"]:
            state["citation_to_doc_id"][citation_id] = doc_id
        elif not doc_id and self.verbose:
            # Log warning when doc_id is missing
            self._print(f"  ⚠ Citation [{citation_id}] assigned without doc_id")

        return citation_id

    def _extract_bibkeys(self, text: str) -> List[str]:
        """Extract citation IDs from text."""
        pattern = r"\[(\d+)\]"
        matches = re.findall(pattern, text)
        return list(set(matches))

    def _extract_doc_ids_from_cited_text(self, text: str, state: Dict[str, Any], limit: int = 20) -> tuple[int, list[str]]:
        """Extract doc IDs from text containing citations.

        Args:
            text: Text with citations like "[1] content"
            state: Agent state containing citation_to_doc_id mapping
            limit: Maximum number of doc IDs to return

        Returns:
            Tuple of (total_count, list of doc_ids limited to 'limit')
        """
        citation_ids = self._extract_bibkeys(text)
        doc_ids = []
        for cit_id in citation_ids:
            cit_id_int = int(cit_id)
            doc_id = state["citation_to_doc_id"].get(cit_id_int) or state["citation_to_doc_id"].get(str(cit_id_int))
            if doc_id:
                doc_ids.append(doc_id)

        return len(doc_ids), doc_ids[:limit]

    def _check_language_consistency(self, item: Any, user_instruction: str) -> bool:
        """Check if text language matches user instruction language."""
        # Extract text from various structures
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = ""
            for v in item.values():
                if isinstance(v, str):
                    text += v + "\n"
                elif isinstance(v, list):
                    for vv in v:
                        if isinstance(vv, str):
                            text += vv + "\n"
                        elif isinstance(vv, dict):
                            for vvv in vv.values():
                                if isinstance(vvv, str):
                                    text += vvv + "\n"
        elif isinstance(item, list):
            text = ""
            for v in item:
                if isinstance(v, str):
                    text += v + "\n"
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, str):
                            text += vv + "\n"
        else:
            return False

        text = text.strip()
        text = text.replace(" ", "").replace("\n", "").replace("\t", "")

        # Remove citations
        text = re.sub(r'\[(\d+)\]', '', text)

        # Remove punctuation
        comma_english = r'[!"#$%&\'()\*\+,-./:;<=>\?@\\\[\]^_`{\|}~1234567890]'
        text = re.sub(comma_english, "", text)

        if len(text) == 0:
            return True

        # Check language
        is_chinese = re.search(r'[\u4e00-\u9fff]', user_instruction) is not None

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_count = len(chinese_chars)
        total_chars = len(text)

        if is_chinese:
            return chinese_count / total_chars > 0.6
        else:
            return chinese_count / total_chars < 0.01

    def _format_final_report(self, survey: Dict[str, Any]) -> str:
        """Format survey as clean Markdown for final output."""
        if not survey or survey == {}:
            return "No report generated."

        lines = []

        # Title
        title = survey.get("title", "Untitled Report")
        lines.append(f"# {title}")
        lines.append("")

        # Sections
        sections = survey.get("sections", [])
        for i, section in enumerate(sections):
            title_key = "name" if "name" in section else "title"
            section_title = section.get(title_key, "")
            section_num = i + 1

            lines.append(f"## {section_num}. {section_title}")
            lines.append("")

            # Section content
            if "content" in section and section["content"]:
                lines.append(section["content"])
                lines.append("")
            elif "plan" in section and section["plan"]:
                lines.append(f"*{section['plan'].strip()}*")
                lines.append("")

            # Subsections
            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    subsection_title = subsection.get(title_key, "")
                    subsection_num = f"{section_num}.{j + 1}"

                    lines.append(f"### {subsection_num} {subsection_title}")
                    lines.append("")

                    if "content" in subsection and subsection["content"]:
                        lines.append(subsection["content"])
                        lines.append("")
                    elif "plan" in subsection and subsection["plan"]:
                        lines.append(f"*{subsection['plan'].strip()}*")
                        lines.append("")

                    # Sub-subsections
                    if "subsections" in subsection:
                        for k, subsubsection in enumerate(subsection["subsections"]):
                            subsubsection_title = subsubsection.get(title_key, "")
                            subsubsection_num = f"{section_num}.{j + 1}.{k + 1}"

                            lines.append(f"#### {subsubsection_num} {subsubsection_title}")
                            lines.append("")

                            if "content" in subsubsection and subsubsection["content"]:
                                lines.append(subsubsection["content"])
                                lines.append("")
                            elif "plan" in subsubsection and subsubsection["plan"]:
                                lines.append(f"*{subsubsection['plan'].strip()}*")
                                lines.append("")

        # Final cleanup
        result = "\n".join(lines)
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result.strip()

    def _generate_references_section(self, final_report: str, state: Dict[str, Any]) -> str:
        """Generate a References section with cited documents.

        Args:
            final_report: The formatted report text
            state: Agent state containing citation mappings

        Returns:
            References section markdown text
        """
        # Extract all unique citation numbers from the report
        citation_pattern = r'\[(\d+)\]'
        citations_in_report = re.findall(citation_pattern, final_report)
        unique_citations = sorted(set(int(c) for c in citations_in_report))

        if not unique_citations:
            return ""

        # Get citation mappings
        citation_to_doc_id = state.get("citation_to_doc_id", {})

        # Build references section
        lines = [
            "\n\n",
            "## References\n",
            "\n",
        ]

        # Check if we have doc_id mappings
        if citation_to_doc_id:
            lines.append(f"This report cites {len(unique_citations)} documents:\n\n")

            # Get all retrieved docs to find titles and contexts
            retrieved_docs_cache = {}  # Cache docs by doc_id

            # Build cache from retrieved_docs_per_search (each entry is a flat fused list)
            for search_docs in state.get("retrieved_docs_per_search", []):
                for doc in search_docs:
                    if isinstance(doc, dict):
                        doc_id = doc.get("id") or doc.get("doc_id")
                        if doc_id:
                            retrieved_docs_cache[doc_id] = doc

            # Generate citation entries
            for citation_id in unique_citations:
                # Get doc_id from mapping (check both int and string keys)
                doc_id = citation_to_doc_id.get(citation_id) or citation_to_doc_id.get(str(citation_id))

                if doc_id and doc_id in retrieved_docs_cache:
                    doc = retrieved_docs_cache[doc_id]
                    # Try to get title from multiple possible locations
                    # For local retrievers: doc["title"]
                    # For API endpoint: doc["metadata"]["title"]
                    title = doc.get("title") or (doc.get("metadata", {}).get("title") if isinstance(doc.get("metadata"), dict) else None) or "N/A"
                    # Get context from contents or text field
                    context = doc.get("contents") or doc.get("text") or doc.get("relevant_text") or "N/A"

                    # Truncate context if too long
                    if isinstance(context, str) and len(context) > 300:
                        context = context[:300] + "..."

                    lines.append(f"[{citation_id}] **{title}**\n")
                    lines.append(f"    {context}\n")
                    lines.append(f"    *Document ID: `{doc_id}`*\n\n")
                elif doc_id:
                    # Have doc_id but not in cache
                    lines.append(f"[{citation_id}] *Document ID: `{doc_id}`*\n")
                    lines.append(f"    *(Full details not available in retrieval cache)*\n\n")
                else:
                    # No doc_id mapping
                    lines.append(f"[{citation_id}] *(Citation mapping not available)*\n\n")
        else:
            # No citation_to_doc_id mapping
            lines.append(f"This report cites {len(unique_citations)} documents, but full citation details ")
            lines.append("are not available because the citation_to_doc_id mapping was not saved.\n\n")
            lines.append(f"**Citations used:** {', '.join(f'[{c}]' for c in unique_citations[:30])}")
            if len(unique_citations) > 30:
                lines.append(f", ... and {len(unique_citations) - 30} more\n")

        return ''.join(lines)

    # ==================== Main Orchestration ====================

    def inference(self, query: str, generation_temp: float = 0.7) -> tuple:
        """Run the CPM-Report state machine and return the standard 3-tuple.

        Returns:
            (reasoning_path, prediction, num_iterations)
        """
        self._default_temperature = generation_temp
        return asyncio.run(self._inference_async(query))

    async def _inference_async(self, query: str) -> tuple:
        """Async core of the CPM-Report state machine."""

        # Initialize state
        state = self._init_state(query)
        self._current_survey = state["survey"]

        # Oracle mode: skip initial search, generate outline from oracle aspects
        if self._oracle_data is not None:
            state = await self._init_plan_oracle(state, self._current_query_id)
            if state["parsed"] and state["cursor"] != "outline":
                state["state"] = "search"
                state["step"] += 1
            else:
                self._print(f"  ⚠ Oracle init plan failed, aborting")
                state["state"] = "done"

        _STAGE_LABELS = {
            "search":               "search",
            "analyst-init_plan":    "plan",
            "analyst-extend_plan":  "plan+",
            "write":                "write",
        }

        num_iterations = 0

        # State machine loop
        while state["state"] != "done":
            current_state = state["state"]

            self._notify_progress(
                _STAGE_LABELS.get(current_state, current_state),
                state.get("step", 0),
            )

            try:
                if current_state == "search":
                    state = await self._search(state)
                elif current_state == "analyst-init_plan":
                    state = await self._init_plan(state)
                elif current_state == "analyst-extend_plan":
                    state = await self._extend_plan(state)
                elif current_state == "write":
                    state = await self._write(state)
                else:
                    self._print(f"⚠ Unknown state: {current_state}")
                    break

                # Update state
                state = self._update_state(state)

                # Keep survey reference current for answer candidates
                self._current_survey = state["survey"]

                # Count iterations: plan steps and search-write pairs each count as 1
                num_iterations += 1

            except Exception as e:
                self._print(f"⚠ Error in state {current_state}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Format final report
        final_report = self._format_final_report(state["survey"])

        # Generate and append references section
        references_section = self._generate_references_section(final_report, state)
        if references_section:
            final_report = final_report.rstrip() + references_section
            self._print(f"  ✓ Appended References section")

        num_citations = len(state.get("citation_to_doc_id", {}))
        self._print(f"  → Citations tracked: {num_citations}")
        self._print(f"  → Total steps: {state['step']}")

        # Build reasoning_path from trajectory + CPM-specific extras
        reasoning_path = self._build_reasoning_path(state)

        # Stash state so run_single can attach CPM-specific extras
        self._last_state = state

        return (reasoning_path, final_report, num_iterations)

    def _build_reasoning_path(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert CPM-Report trajectory into the standardised reasoning_path format."""
        reasoning_path: List[Dict[str, Any]] = []
        for entry in state.get("trajectory", []):
            rp_entry: Dict[str, Any] = {
                "think": entry.get("think", ""),
                "action_type": entry.get("action", entry.get("state", "")),
            }
            traj_state = entry.get("state", "")

            if traj_state in ("search", "critical_search"):
                keywords = entry.get("input", {}).get("keywords", [])
                rp_entry["search_query"] = "; ".join(keywords) if keywords else ""
                all_docs = entry.get("all_docs", [])
                rp_entry["docs"] = all_docs
                rp_entry["all_docs"] = all_docs
                rp_entry["component_doc_ids"] = entry.get("output", {}).get("doc_ids", [])
                if entry.get("is_critical_think"):
                    rp_entry["is_critical_think"] = True

            elif traj_state == "write":
                rp_entry["docs"] = []
                rp_entry["component_doc_ids"] = entry.get("input", {}).get("retrieved_doc_ids", [])

            elif traj_state in ("analyst-init_plan", "analyst-init_plan_oracle",
                                "analyst-extend_plan"):
                rp_entry["docs"] = []
                rp_entry["component_doc_ids"] = entry.get("input", {}).get("retrieved_doc_ids", [])

            # Attach CPM-specific metadata for downstream evaluators
            rp_entry["_cpm_state"] = traj_state
            rp_entry["_cpm_step"] = entry.get("step")
            rp_entry["_cpm_output"] = entry.get("output", {})

            reasoning_path.append(rp_entry)

        return reasoning_path

    def run_single(self, query_id: str, query_text: str, temperature: float = 0.7, status_callback=None) -> Optional[Dict[str, Any]]:
        """Override to attach CPM-specific extras to the base result."""
        self._current_query_id = query_id
        self._current_survey = {}
        result = super().run_single(query_id, query_text, temperature, status_callback)
        if result is None:
            return None

        # Attach CPM-specific keys that evaluators need
        state_extras = getattr(self, "_last_state", None)
        if state_extras:
            result["citation_to_doc_id"] = state_extras.get("citation_to_doc_id", {})
            result["survey"] = state_extras.get("survey", {})
            result["keywords_history"] = state_extras.get("keywords_history", [])
            result["search_history"] = state_extras.get("search_history", [])
            result["retrieved_docs_per_search"] = state_extras.get("retrieved_docs_per_search", [])
        return result

