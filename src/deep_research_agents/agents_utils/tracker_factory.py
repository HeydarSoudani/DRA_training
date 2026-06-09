"""Factory function for building TrajectoryTracker instances."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_trajectory_tracker(
    tt_mode: str,
    retriever=None,
    qrels=None,
    seen_top_k: int = 5,
    llm_decision_maker: Optional[str] = None,
    llm_intervene: Optional[str] = None,
    agent=None,
    agentic_model: Optional[str] = None,
    strict: bool = True,
    decision_maker_history_window: Optional[int] = None,
    dm_prompt_variant: str = "nov_cov_sim",
    max_iteration: Optional[int] = None,
    aspect_coverage_mode: str = "dynamic",
    aspect_coverage_max_aspects: int = 8,
    llm_aspect_coverage: Optional[str] = None,
    ac_temperature: float = 0.0,
    dataset: Optional[str] = None,
):
    """Build a TrajectoryTracker with decision maker and answer candidate generator.

    Shared by both the main process (run_pipeline) and spawned GPU workers.

    Args:
        tt_mode: "off", "monitor", or "decision_maker".
        retriever: Retriever instance (for encoding function).
        qrels: Ground-truth relevance judgements.
        seen_top_k: Number of docs considered "seen" per step.
        llm_decision_maker: Model name for decision maker LLM.
        llm_intervene: Model name for critical thinking generator LLM.
        agent: Agent instance (to wire up answer candidate generator).
        agentic_model: Agent type name (for format selection).
        strict: If True, raise ValueError for missing required params.
        aspect_coverage_mode: "static" or "dynamic".
        aspect_coverage_max_aspects: Soft cap on the number of aspects.
        llm_aspect_coverage: Model name for aspect coverage LLM. Falls
            back to llm_decision_maker or llm_intervene if not set.

    Returns:
        TrajectoryTracker instance, or None if tt_mode is "off".
    """
    if tt_mode == "off":
        return None

    from agent_tools import TrajectoryTracker, LLMCriticalThinkingGenerator, LLMDecisionMaker
    from agent_tools.trajectory_tracker import encode_fn_from_retriever
    from prompts.trajectory_tracker.answer_prompts import get_candidate_format

    _tt_affect = "none" if tt_mode == "monitor" else "active"

    # CT generator needed when "intervene" action is possible
    _tt_notice_gen = None
    if tt_mode == "decision_maker":
        if llm_intervene:
            from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
            _tt_llm_client = LiteLLMClient(model=llm_intervene)
            _tt_notice_gen = LLMCriticalThinkingGenerator(llm_client=_tt_llm_client)
        elif strict:
            raise ValueError("--llm-intervene is required when using intervene action")

    # Build decision maker (used in both monitor and decision_maker modes)
    _decision_maker = None
    _dm_llm_model = llm_decision_maker or llm_intervene
    if _dm_llm_model:
        from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
        _dm_llm_client = LiteLLMClient(model=_dm_llm_model)
        _decision_maker = LLMDecisionMaker(llm_client=_dm_llm_client, history_window=decision_maker_history_window, dm_prompt_variant=dm_prompt_variant, max_iteration=max_iteration)
    elif tt_mode == "decision_maker" and strict:
        raise ValueError(
            "--llm-decision-maker (or --llm-intervene) is required "
            "when --trajectory-tracker=decision_maker"
        )

    tracker = TrajectoryTracker(
        affect_mode=_tt_affect,
        critical_thinking_generator=_tt_notice_gen,
        encode_fn=encode_fn_from_retriever(retriever) if retriever else None,
        qrels=qrels or {},
        seen_top_k=seen_top_k,
        decision_maker=_decision_maker,
    )

    # Wire up aspect coverage signal (always-on; dm_prompt_variant controls what the DM sees)
    _ac_llm_model = llm_aspect_coverage or llm_decision_maker or llm_intervene
    if _ac_llm_model:
        from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
        from agent_tools.trajectory_tracker import AspectCoverageSignal
        _ac_llm_client = LiteLLMClient(model=_ac_llm_model)
        tracker._aspect_coverage = AspectCoverageSignal(
            llm_client=_ac_llm_client,
            mode=aspect_coverage_mode,
            max_aspects=aspect_coverage_max_aspects,
            temperature=ac_temperature,
        )
        print(f"Aspect coverage signal enabled: mode={aspect_coverage_mode}, model={_ac_llm_model}")
    else:
        logger.warning("No LLM model available for aspect coverage; signal disabled")

    # Wire up answer candidate generation through the agent (browsecomp_plus only)
    if agent is not None and agentic_model is not None:
        # Always set the canonical per-agent format on inference_config
        _cfg = getattr(agent, "inference_config", None)
        if _cfg is not None:
            _cfg.format_instructions = get_candidate_format(agentic_model)

        if dataset == "browsecomp_plus" and agentic_model != "cpm_report":
            if hasattr(agent, "generate_answer_candidate"):
                tracker._answer_candidate_fn = agent.generate_answer_candidate
                _model_id = _cfg.model_name if _cfg else "unknown"
                print(f"Answer candidate via agent.generate_answer_candidate: {_model_id}")
            else:
                pass

    return tracker
