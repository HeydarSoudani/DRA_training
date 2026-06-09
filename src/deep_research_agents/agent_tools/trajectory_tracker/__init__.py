"""Trajectory tracker: post-search quality gate for deep research agents.

After each search step the tracker receives the sub-query and retrieved
documents, computes signals from three sources
(doc novelty, consecutive query similarity, original query similarity),
and decides whether the agent should continue normally or receive an
injected observation.

Decisions:
- **continue**: trajectory is healthy — continue normally.
- **intervene**: trajectory is stale — inject critical thinking and a new search.
- **stop**: trajectory is exhausted — force final answer generation.

Usage (inside any agent's inference loop)::

    tracker_result = self.post_search_evaluate(
        subquery=search_query,       # str or List[str]
        docs=search_docs,
        iter_num=iter_num,
        original_query=self.query,
    )
    if isinstance(tracker_result, TrackerCriticalThinkResult):
        # inject critical_think as additional trajectory step
    elif isinstance(tracker_result, TrackerEarlyStopResult):
        # break loop and force answer generation

Module layout:
    tracker_types      — dataclasses and constants
    signals            — DocNoveltySignal, ConsecQuerySimilaritySignal,
                         OrigQuerySimilaritySignal, AspectCoverageSignal,
                         encode_fn_from_retriever
    decision_makers    — DecisionMaker, LLMDecisionMaker
    generators         — CriticalThinkingGenerator, LLMCriticalThinkingGenerator,
                         LLMAnswerCandidateGenerator, ResponsesAPICandidateGenerator
    trajectory_tracker — TrajectoryTracker (this file)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Re-export all public symbols so existing ``from agent_tools.trajectory_tracker import X``
# statements continue to work after the split.
from .tracker_types import (  # noqa: F401
    AFFECT_NONE,
    Decision,
    EarlyStoppingOutput,
    CriticalThinkingOutput,
    TrackerCriticalThinkDeferred,
    TrackerCriticalThinkResult,
    TrackerEarlyStopResult,
    TrajectoryDecision,
    TrajectoryTurn,
    TrajectoryContext,
)
from .signals import (  # noqa: F401
    DocNoveltySignal,
    ConsecQuerySimilaritySignal,
    OrigQuerySimilaritySignal,
    AspectCoverageSignal,
    encode_fn_from_retriever,
)
from .decision_makers import (  # noqa: F401
    DecisionMaker,
    LLMDecisionMaker,
)
from .generators import (  # noqa: F401
    CriticalThinkingGenerator,
    LLMCriticalThinkingGenerator,
    LLMAnswerCandidateGenerator,
    ResponsesAPICandidateGenerator,
)

from prompts.trajectory_tracker.answer_prompts import FINAL_ANSWER_INSTRUCTION
from prompts.trajectory_tracker.aspect_coverage import (
    AspectCoverageSummary,
    format_summary_for_log,
)

logger = logging.getLogger(__name__)


class TrajectoryTracker:
    """Post-search trajectory quality gate.

    Computes three analysis-aligned signals (doc novelty, consecutive
    query similarity, original query similarity) and delegates the
    continue/intervene/stop decision to a pluggable ``DecisionMaker``.

    - **continue**: trajectory is healthy — proceed normally.
    - **intervene**: trajectory is stale — inject critical thinking and
      a new search iteration.
    - **stop**: trajectory is exhausted — force final answer generation.

    Args:
        affect_mode: ``"none"`` for monitor-only (compute and log scores,
            never inject observations).  Any other value enables active
            intervention based on the decision maker's decisions.
        critical_thinking_generator: Instance of ``CriticalThinkingGenerator``
            (or subclass) used when action is ``"intervene"``.
        min_iter: Don't fire on the first N iterations (need history).
        encode_fn: Callable ``(List[str]) -> np.ndarray`` that encodes
            texts into embeddings.  Required for query similarity signals.
            When ``None``, the composite score falls back to doc_novelty only.
            Use ``encode_fn_from_retriever(retriever)`` to extract this from
            the retriever object.
        qrels: Optional ground-truth relevance judgements, mapping
            ``{query_id: {doc_id: relevance_score}}``.  When provided,
            ``evaluate()`` computes marginal recall at each step.
        seen_top_k: Number of top docs per iteration that are actually
            "seen" by the agent.  Novelty and marginal recall are computed
            over only the first ``seen_top_k`` docs (default 5).
    """

    def __init__(
        self,
        affect_mode: str = AFFECT_NONE,
        critical_thinking_generator: Optional[CriticalThinkingGenerator] = None,
        min_iter: int = 0,
        n_last_turns: Optional[int] = 10,
        encode_fn: Optional[Callable] = None,
        qrels: Optional[Dict[str, Dict[str, Any]]] = None,
        seen_top_k: int = 5,
        decision_maker: Optional[DecisionMaker] = None,
    ) -> None:
        self.affect_mode = affect_mode
        self._critical_thinking_generator = critical_thinking_generator
        self._answer_candidate_generator: Optional[LLMAnswerCandidateGenerator] = None
        self._answer_candidate_fn: Optional[Callable] = None
        self._n_last_turns = n_last_turns

        self.min_iter = min_iter
        self.seen_top_k = seen_top_k

        self._qrels = qrels or {}
        self._current_relevant_ids: Set[str] = set()
        self._gold_seen_so_far: Set[str] = set()

        self._doc_novelty = DocNoveltySignal()
        self._consec_query_sim = ConsecQuerySimilaritySignal(encode_fn)
        self._orig_query_sim = OrigQuerySimilaritySignal(encode_fn)

        self._decision_maker = decision_maker
        self._aspect_coverage: Optional[AspectCoverageSignal] = None

        self._turns: List[TrajectoryTurn] = []
        self.score_history: List[Dict[str, Any]] = []
        self.answer_candidates: List[Dict[str, str]] = []

    @property
    def unique_doc_ids(self) -> Set[str]:
        """All unique document IDs seen across the current trajectory."""
        return self._doc_novelty._seen_ids

    @property
    def unique_doc_count(self) -> int:
        """Number of unique documents seen across the current trajectory."""
        return len(self._doc_novelty._seen_ids)

    def reset(self, query_id: Optional[str] = None) -> None:
        """Reset state for a new query.

        Args:
            query_id: When provided and qrels are available, loads the
                relevant doc IDs for this query to enable supervised recall.
        """
        self._doc_novelty.reset()
        self._turns.clear()
        self._consec_query_sim.reset()
        self._orig_query_sim.reset()
        if self._decision_maker is not None:
            self._decision_maker.reset()
        if self._aspect_coverage is not None:
            self._aspect_coverage.reset()
        self.score_history.clear()
        self.answer_candidates.clear()

        self._gold_seen_so_far.clear()
        if query_id and self._qrels:
            self._current_relevant_ids = {
                d for d, r in self._qrels.get(query_id, {}).items() if r > 0
            }
        else:
            self._current_relevant_ids = set()

    def _compute_marginal_recall(self, seen_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute supervised marginal recall for a single search step."""
        current_ids = {
            doc.get("doc_id") or doc.get("id") or ""
            for doc in seen_docs
        }
        current_ids.discard("")
        num_docs_this_step: int = len(current_ids)

        if self._current_relevant_ids and num_docs_this_step > 0:
            all_gold_this_step = current_ids & self._current_relevant_ids
            new_gold = all_gold_this_step - self._gold_seen_so_far
            num_new_relevant = len(new_gold)
            num_repeated_relevant = len(all_gold_this_step) - num_new_relevant
            num_irrelevant = num_docs_this_step - len(all_gold_this_step)
            marginal_recall = num_new_relevant / num_docs_this_step
            self._gold_seen_so_far.update(new_gold)
        elif num_docs_this_step > 0:
            num_new_relevant = 0
            num_repeated_relevant = 0
            num_irrelevant = num_docs_this_step
            marginal_recall = 0.0
        else:
            num_new_relevant = 0
            num_repeated_relevant = 0
            num_irrelevant = 0
            marginal_recall = 0.0

        return {
            "marginal_recall": marginal_recall,
            "num_new_relevant": num_new_relevant,
            "num_repeated_relevant": num_repeated_relevant,
            "num_irrelevant": num_irrelevant,
            "num_docs_this_step": num_docs_this_step,
        }

    def evaluate(self, subquery: Union[str, List[str]], docs: List[Dict[str, Any]], original_query: str, iter_num: int, thinking: str = "", trajectory: Any = None, reasoning_path: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> TrajectoryDecision:
        """Evaluate the search step and decide whether to inject an observation.

        Args:
            subquery: A single subquery string or a list of subqueries.
            docs: Retrieved documents for this iteration.
            original_query: The original user query.
            iter_num: The current iteration number.
        """
        subqueries: List[str] = [subquery] if isinstance(subquery, str) else list(subquery)

        self._turns.append(TrajectoryTurn(
            thinking=thinking,
            search_query="; ".join(subqueries),
        ))

        # --- Signals ----------------------------------------------------------
        novelty_score, num_novel = self._doc_novelty.score(docs)
        consec_sim = self._consec_query_sim.score(subqueries, iter_num)
        orig_sim = self._orig_query_sim.score(subqueries, original_query, iter_num)

        raw_signals = {
            "doc_novelty": novelty_score,
            "consec_query_sim": consec_sim,
            "orig_query_sim": orig_sim,
        }

        # --- Supervised marginal recall ---------------------------------------
        recall_stats = self._compute_marginal_recall(docs)
        marginal_recall = recall_stats["marginal_recall"]
        num_new_relevant = recall_stats["num_new_relevant"]
        num_repeated_relevant = recall_stats["num_repeated_relevant"]
        num_irrelevant = recall_stats["num_irrelevant"]
        num_docs_this_step = recall_stats["num_docs_this_step"]

        # --- Answer candidate generation --------------------------------------
        answer_candidates_iter: List[Dict[str, str]] = []
        _has_ac = self._answer_candidate_fn is not None or self._answer_candidate_generator is not None
        if _has_ac and (reasoning_path is not None or trajectory is not None):
            if self._answer_candidate_fn is not None:
                ac_outputs = self._answer_candidate_fn(
                    original_query=original_query,
                    trajectory=trajectory,
                    reasoning_path=reasoning_path,
                    seen_top_k=self.seen_top_k,
                )
            else:
                ac_outputs = self._answer_candidate_generator.generate(
                    original_query=original_query,
                    trajectory=trajectory,
                    reasoning_path=reasoning_path,
                    seen_top_k=self.seen_top_k,
                )
            logger.debug("Answer candidate generator returned %d outputs", len(ac_outputs))
            for ac in ac_outputs:
                entry = {"candidate": ac.candidate, "reasoning": ac.reasoning, "confidence": ac.confidence}
                answer_candidates_iter.append(entry)
                if ac.candidate.lower() != "no candidate":
                    self.answer_candidates.append(entry)
        elif not _has_ac:
            logger.debug("Answer candidate generator not configured; skipping.")
        elif trajectory is None and reasoning_path is None:
            logger.debug("No trajectory provided; skipping answer candidate generation.")

        # --- Aspect coverage --------------------------------------------------
        aspect_summary: Optional[AspectCoverageSummary] = None
        if self._aspect_coverage is not None:
            try:
                aspect_summary = self._aspect_coverage.update(
                    iter_num=iter_num,
                    docs=docs,
                    subqueries=subqueries,
                    query=original_query,
                )
                logger.info(format_summary_for_log(aspect_summary))
            except Exception:
                logger.warning("AspectCoverageSignal: update failed", exc_info=True)

        # --- Grace period -----------------------------------------------------
        _in_grace_period = iter_num <= self.min_iter

        # --- Online learning: reward for previous action ----------------------
        reward_signal = getattr(self._decision_maker, "reward_signal", None)
        if reward_signal and self.score_history:
            if reward_signal == "marginal_recall":
                reward = marginal_recall
            else:
                reward_val = raw_signals.get(reward_signal)
                reward = reward_val if reward_val is not None else 0.0
            self._decision_maker.update(reward)

        # --- Decision maker ---------------------------------------------------
        if self._decision_maker is not None and not _in_grace_period:
            decision = self._decision_maker.decide(
                raw_signals, iter_num,
                original_query=original_query,
                turns=self._turns,
                score_history=self.score_history,
                num_novel_docs=num_novel,
                num_total_docs=len(docs),
                answer_candidates=answer_candidates_iter,
                aspect_summary=aspect_summary,
            )
        else:
            decision = Decision(action="continue")
        action = decision.action

        # --- Scores dict ------------------------------------------------------
        scores: Dict[str, Any] = {
            "iter_num": iter_num,
            "subqueries": subqueries,
            "doc_novelty": round(novelty_score, 3),
            "num_novel_docs": num_novel,
            "num_total_docs": len(docs),
            "consec_query_sim": round(consec_sim, 3) if consec_sim is not None else None,
            "orig_query_sim": round(orig_sim, 3) if orig_sim is not None else None,
            "marginal_recall": round(marginal_recall, 3),
            "num_new_relevant": num_new_relevant,
            "num_repeated_relevant": num_repeated_relevant,
            "num_irrelevant": num_irrelevant,
            "num_docs_this_step": num_docs_this_step,
            "answer_candidates": answer_candidates_iter,
            "aspect_coverage": aspect_summary.to_dict() if aspect_summary is not None else None,
            **decision.scores,
        }

        self.score_history.append(scores)

        if _in_grace_period:
            return TrajectoryDecision(action="continue", scores=scores)

        if self.affect_mode == AFFECT_NONE:
            return TrajectoryDecision(action=action, scores=scores)

        # --- Act on decision --------------------------------------------------
        if action == "continue":
            return TrajectoryDecision(action="continue", scores=scores)

        if action == "intervene":
            if self._critical_thinking_generator is None:
                return TrajectoryDecision(action="continue", scores=scores)
            ctx = TrajectoryContext(
                original_query=original_query,
                last_turns=list(self._turns) if self._n_last_turns is None else list(self._turns[-self._n_last_turns:]),
                current_subqueries=subqueries,
                action=action,
                current_scores=scores,
                score_history=list(self.score_history),
            )
            pno = self._critical_thinking_generator.generate(ctx)
            if pno.search_query.strip():
                return TrajectoryDecision(
                    action="intervene",
                    scores=scores,
                    critical_thinking_output=pno,
                )
            return TrajectoryDecision(action="continue", scores=scores)

        if action == "stop":
            return TrajectoryDecision(
                action="stop",
                scores=scores,
                early_stopping_output=EarlyStoppingOutput(
                    reasoning=FINAL_ANSWER_INSTRUCTION,
                ),
            )

        return TrajectoryDecision(action="continue", scores=scores)
