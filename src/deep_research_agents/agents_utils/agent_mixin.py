"""Shared mixin for verbose logging and trajectory-tracker integration.

Provides ``_print``, ``_vprint``, ``_vprint_docs``, and
``post_search_evaluate`` so that every agent type (BasicAgent, AgentCPM)
shares a single implementation instead of duplicating the same code.

The mixin reads ``self.name`` (or ``self.AGENT_NAME``) for the display
label and ``self.verbose`` for the verbosity flag.  If ``self.verbose``
is not set it defaults to ``True``.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from agent_tools.trajectory_tracker import TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult
# post_search_evaluate returns either critical_think, deferred, or early-stop result
_TrackerResult = Union[TrackerCriticalThinkDeferred, TrackerCriticalThinkResult, TrackerEarlyStopResult]
from utils.doc_formatting import passages2string
from utils.printing import verbose_print, verbose_print_search_results, verbose_print_tracker

logger = logging.getLogger(__name__)


class AgentVerboseMixin:
    """Mixin providing verbose logging and trajectory-tracker helpers."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _display_name(self) -> str:
        """Agent label used for log lines."""
        return getattr(self, "name", None) or getattr(self, "AGENT_NAME", "Agent")

    @property
    def _is_verbose(self) -> bool:
        return getattr(self, "verbose", True)

    # ------------------------------------------------------------------
    # Verbose printing
    # ------------------------------------------------------------------

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self._is_verbose:
            print(f"[{self._display_name}] {message}")

    def _vprint(
        self,
        iter_num: int,
        component: str,
        message: str,
        *,
        sub_iter: int = None,
    ) -> None:
        """Print a consistently formatted verbose line if verbose is enabled."""
        if self._is_verbose:
            verbose_print(
                iter_num, component, message,
                agent_name=self._display_name, sub_iter=sub_iter,
            )

    def _vprint_docs(
        self, iter_num: int, docs: list, *, sub_iter: int = None,
    ) -> None:
        """Print top search results if verbose is enabled."""
        if self._is_verbose:
            verbose_print_search_results(
                iter_num, docs,
                agent_name=self._display_name, sub_iter=sub_iter,
            )

    def _vprint_tracker(
        self, iter_num: int, scores: dict, action: str,
        *, sub_iter: int = None,
    ) -> None:
        """Print tracker scores if verbose is enabled."""
        if self._is_verbose:
            verbose_print_tracker(
                iter_num, scores, action,
                agent_name=self._display_name, sub_iter=sub_iter,
            )

    # ------------------------------------------------------------------
    # Critical-think helpers
    # ------------------------------------------------------------------

    def _critical_think_to_reasoning_entry(
        self, ct: TrackerCriticalThinkResult, *, include_all_docs: bool = False,
    ) -> dict:
        """Build a reasoning_path entry from a TrackerCriticalThinkResult."""
        seen_top_k = getattr(self, "seen_top_k", 5)
        entry = {
            "action_type": "critical_search",
            "think": ct.critical_think,
            "search_query": ct.critical_search_query,
            "docs": ct.critical_docs,
            "component_doc_ids": [
                d.get("doc_id", "") for d in ct.critical_docs[:seen_top_k]
            ],
            "is_critical_think": True,
        }
        if include_all_docs:
            entry["all_docs"] = ct.critical_docs
        return entry

    @staticmethod
    def _format_critical_redirect_text(ct: TrackerCriticalThinkResult) -> str:
        """Format the critical-redirect observation text injected into agent context."""
        return (
            f"\n\n[Critical Redirect — {ct.critical_search_query}]\n"
            f"{ct.critical_observation}"
        )

    # ------------------------------------------------------------------
    # Trajectory tracker integration
    # ------------------------------------------------------------------

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
        """Evaluate a search step via the trajectory tracker.

        Call this after each search to let the tracker decide whether to
        inject an observation.

        Returns:
            ``None``: tracker is disabled or decides "continue".
            ``TrackerCriticalThinkResult``: prompt-notice mode — the tracker
                generated a critical search query, executed retrieval,
                and re-evaluated.  The agent should inject this as an
                additional full turn in its trajectory.
            ``TrackerCriticalThinkDeferred``: returned when
                ``defer_critical_search=True`` and action is "intervene".
                Carries the decision info without executing retrieval.
                Call ``_execute_deferred_critical_search`` to resolve.

        Args:
            subquery: A single subquery string or a list of subqueries
                issued in this iteration (e.g. AgentCPM, webWeaver).
            seen_docs: The subset of docs shown to the agent (for verbose
                printing).  When provided, this method handles the
                doc/notice printing: if the tracker injects a notice the
                notice is printed instead of the retrieved docs; otherwise
                the docs are printed as usual.  Callers that pass
                ``seen_docs`` should **not** call ``_vprint_docs``
                separately.
            sub_iter: Optional sub-iteration for nested loops.
            defer_critical_search: When True, an "intervene" decision returns
                a ``TrackerCriticalThinkDeferred`` without executing
                retrieval.  Used by agents that process multiple sub-queries
                per iteration and need to finish all sub-queries before
                running the critical search.

        In verbose mode, prints tracker diagnostics (supervised recall,
        signals, decision maker output) for the user's visibility.
        This info is NOT seen by the agent.
        """
        tracker = getattr(self, "trajectory_tracker", None)
        if tracker is None:
            if seen_docs is not None:
                self._vprint_docs(iter_num, seen_docs, sub_iter=sub_iter)
            return None

        decision = tracker.evaluate(
            subquery=subquery,
            docs=docs,
            original_query=original_query or "",
            iter_num=iter_num,
            thinking=thinking,
            trajectory=trajectory,
            reasoning_path=reasoning_path,
            **kwargs,
        )

        critical_think_triggered = (
            decision.critical_thinking_output is not None
            and decision.critical_thinking_output.search_query.strip()
        )

        if self._is_verbose:
            if seen_docs is not None:
                self._vprint_docs(iter_num, seen_docs, sub_iter=sub_iter)
            self._vprint_tracker(
                iter_num, decision.scores, decision.action,
                sub_iter=sub_iter,
            )
            if critical_think_triggered:
                self._vprint(
                    iter_num, "notice",
                    f"[{decision.action}] critical_think"
                    + (" (deferred)" if defer_critical_search else ""),
                    sub_iter=sub_iter,
                )

        # --- Prompt-notice critical_think: execute critical search query ----------
        if critical_think_triggered:
            pno = decision.critical_thinking_output

            if defer_critical_search:
                return TrackerCriticalThinkDeferred(
                    critical_think=pno.reasoning,
                    critical_search_query=pno.search_query,
                    scores=decision.scores,
                    iter_num=iter_num,
                )

            retrieve_fn = getattr(self, "retrieve_documents", None)
            if retrieve_fn is None:
                logger.warning("Tracker critical_think requested but no retriever available")
                return None

            critical_think_iter = iter_num + 1
            self._notify_progress("critical_think", critical_think_iter)
            critical_docs = retrieve_fn(
                pno.search_query, original_query=original_query,
            )
            self._notify_progress("critical_search", critical_think_iter)
            seen_top_k = getattr(self, "seen_top_k", 5)
            critical_observation = passages2string(critical_docs[:seen_top_k])

            if self._is_verbose:
                self._vprint(
                    critical_think_iter, "critical think",
                    pno.reasoning or "(no reasoning)",
                    sub_iter=sub_iter,
                )
                self._vprint(
                    critical_think_iter, "critical search",
                    pno.search_query,
                    sub_iter=sub_iter,
                )
                self._vprint_docs(critical_think_iter, critical_docs[:seen_top_k], sub_iter=sub_iter)

            critical_think_decision = tracker.evaluate(
                subquery=pno.search_query,
                docs=critical_docs[:seen_top_k],
                original_query=original_query or "",
                iter_num=critical_think_iter,
                thinking=pno.reasoning,
                trajectory=trajectory,
                reasoning_path=reasoning_path,
            )

            if self._is_verbose:
                self._vprint_tracker(
                    critical_think_iter, critical_think_decision.scores,
                    critical_think_decision.action, sub_iter=sub_iter,
                )

            return TrackerCriticalThinkResult(
                critical_think=pno.reasoning,
                critical_search_query=pno.search_query,
                critical_docs=critical_docs,
                critical_observation=critical_observation,
                critical_think_scores=critical_think_decision.scores,
                critical_think_iter=critical_think_iter,
            )

        # --- Early stopping: signal the agent to produce its answer ----------
        early_stop = getattr(decision, "early_stopping_output", None)
        if early_stop is not None:
            if self._is_verbose:
                self._vprint(
                    iter_num, "notice",
                    f"[{decision.action}] early_stopping",
                    sub_iter=sub_iter,
                )
            return TrackerEarlyStopResult(
                reasoning=early_stop.reasoning,
                scores=decision.scores,
            )

        return None

    def _execute_deferred_critical_search(
        self,
        deferred: TrackerCriticalThinkDeferred,
        original_query: str,
        trajectory: Any = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[TrackerCriticalThinkResult]:
        """Execute a deferred critical search and return the full result.

        Called after all sub-queries in an iteration have completed, so that
        the critical search runs at the end of the iteration rather than
        in the middle.
        """
        retrieve_fn = getattr(self, "retrieve_documents", None)
        if retrieve_fn is None:
            logger.warning("Tracker critical_think requested but no retriever available")
            return None

        tracker = getattr(self, "trajectory_tracker", None)
        critical_think_iter = deferred.iter_num + 1

        self._notify_progress("critical_think", critical_think_iter)
        critical_docs = retrieve_fn(
            deferred.critical_search_query, original_query=original_query,
        )
        self._notify_progress("critical_search", critical_think_iter)
        seen_top_k = getattr(self, "seen_top_k", 5)
        critical_observation = passages2string(critical_docs[:seen_top_k])

        if self._is_verbose:
            self._vprint(
                critical_think_iter, "critical think",
                deferred.critical_think or "(no reasoning)",
            )
            self._vprint(
                critical_think_iter, "critical search",
                deferred.critical_search_query,
            )
            self._vprint_docs(critical_think_iter, critical_docs[:seen_top_k])

        critical_think_scores = {}
        if tracker is not None:
            critical_think_decision = tracker.evaluate(
                subquery=deferred.critical_search_query,
                docs=critical_docs[:seen_top_k],
                original_query=original_query,
                iter_num=critical_think_iter,
                thinking=deferred.critical_think,
                trajectory=trajectory,
                reasoning_path=reasoning_path,
            )
            critical_think_scores = critical_think_decision.scores
            if self._is_verbose:
                self._vprint_tracker(
                    critical_think_iter, critical_think_scores,
                    critical_think_decision.action,
                )

        return TrackerCriticalThinkResult(
            critical_think=deferred.critical_think,
            critical_search_query=deferred.critical_search_query,
            critical_docs=critical_docs,
            critical_observation=critical_observation,
            critical_think_scores=critical_think_scores,
            critical_think_iter=critical_think_iter,
        )

    def _reset_tracker(self, query_id: Optional[str] = None) -> None:
        """Reset the trajectory tracker for a new query.

        Should be called at the start of ``run_single`` before inference
        begins.  No-op when the tracker is not set.
        """
        self._search_step = 0
        tracker = getattr(self, "trajectory_tracker", None)
        if tracker is not None:
            tracker.reset(query_id=query_id)

    def _attach_tracker_stats(self, result: dict) -> None:
        """Attach trajectory-tracker statistics to a result dict.

        Should be called by ``run_single`` (or the async wrapper that
        builds the result dict) before returning.
        """
        tracker = getattr(self, "trajectory_tracker", None)
        if tracker is not None:
            result["tracker_score_history"] = list(tracker.score_history)
            result["tracker_unique_doc_ids"] = sorted(tracker.unique_doc_ids)
            result["tracker_unique_doc_count"] = tracker.unique_doc_count
            result["tracker_answer_candidates"] = list(tracker.answer_candidates)
