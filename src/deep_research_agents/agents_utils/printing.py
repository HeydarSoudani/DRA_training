"""Verbose printing helpers for deep research agent pipelines.

These functions produce consistently formatted log output for the user's
terminal.  They are pure display helpers -- not seen by the agent.
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core verbose-print
# ---------------------------------------------------------------------------

def verbose_print(
    iter_num: int,
    component: str,
    message: str,
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
) -> None:
    """Print a consistently formatted verbose line.

    Format::

        [Iter 1] [think]: I need to find information about ...
        [Iter 1] [search]: Palestinian leaders reaction to Abraham Accords
        [Iter 1] [ret result]: (top 3 docs shown)

    Args:
        iter_num:   Current iteration / step number (1-based).
        component:  Component label (e.g. ``"think"``, ``"search"``, ``"plan"``).
        message:    The content to print.
        agent_name: Optional agent name prefix (e.g. ``"ReAct"``).
        sub_iter:   Optional sub-iteration for nested loops (e.g. ``1.2``).
    """
    prefix = f"[{agent_name}] " if agent_name else ""
    if sub_iter is not None:
        iter_label = f"[Iter {iter_num}.{sub_iter}]"
    else:
        iter_label = f"[Iter {iter_num}]"
    # Truncate very long messages to keep output readable
    if len(message) > 2000:
        message = message[:2000] + "..."
    print(f"{prefix}{iter_label} [{component}]: {message}")


# ---------------------------------------------------------------------------
# Search-result display
# ---------------------------------------------------------------------------

def verbose_print_search_results(
    iter_num: int,
    docs: List[Dict[str, Any]],
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
    top_n: int = 5,
    max_context_len: int = 100,
) -> None:
    """Print top search results in a compact, consistent format.

    Format::

        [Iter 1] [ret result]: [1] doc_id_1, Title of Document, First 100 chars of text...
                               [2] doc_id_2, Title of Document, First 100 chars of text...
                               [3] doc_id_3, Title of Document, First 100 chars of text...
                               ... and 97 more docs

    Args:
        iter_num:        Current iteration / step number.
        docs:            List of retrieved document dicts.
        agent_name:      Optional agent name prefix.
        sub_iter:        Optional sub-iteration for nested loops.
        top_n:           Number of top docs to display (default 3).
        max_context_len: Max chars of ``relevant_text`` to show per doc (default 100).
    """
    prefix = f"[{agent_name}] " if agent_name else ""
    if sub_iter is not None:
        iter_label = f"[Iter {iter_num}.{sub_iter}]"
    else:
        iter_label = f"[Iter {iter_num}]"

    if not docs:
        print(f"{prefix}{iter_label} [ret result]: (no docs retrieved)")
        return

    for rank, doc in enumerate(docs[:top_n], 1):
        doc_id = doc.get("doc_id") or doc.get("id", "?")
        title = doc.get("title") or doc.get("metadata", {}).get("title", "") or "Untitled"
        context = doc.get("relevant_text") or doc.get("text") or doc.get("contents", "")
        if len(context) > max_context_len:
            context = context[:max_context_len] + "..."
        # Collapse newlines for compact display
        context = context.replace("\n", " ").strip()
        print(f"{prefix}{iter_label} [ret result]: [{rank}] {doc_id}, {title}, {context}")

    remaining = len(docs) - top_n
    if remaining > 0:
        print(f"{prefix}{iter_label} [ret result]: ... and {remaining} more docs")


# ---------------------------------------------------------------------------
# Trajectory-tracker display
# ---------------------------------------------------------------------------

def verbose_print_tracker(
    iter_num: int,
    scores: Dict[str, Any],
    action: str,
    *,
    agent_name: str = "",
    sub_iter: Optional[int] = None,
) -> None:
    """Print trajectory tracker info for the current search step.

    This output is for the user's visibility only -- it is NOT seen by the
    agent.  Printed after the observation in verbose mode.

    Format::

        [ReAct] [Iter 1] [tracker]: marginal_recall=0.050 (1/20) | novelty=0.800 | consec_sim=0.743 | orig_sim=0.620
                                     dm_action=continue

    Args:
        iter_num:   Current iteration / step number (1-based).
        scores:     The scores dict from ``TrajectoryDecision.scores``.
        action:     The tracker decision action (``"continue"``, ``"intervene"``, or ``"stop"``).
        agent_name: Optional agent name prefix.
        sub_iter:   Optional sub-iteration for nested loops.
    """
    prefix = f"[{agent_name}] " if agent_name else ""
    if sub_iter is not None:
        iter_label = f"[Iter {iter_num}.{sub_iter}]"
    else:
        iter_label = f"[Iter {iter_num}]"

    def _fmt(val, fmt=".3f"):
        return f"{val:{fmt}}" if val is not None else "\u2014"

    # Line 1: marginal recall (per-step) + signals
    recall = scores.get("marginal_recall")
    n_new = scores.get("num_new_relevant", 0)
    n_total = scores.get("num_docs_this_step", 0)
    if recall is not None:
        recall_str = f"marginal_recall={_fmt(recall)} ({n_new}/{n_total})"
    else:
        recall_str = "marginal_recall=\u2014 (no qrels)"

    novelty = _fmt(scores.get("doc_novelty"))
    consec_sim = _fmt(scores.get("consec_query_sim"))
    orig_sim = _fmt(scores.get("orig_query_sim"))
    signals_line = f"{recall_str} | novelty={novelty} | consec_sim={consec_sim} | orig_sim={orig_sim}"

    print(f"{prefix}{iter_label} [tracker]: {signals_line}")
    padding = " " * len(f"{prefix}{iter_label} [tracker]: ")

    # Aspect coverage
    ac = scores.get("aspect_coverage")
    if ac is not None and isinstance(ac, dict):
        ac_total = ac.get("total", 0)
        if ac_total > 0:
            ac_line = (
                f"aspect_coverage={ac.get('num_covered', 0)}/{ac_total} covered, "
                f"{ac.get('num_partial', 0)}/{ac_total} partial, "
                f"{ac.get('num_not_covered', 0)}/{ac_total} not_covered"
            )
            critical = ac.get("critical_gaps", [])
            minor = ac.get("minor_gaps", [])
            if critical:
                ac_line += f" | critical_gaps=[{', '.join(critical)}]"
            if minor:
                ac_line += f" | minor_gaps=[{', '.join(minor)}]"
            if ac.get("frozen"):
                ac_line += " | FROZEN"
            print(f"{padding}{ac_line}")

    answer_candidates = scores.get("answer_candidates", [])
    if answer_candidates:
        for i, ac in enumerate(answer_candidates):
            candidate_display = ac["candidate"].replace("\n", " ")
            reasoning = ac.get("reasoning", "")
            conf = ac.get("confidence")
            label = f"candidate_answer[{i}]" if len(answer_candidates) > 1 else "candidate_answer"
            parts = [f"{label}={candidate_display}"]
            if conf is not None:
                parts.append(f"confidence={conf}%")
            if reasoning:
                reasoning_display = reasoning.replace("\n", " ")
                if len(reasoning_display) > 150:
                    reasoning_display = reasoning_display[:150] + "..."
                parts.append(f"reasoning={reasoning_display}")
            print(f"{padding}{' | '.join(parts)}")
    else:
        print(f"{padding}candidates=[] | reasoning=—")

    dm_action_val = scores.get("dm_action", action)
    dm_reasoning = scores.get("dm_reasoning", "")
    dm_line = f"dm_action={dm_action_val}"
    if dm_reasoning:
        dm_line += f" | dm_reasoning={dm_reasoning}"
    print(f"{padding}{dm_line}")
