"""Trajectory evaluator for deep-research agents.

Accepts the **unified** result format produced by all agents:

    {
        query_id: {
            "trajectory": List[Dict]  – per-step trace
            "num_steps":  int
            "num_searches": int
        }
    }

Each trajectory step is a dict whose schema varies by agent type:

    ReAct / ReAct-WoPlan (action_type field):
        {"action_type": "search", "search_query": str, "docs": List[Dict], "think": str}
        {"action_type": "finish", "conclusion": str, "think": str}

    SearchR1 / ReSearch / StepSearch / SelfAsk (inferred from keys):
        {"think": str, "search_query": str, "docs": List[Dict]}
        {"think": str, "prediction": str}

    AgentCPM (action field):
        {"step": int, "state": str, "action": str,
         "input": {...}, "output": {...}}

The evaluator normalises all these formats into a common ``action_type``
string and then computes aggregate statistics.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_action_type(step: Dict[str, Any]) -> str:
    """Return a normalised action type string for any agent step format."""
    # ReAct family — explicit action_type
    if "action_type" in step:
        return str(step["action_type"]).lower()

    # AgentCPM — explicit action field
    if "action" in step:
        return str(step["action"]).lower()

    # Infer from keys (SearchR1, ReSearch, StepSearch, SelfAsk…)
    if step.get("search_query") or step.get("query"):
        return "search"
    if "prediction" in step:
        return "predict"
    if "conclusion" in step:
        return "finish"
    if "follow_up" in step:
        return "followup"

    return "unknown"


def _get_num_docs(step: Dict[str, Any]) -> int:
    """Return the number of documents retrieved in a step (0 if not a search)."""
    # Flat doc list (most agents)
    docs = step.get("docs")
    if docs:
        return len(docs)

    # all_docs: can be flat list of dicts (OSS/GLM/Tongyi) or list-of-lists
    # (CPM/ReactWithPlan).  Detect format by checking the first element.
    all_docs = step.get("all_docs")
    if all_docs:
        if isinstance(all_docs[0], dict):
            # Flat list of doc dicts
            return len(all_docs)
        else:
            # List of lists — sum inner lengths
            return sum(len(d) for d in all_docs if d)

    # AgentCPM: output.num_results
    output = step.get("output", {})
    if isinstance(output, dict) and "num_results" in output:
        return int(output["num_results"])

    return 0


# Heavy/stable per-doc fields that are identical every time a doc is surfaced.
# These are hoisted into a single per-file ``documents`` pool (keyed by doc_id)
# so the text is stored once instead of once per step that re-surfaces the doc.
_DOC_POOL_FIELDS = ("title", "contents", "text", "metadata")
# Text fields dropped entirely when ``save_doc_text`` is False.
_DOC_TEXT_FIELDS = ("contents", "text")


def _clean_doc_for_save(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Strip redundant fields from a single retrieved-doc dict.

    Removes:
    - ``_emb`` — embedding vectors (huge, never needed on disk).
    - ``relevant_text`` — a verbatim copy of ``contents`` (every reader falls
      back to ``contents``, so this is pure duplication).
    - ``id`` when ``doc_id`` is also present.
    - ``metadata.title`` when it duplicates the top-level ``title``.
    """
    cleaned = {
        k: v for k, v in doc.items()
        if k != "_emb"
        and k != "relevant_text"
        and not (k == "id" and "doc_id" in doc)
    }
    # Drop metadata entirely if its only content duplicates the top-level title.
    meta = cleaned.get("metadata")
    if isinstance(meta, dict):
        meta = {k: v for k, v in meta.items()
                if not (k == "title" and v == cleaned.get("title"))}
        if meta:
            cleaned["metadata"] = meta
        else:
            cleaned.pop("metadata", None)
    return cleaned


def _clean_trajectory_for_save(trajectory: list) -> list:
    """Clean trajectory steps before saving to disk.

    Per-step transformations:
    - Remove ``all_docs`` (redundant with ``docs``).
    - Strip ``_emb``/``relevant_text``/duplicate ``id``/duplicate
      ``metadata.title`` from docs (see :func:`_clean_doc_for_save`).
    """
    cleaned_steps = []
    for step in trajectory:
        step = dict(step)  # shallow copy

        # Remove all_docs — docs is sufficient
        step.pop("all_docs", None)

        if "docs" in step and step["docs"]:
            step["docs"] = [_clean_doc_for_save(doc) for doc in step["docs"]]

        cleaned_steps.append(step)
    return cleaned_steps


def _extract_doc_pool(trajectory: list, save_doc_text: bool) -> tuple:
    """Hoist heavy/stable doc fields into a shared per-file ``documents`` pool.

    Each step's ``docs`` are reduced to lightweight references that keep only
    per-step fields (``doc_id``, rank/score, controller fields, …).  The stable
    fields (``title``, ``contents``, ``text``, ``metadata``) are stored once in
    the returned pool keyed by ``doc_id``.  A doc retrieved across N steps thus
    stores its text once instead of N times.

    When *save_doc_text* is False the text fields (``contents``/``text``) are
    omitted from the pool entirely — trajectory files then carry only doc-id
    references plus titles, recoverable from the corpus at analysis time.

    Docs without a resolvable ``doc_id`` are left inline (cannot be pooled).

    Returns:
        ``(trajectory_with_refs, documents_pool)``.
    """
    pool: Dict[str, Dict[str, Any]] = {}
    out_steps = []
    for step in trajectory:
        step = dict(step)
        docs = step.get("docs")
        if docs:
            ref_docs = []
            for doc in docs:
                did = doc.get("doc_id") or doc.get("id")
                if not did:
                    ref_docs.append(doc)  # cannot pool without an id
                    continue
                # Collect the stable fields into the pool (once per doc_id).
                if did not in pool:
                    entry = {}
                    for f in _DOC_POOL_FIELDS:
                        if f in _DOC_TEXT_FIELDS and not save_doc_text:
                            continue
                        if f in doc:
                            entry[f] = doc[f]
                    pool[did] = entry
                # Reference keeps only the per-step fields.
                ref_docs.append({
                    k: v for k, v in doc.items() if k not in _DOC_POOL_FIELDS
                })
            step["docs"] = ref_docs
        out_steps.append(step)
    return out_steps, pool


def _fold_controller_into_steps(
    trajectory: list, score_history: Optional[list],
) -> list:
    """Attach controller decisions to their matching search step.

    The controller produces one score-history entry per search/post-search
    evaluation.  These entries are matched positionally to the trajectory's
    search steps (1:1 in order), so each search step gains the controller's
    ``controller_action`` (one of continue/intervene/stop) and
    ``controller_reasoning`` — the same field names the controller uses.
    Non-search steps and unmatched entries are left untouched.  The full
    per-iteration signal record remains available separately under
    ``controller_score_history``.
    """
    if not score_history:
        return trajectory

    by_iter: Dict[Any, Dict[str, Any]] = {}
    for idx, sc in enumerate(score_history):
        by_iter[sc.get("iter_num", idx)] = sc

    search_idx = 0
    out = []
    for step in trajectory:
        is_search = bool(
            step.get("docs") or step.get("all_docs")
            or step.get("component_doc_ids") or step.get("search_query")
        )
        if is_search:
            sc = by_iter.get(search_idx)
            if sc is not None and "controller_action" not in step:
                folded = {}
                if sc.get("controller_action") is not None:
                    folded["controller_action"] = sc["controller_action"]
                if sc.get("controller_reasoning") is not None:
                    folded["controller_reasoning"] = sc["controller_reasoning"]
                if folded:
                    step = {**step, **folded}
            search_idx += 1
        out.append(step)
    return out


def _is_search_step(action_type: str, step: Dict[str, Any]) -> bool:
    """Return True if this step involved a retrieval call."""
    if action_type == "search":
        return True
    # Some agents may not set action_type but do have docs
    if step.get("docs") or step.get("all_docs"):
        return True
    return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TrajectoryEvaluator:
    """Evaluate agent trajectories from the unified result format.

    Usage::

        evaluator = TrajectoryEvaluator()
        metrics = evaluator.evaluate(results)
        evaluator.print_results(metrics)
    """

    def __init__(self, save_doc_text: bool = True, dedup_docs: bool = True) -> None:
        """
        Args:
            save_doc_text: When True (default) the full document text is written
                to the per-query trajectory file.  When False only doc-id
                references + titles are kept, producing much smaller files
                (text is recoverable from the corpus at analysis time).
            dedup_docs:    When True (default) stable per-doc fields are hoisted
                into a single per-file ``documents`` pool so a doc surfaced in
                multiple steps stores its text once.  Set False to keep the
                legacy inline-per-step format.
        """
        self.save_doc_text = save_doc_text
        self.dedup_docs = dedup_docs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute trajectory statistics aggregated over all queries.

        Always returns:
        - ``num_queries``
        - ``steps`` – mean/std/min/max for total steps per query
        - ``search_steps`` – mean/std/min/max for search steps per query
        - ``docs_per_search`` – mean/std/min/max docs retrieved per search step
        - ``action_type_counts`` – avg count of each action type per query
        - ``queries_no_search`` – number of queries with zero search steps
        - ``queries_max_iter`` – number of queries whose trajectory ended without
          a finish/predict action (possible truncation)

        Args:
            results: Unified agent results keyed by *query_id*.

        Returns:
            Nested metrics dict.
        """
        if not results:
            return {}

        _TERMINAL_ACTIONS = {"finish", "predict", "done", "answer", "terminate", "early_stop"}
        _FORCE_ANSWER_ACTIONS = {"context_limit", "max_iter_force"}

        total_steps_per_query: List[int] = []
        search_steps_per_query: List[int] = []
        docs_per_search_all: List[int] = []
        action_type_counts_per_query: List[Dict[str, int]] = []
        queries_no_search = 0
        queries_with_force_answer = 0
        queries_max_iter = 0

        for query_id, result in results.items():
            trajectory = result.get("trajectory", [])

            if not trajectory:
                n_steps = result.get("num_steps", 0)
                n_searches = result.get("num_searches", 0)
                total_steps_per_query.append(n_steps)
                search_steps_per_query.append(n_searches)
                if n_searches == 0:
                    queries_no_search += 1
                action_type_counts_per_query.append({})
                continue

            step_action_counts: Dict[str, int] = defaultdict(int)
            n_search = 0
            has_terminal = False
            has_force_answer = False

            for step in trajectory:
                atype = _infer_action_type(step)
                step_action_counts[atype] += 1

                if _is_search_step(atype, step):
                    n_search += 1
                    n_docs = _get_num_docs(step)
                    docs_per_search_all.append(n_docs)

                if atype in _TERMINAL_ACTIONS:
                    has_terminal = True
                if atype in _FORCE_ANSWER_ACTIONS:
                    has_force_answer = True

            total_steps_per_query.append(len(trajectory))
            search_steps_per_query.append(n_search)
            action_type_counts_per_query.append(dict(step_action_counts))

            if n_search == 0:
                queries_no_search += 1
            if has_force_answer or not has_terminal:
                queries_with_force_answer += 1
            if not has_terminal:
                queries_max_iter += 1

        def _stats(lst: List) -> Dict[str, float]:
            if not lst:
                return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
            arr = np.array(lst, dtype=float)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
            }

        # Aggregate action type counts (mean per query)
        all_action_types = set(
            atype
            for counts in action_type_counts_per_query
            for atype in counts
        )
        avg_action_type_counts: Dict[str, float] = {}
        n_queries = len(results)
        for atype in sorted(all_action_types):
            total = sum(c.get(atype, 0) for c in action_type_counts_per_query)
            avg_action_type_counts[atype] = round(total / n_queries, 3)

        metrics: Dict[str, Any] = {
            "num_queries": n_queries,
            "steps": _stats(total_steps_per_query),
            "search_steps": _stats(search_steps_per_query),
            "docs_per_search": _stats(docs_per_search_all),
            "avg_action_type_counts": avg_action_type_counts,
            "queries_no_search": queries_no_search,
            "queries_with_force_answer": queries_with_force_answer,
            "queries_max_iter_reached": queries_max_iter,
        }

        # Token usage (per-query totals attached by the agents as result["token_usage"])
        token_usages = [
            result["token_usage"]
            for result in results.values()
            if result.get("token_usage")
        ]
        if token_usages:
            def _tok(key: str) -> List[float]:
                return [float(tu.get(key, 0) or 0) for tu in token_usages]
            metrics["tokens"] = {
                "num_queries_with_tokens": len(token_usages),
                "input_tokens": _stats(_tok("input_tokens")),
                "output_tokens": _stats(_tok("output_tokens")),
                "total_tokens": _stats(_tok("total_tokens")),
                "llm_calls": _stats(_tok("num_calls")),
            }

        return metrics

    def print_results(self, metrics: Dict[str, Any], header: str = "TRAJECTORY STATISTICS") -> None:
        """Pretty-print trajectory statistics.

        Args:
            metrics: Output of :meth:`evaluate`.
            header:  Section header string.
        """
        if not metrics:
            print("  ⚠ No trajectory metrics available")
            return

        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)

        n = metrics.get("num_queries", 0)
        print(f"  Queries evaluated:        {n}")

        def _fmt_stats(d: Optional[Dict]) -> str:
            if not d:
                return "n/a"
            return (
                f"{d['mean']:.1f} ± {d['std']:.1f}"
                f"  (min: {d['min']}, max: {d['max']})"
            )

        print(f"  Steps per query:          {_fmt_stats(metrics.get('steps'))}")
        print(f"  Search steps per query:   {_fmt_stats(metrics.get('search_steps'))}")
        print(f"  Docs per search step:     {_fmt_stats(metrics.get('docs_per_search'))}")

        no_search = metrics.get("queries_no_search", 0)
        pct = (no_search / n * 100) if n else 0
        print(f"  Queries with no search:   {no_search} ({pct:.1f}%)")

        force_ans = metrics.get("queries_with_force_answer", 0)
        pct_force = (force_ans / n * 100) if n else 0
        print(f"  Queries w/ force answer:  {force_ans} ({pct_force:.1f}%)")

        max_iter = metrics.get("queries_max_iter_reached", 0)
        pct_max = (max_iter / n * 100) if n else 0
        print(f"  Queries hitting max iter: {max_iter} ({pct_max:.1f}%)")

        tokens = metrics.get("tokens")
        if tokens:
            tot = tokens.get("total_tokens", {})
            inp = tokens.get("input_tokens", {})
            out = tokens.get("output_tokens", {})
            calls = tokens.get("llm_calls", {})
            print(f"  Tokens/query (total):     {tot.get('mean', 0):.0f} ± {tot.get('std', 0):.0f}"
                  f"  (in: {inp.get('mean', 0):.0f}, out: {out.get('mean', 0):.0f})")
            print(f"  LLM calls per query:      {calls.get('mean', 0):.1f}")

        action_counts = metrics.get("avg_action_type_counts", {})
        if action_counts:
            print(f"  Action type distribution (avg per query):")
            for atype, avg in sorted(action_counts.items(), key=lambda x: -x[1]):
                print(f"    {atype:<20s}: {avg:.2f}")

        print("=" * 80)

    def save_item(self, query_id: str, question: str, result: Dict[str, Any], output_dir) -> None:
        """Save per-query trajectory as a JSON file.

        The file contains the full trajectory along with metadata needed to
        reconstruct the result for resumed runs (generation, step counts,
        citation mapping, etc.).

        Supports both local paths and S3 URIs.

        Args:
            query_id:   Query identifier.
            question:   Original query text.
            result:     Unified agent result dict for this query.
            output_dir: Directory where ``{query_id}.json`` will be written.
        """
        output_dir_str = str(output_dir)
        Path(output_dir_str).mkdir(parents=True, exist_ok=True)
        item: Dict[str, Any] = {
            "qid": query_id,
            "question": question,
            "trajectory": result.get("trajectory", []),
            "generation": result.get("generation", ""),
            "num_steps": result.get("num_steps", 0),
            "num_searches": result.get("num_searches", 0),
            "num_iterations": result.get("num_iterations"),
        }
        # Preserve agent-specific metadata when present.
        #
        # Note: the full ``controller_score_history`` (and its ``*_unique_doc_*``
        # companions) is deliberately NOT duplicated here — it is persisted once
        # by ``ControllerEvaluator`` under ``controller/{qid}.json`` and the
        # controller *decisions* are folded into the search steps below.  The
        # eval loader reconstructs the history from the controller file on
        # resume.
        for optional_key in (
            "citation_to_doc_id",
            "cited_docs_ranked_list",
            "memory_bank",
            "query_outputs",       # GTR baseline metadata
            "token_usage",
        ):
            if optional_key in result and result[optional_key]:
                item[optional_key] = result[optional_key]

        # Fold controller decisions into their matching search steps so the
        # trajectory log carries reasoning + query + doc ids + controller values
        # + tokens in one place.
        item["trajectory"] = _fold_controller_into_steps(
            item["trajectory"],
            result.get("controller_score_history") or result.get("tracker_score_history"),
        )

        # Clean trajectory: remove all_docs, strip _emb/relevant_text/duplicate id.
        item["trajectory"] = _clean_trajectory_for_save(item["trajectory"])

        # Hoist stable doc fields into a shared pool (dedup across steps) and,
        # when configured, drop the document text entirely.
        if self.dedup_docs:
            item["trajectory"], documents = _extract_doc_pool(
                item["trajectory"], self.save_doc_text,
            )
            if documents:
                item["documents"] = documents
        elif not self.save_doc_text:
            # No pooling, but still honour text suppression by stripping inline.
            for step in item["trajectory"]:
                for doc in step.get("docs", []) or []:
                    for f in _DOC_TEXT_FIELDS:
                        doc.pop(f, None)

        json_path = f"{output_dir_str.rstrip('/')}/{query_id}.json"
        with open(json_path, "w") as f:
            json.dump(item, f, separators=(",", ":"), default=str)

    def save_results(self, metrics: Dict[str, Any], output_path, summary: Optional[Dict[str, Any]] = None, summary_path=None) -> None:
        """Save trajectory statistics and optionally the run summary to JSON files.

        Args:
            metrics:      Output of :meth:`evaluate`.
            output_path:  Destination file for trajectory metrics (parents created if needed).
            summary:      Optional full-run summary dict to persist alongside metrics.
            summary_path: Destination file for the summary (required when *summary* is given).
        """
        if not metrics:
            return
        output_path_str = str(output_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  ✓ Saved trajectory metrics: {output_path_str}")

        if summary and summary_path:
            summary_path_str = str(summary_path)
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  ✓ Saved summary: {summary_path_str}")
