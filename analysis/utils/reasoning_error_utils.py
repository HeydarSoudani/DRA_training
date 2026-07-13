"""Machinery for reasoning-error attribution of TRQA agent trajectories.

This module holds every helper the analysis driver
(:mod:`analysis.reasoning_error_analysis`) needs to turn one query into a
*score* — a per-query root-cause record. The driver only orchestrates the loop
and hands the scores to visualisation; all the logic lives here.

--------------------------------------------------------------------------
Gold: the intermediate-info schema (fixed, ``*_intermediate_info.jsonl``)
--------------------------------------------------------------------------
One JSON object per line::

    {
      "qid": "13_p569",
      "entity_values": [{"entity": "Joseph Stalin", "value": 1878},
                        {"entity": "Nikita Khrushchev", "value": 1894}],
      "property": {"label": "date of birth", "datatype": "Time"},
      "aggregation": "AVG",         # COUNT|AVG|SUM|MAX|MIN|LATEST|EARLIEST
      "answer": 1886
    }

Notes about this data (handled explicitly below):
  * ``answer`` is the ground truth. ``entity_values`` is supporting evidence and
    is occasionally *incomplete* (a recompute over it does not reach ``answer``);
    such rows are flagged ``gold_consistent = false`` and attributed
    conservatively rather than trusting the counterfactuals.
  * ``COUNT`` counts the entities whose property value matches a target *named in
    the question* (e.g. "Europe"). We mark an entity as satisfying iff one of its
    value-items occurs in the question, then pick the polarity
    (positive / negated) that reproduces ``answer``.
  * There are no per-entity gold documents. Per-entity retrieval (``E_ret``) is
    therefore assessed by matching entity names against the *content* of the
    trajectory's retrieved docs (requires a corpus); without a corpus the
    retrieval stage is left unassessed.

--------------------------------------------------------------------------
Method
--------------------------------------------------------------------------
Per trajectory we compute coverage sets over the gold entities E*:

    E_plan   entities whose name appears in the think+query text   (string match)
    E_ret    entities whose name appears in retrieved-doc content  (needs corpus)
    a        the agent's final numeric answer                      (re-extracted)

Because the aggregation ``f`` is known we recompute it over each subset and match
``a``:

    a_star = f(E*)          (equals ``answer`` when gold_consistent)
    a_ret  = f(E_ret)       correct reading/agg over the retrieved set
    a_plan = f(E_plan)      over the planned set

Attribution (earliest-failing stage):

    not gold_consistent                          -> success | gold_incomplete
    a ~= answer                                  -> success
    E_plan subset of E*                          -> planning
    E_ret  subset of E*  and gap is decisive     -> retrieval   (needs corpus)
    E_ret  == E*  (had every entity), a != a*    -> reading / aggregation
    else                                         -> reading / aggregation
"""

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("reasoning_error_analysis")

# ---------------------------------------------------------------------------
# Reused numeric extraction (ported TRQA reference logic).
# ---------------------------------------------------------------------------
try:
    from evaluation.generation.trqa_match import (
        extract_number_from_string,
        _extract_prediction as _extract_answer_text,
    )
except Exception:  # pragma: no cover - keep the module importable standalone
    logger.debug("evaluation.generation.trqa_match unavailable; using local fallbacks")

    _NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)")

    def extract_number_from_string(s):
        if s is None or (isinstance(s, str) and not s.strip()):
            return None
        m = _NUMBER_RE.search(str(s).strip())
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None

    def _extract_answer_text(generation: str) -> str:
        if not generation:
            return ""
        marker = "\n\n## References\n"
        idx = generation.find(marker)
        if idx != -1:
            generation = generation[:idx]
        m = re.search(r"(?i)exact answer[:*\s]+(.+)", generation)
        return (m.group(1) if m else generation).strip()


# Pipeline order: earliest-failing attribution assigns each query to one stage.
# Shared by the stdout table (format_summary_table) and every figure below.
STAGE_ORDER = ["success", "planning", "retrieval", "reading_or_aggregation", "gold_incomplete"]
STAGE_LABELS = {
    "success": "success",
    "planning": "planning",
    "retrieval": "retrieval",
    "reading_or_aggregation": "reading/agg",
    "gold_incomplete": "gold incompl.",
}

# aggregation string -> reducer kind over per-entity contribution values
_REDUCE = {
    "COUNT": "sum",      # sum of 0/1 satisfaction flags
    "SUM": "sum",
    "AVG": "mean",
    "MAX": "max",
    "MIN": "min",
    "LATEST": "max",
    "EARLIEST": "min",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gold(path: str) -> Dict[str, dict]:
    """Load the per-query intermediate-info JSONL keyed by ``qid``."""
    gold: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold[str(rec["qid"])] = rec  # duplicate ids: last wins (rows are identical)
    return gold


def load_trajectory(path: Path) -> Optional[dict]:
    """Parse one ``trajectory/{qid}.jsonl`` (meta record + per-iteration steps)."""
    if not path.exists():
        return None
    meta: dict = {}
    steps: List[dict] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record") == "meta":
                meta = rec
                continue
            steps.append(rec)
            for d in rec.get("seen_docs", []) or []:
                seen.add(str(d))
    return {
        "question": meta.get("question", ""),
        "generation": meta.get("generation", ""),
        "steps": steps,
        "seen_docs": seen,
    }


def load_corpus_subset(path: str, needed: Set[str]) -> Dict[str, str]:
    """Stream the corpus JSONL, keeping ``id -> contents`` only for ``needed`` ids."""
    corpus: Dict[str, str] = {}
    if not needed:
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = str(rec.get("id"))
            if rid in needed:
                corpus[rid] = str(rec.get("contents", ""))
                if len(corpus) == len(needed):
                    break
    return corpus


def load_inputs(args) -> tuple:
    """Load gold, trajectories and (optionally) the retrieved-doc corpus + judge.

    Consumes the resolved ``args`` (see :mod:`analysis.utils.config`) and returns
    ``(gold, trajs, corpus, judge_client, n_missing_gold)`` ready for the
    per-query attribution loop.
    """
    gold = load_gold(args.gold)
    traj_dir = Path(args.run_dir) / "trajectory"
    if not traj_dir.is_dir():
        raise SystemExit(f"no trajectory/ under {args.run_dir}")

    qids = args.qids or sorted(p.stem for p in traj_dir.glob("*.jsonl") if p.stem in gold)
    if args.limit:
        qids = qids[: args.limit]

    # Load trajectories once (needed for the corpus-subset pass and analysis).
    trajs: Dict[str, dict] = {}
    n_missing_gold = 0
    for qid in qids:
        if qid not in gold:
            n_missing_gold += 1
            continue
        t = load_trajectory(traj_dir / f"{qid}.jsonl")
        if t is None:
            logger.warning("no trajectory for %s", qid)
            continue
        trajs[qid] = t

    corpus = None
    if args.corpus:
        needed = set().union(*(t["seen_docs"] for t in trajs.values())) if trajs else set()
        corpus = load_corpus_subset(args.corpus, needed)
        logger.info("loaded %d/%d retrieved docs from corpus", len(corpus), len(needed))
    else:
        logger.warning("no --corpus: per-entity retrieval (E_ret) will be left unassessed")

    judge_client = get_judge_client(args.judge_model) if args.judge else None

    return gold, trajs, corpus, judge_client, n_missing_gold


# ---------------------------------------------------------------------------
# Text / entity matching
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", str(text).lower()).strip()


def _mentions(name: str, normalized_text: str) -> bool:
    """Word-bounded occurrence of ``name`` in already-normalized ``normalized_text``."""
    name = _normalize(name)
    if not name:
        return False
    return re.search(r"\b" + re.escape(name) + r"\b", normalized_text) is not None


def _entity_label(ev: dict) -> str:
    return str(ev.get("entity"))


# ---------------------------------------------------------------------------
# Aggregation:  resolve per-entity contribution values, then reduce over subsets
# ---------------------------------------------------------------------------

def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return extract_number_from_string(value)


def _reduce(kind: str, vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if kind == "sum":
        return float(sum(vals))
    if kind == "mean":
        return float(sum(vals) / len(vals))
    if kind == "max":
        return float(max(vals))
    if kind == "min":
        return float(min(vals))
    raise ValueError(f"unknown reduce kind: {kind}")


def resolve_gold(gold: dict, question: str, gold_tol_abs: float, gold_tol_rel: float) -> dict:
    """Attach per-entity contribution values + consistency flags to a gold record.

    Returns a dict with:
        entities    list of {label, contrib}
        reduce      reducer kind (sum|mean|max|min)
        aggregation original aggregation string
        answer      stored ground-truth answer (float or None)
        a_star      f(E*) recomputed from entity_values
        count_polarity  'positive'|'negated'|None  (COUNT only)
        gold_consistent whether a_star reproduces answer
    """
    agg = str(gold.get("aggregation", "")).upper()
    reduce_kind = _REDUCE.get(agg, "sum")
    answer = _to_number(gold.get("answer"))
    qn = _normalize(question)

    entities: List[dict] = []
    polarity = None

    if agg == "COUNT":
        # satisfaction: any value-item of the entity is named in the question
        def _sat(ev: dict) -> bool:
            v = ev.get("value")
            items = v if isinstance(v, list) else [v]
            return any(_mentions(str(it), qn) for it in items)

        sats = [(_entity_label(ev), _sat(ev)) for ev in gold.get("entity_values", [])]
        n = len(sats)
        pos = sum(1 for _, s in sats if s)
        # choose polarity that reproduces the answer; prefer positive on ties
        if answer is not None and abs(pos - answer) <= 0.5:
            polarity = "positive"
        elif answer is not None and abs((n - pos) - answer) <= 0.5:
            polarity = "negated"
        else:
            polarity = "positive"  # best effort; will be flagged inconsistent
        for label, s in sats:
            contrib = float(s if polarity == "positive" else (not s))
            entities.append({"label": label, "contrib": contrib})
    else:
        for ev in gold.get("entity_values", []):
            entities.append({"label": _entity_label(ev), "contrib": _to_number(ev.get("value"))})

    a_star = _reduce(reduce_kind, [e["contrib"] for e in entities])
    gold_consistent = (
        answer is not None
        and a_star is not None
        and abs(a_star - answer) <= max(gold_tol_abs, gold_tol_rel * abs(answer))
    )

    return {
        "entities": entities,
        "reduce": reduce_kind,
        "aggregation": agg,
        "answer": answer,
        "a_star": a_star,
        "count_polarity": polarity,
        "gold_consistent": bool(gold_consistent),
    }


# ---------------------------------------------------------------------------
# Coverage sets
# ---------------------------------------------------------------------------

def compute_coverage(
    resolved: dict, think_query_text: str, retrieved_text: Optional[str]
) -> Dict[str, Any]:
    """E_plan (planning) and, when corpus content is available, E_ret (retrieval).

    Sets are indexed by entity position (labels can repeat across a query).
    """
    entities = resolved["entities"]
    e_star = set(range(len(entities)))
    norm_plan = _normalize(think_query_text)
    norm_ret = _normalize(retrieved_text) if retrieved_text is not None else None

    e_plan, e_ret = set(), (set() if norm_ret is not None else None)
    for i, e in enumerate(entities):
        if _mentions(e["label"], norm_plan):
            e_plan.add(i)
        if norm_ret is not None and _mentions(e["label"], norm_ret):
            e_ret.add(i)
    return {"E_star": e_star, "E_plan": e_plan, "E_ret": e_ret}


# ---------------------------------------------------------------------------
# Numeric matching
# ---------------------------------------------------------------------------

def _num_match(pred: Optional[float], target: Optional[float], rel_tol: float, abs_tol: float) -> bool:
    if pred is None or target is None:
        return False
    if abs(target) <= abs_tol:
        return abs(pred - target) <= abs_tol
    return abs(pred - target) <= max(abs_tol, rel_tol * abs(target))


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def attribute(
    resolved: dict,
    coverage: Dict[str, Any],
    a_pred: Optional[float],
    rel_tol: float,
    abs_tol: float,
) -> Dict[str, Any]:
    """Assign the earliest-failing stage from coverage sets + counterfactuals."""
    entities = resolved["entities"]
    reduce_kind = resolved["reduce"]
    answer = resolved["answer"]
    a_star = resolved["a_star"]
    E_star, E_plan, E_ret = coverage["E_star"], coverage["E_plan"], coverage["E_ret"]

    def _f(idxs: Set[int]) -> Optional[float]:
        return _reduce(reduce_kind, [entities[i]["contrib"] for i in idxs])

    a_ret = _f(E_ret & E_star) if E_ret is not None else None
    a_plan = _f(E_plan & E_star)

    n = max(1, len(E_star))
    plan_recall = len(E_plan & E_star) / n
    ret_recall = (len(E_ret & E_star) / n) if E_ret is not None else None
    decisive_gap = (
        E_ret is not None
        and (E_ret & E_star) != E_star
        and not _num_match(a_ret, a_star, rel_tol, abs_tol)
    )

    success = _num_match(a_pred, answer, rel_tol, abs_tol)
    match = {
        "answer": success,
        "a_star": _num_match(a_pred, a_star, rel_tol, abs_tol),
        "a_ret": _num_match(a_pred, a_ret, rel_tol, abs_tol),
        "a_plan": _num_match(a_pred, a_plan, rel_tol, abs_tol),
    }

    secondary: List[str] = []
    downstream_subtype = None

    if success:
        primary = "success"
    elif not resolved["gold_consistent"]:
        # entity_values is incomplete -> counterfactuals are unreliable; do not guess
        primary = "gold_incomplete"
    elif plan_recall < 1.0:
        primary = "planning"
        if match["a_plan"]:
            secondary.append("corroborated:a_plan")
    elif decisive_gap:
        primary = "retrieval"
        if match["a_ret"]:
            secondary.append("corroborated:a_ret")
    elif E_ret is None:
        primary = "reading_or_aggregation"
        secondary.append("retrieval_unassessed")  # no corpus supplied
        downstream_subtype = _downstream_subtype(resolved, a_pred, rel_tol, abs_tol)
    else:
        primary = "reading_or_aggregation"
        if (E_ret & E_star) != E_star:
            secondary.append("benign_retrieval_gap")
        downstream_subtype = _downstream_subtype(resolved, a_pred, rel_tol, abs_tol)

    return {
        "primary_stage": primary,
        "secondary": secondary,
        "downstream_subtype": downstream_subtype,
        "gold_consistent": resolved["gold_consistent"],
        "aggregation": resolved["aggregation"],
        "count_polarity": resolved["count_polarity"],
        "a_pred": a_pred,
        "answer": answer,
        "a_star": a_star,
        "a_ret": a_ret,
        "a_plan": a_plan,
        "answer_match": match,
        "plan_recall": round(plan_recall, 4),
        "retrieval_recall": None if ret_recall is None else round(ret_recall, 4),
        "n_gold_entities": len(E_star),
        "n_planned": len(E_plan & E_star),
        "n_retrieved": None if E_ret is None else len(E_ret & E_star),
    }


def _downstream_subtype(resolved: dict, a_pred: Optional[float], rel_tol: float, abs_tol: float) -> str:
    """Zero-LLM split of reading vs aggregation-logic.

    If the agent's answer is reproducible as some *other reducer* over the gold
    contribution values (or the raw entity count / flipped COUNT polarity), the
    values it used were gold-consistent -> aggregation-logic error. Otherwise it
    used a value not in the gold -> likely a reading error.
    """
    contribs = [e["contrib"] for e in resolved["entities"] if e["contrib"] is not None]
    if not contribs:
        return "reading_or_unknown"
    cand = {
        "sum": _reduce("sum", contribs),
        "mean": _reduce("mean", contribs),
        "max": _reduce("max", contribs),
        "min": _reduce("min", contribs),
        "entity_count": float(len(resolved["entities"])),
    }
    if resolved["aggregation"] == "COUNT":
        # flipped polarity == n - sum(contrib)
        cand["flipped_polarity"] = float(len(resolved["entities"]) - _reduce("sum", contribs))
    for name, val in cand.items():
        if _num_match(a_pred, val, rel_tol, abs_tol):
            return f"aggregation_logic:{name}"
    return "reading_or_unknown"


# ---------------------------------------------------------------------------
# Optional LLM judges (gated)
# ---------------------------------------------------------------------------

def get_judge_client(model: str):
    from reasoner_component.api import get_litellm_client

    return get_litellm_client(model_name=model)


def _judge_json(client, prompt: str) -> Optional[dict]:
    try:
        out = client.complete(messages=[{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", out, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception as e:  # pragma: no cover - judges are best-effort
        logger.warning("judge call failed: %s", e)
        return None


def planning_judge(client, gold: dict, question: str, first_think: str, missing: List[str]) -> Optional[dict]:
    prompt = (
        "You audit the PLANNING step of a multi-hop QA agent. Judge only the plan, "
        "not the final answer.\n\n"
        f"Question: {question}\n"
        f"Gold property: {gold.get('property', {}).get('label')}\n"
        f"Gold aggregation: {gold.get('aggregation')}\n"
        f"Gold entities the agent never mentioned: {missing[:40]}\n\n"
        f"Agent's first-step reasoning:\n\"\"\"\n{first_think[:6000]}\n\"\"\"\n\n"
        'Return ONLY JSON: {"type_ok": bool, "property_ok": bool, '
        '"agg_intent_ok": bool, "enumeration_complete": bool, '
        '"abstain": bool, "rationale": str}. Set abstain=true if this step '
        "contains no real plan (planning likely happened later)."
    )
    return _judge_json(client, prompt)


def aggregation_judge(client, gold: dict, generation: str, attr: Dict[str, Any]) -> Optional[dict]:
    prompt = (
        "You audit the final AGGREGATION step of a multi-hop QA agent that had all "
        "the evidence it needed. Classify the error as either a READING error (it "
        "used a fact not supported by the retrieved documents) or an "
        "AGGREGATION-LOGIC error (correct facts, wrong operator/arithmetic).\n\n"
        f"Gold answer: {attr.get('answer')}\n"
        f"Agent answer: {attr.get('a_pred')}\n"
        f"Aggregation: {gold.get('aggregation')} over property "
        f"{gold.get('property', {}).get('label')}\n\n"
        f"Agent's final answer text:\n\"\"\"\n{generation[:6000]}\n\"\"\"\n\n"
        'Return ONLY JSON: {"error_type": "reading" | "aggregation_logic", '
        '"rationale": str}.'
    )
    return _judge_json(client, prompt)


# ---------------------------------------------------------------------------
# Per-query driver: turn one query into a score (root-cause record)
# ---------------------------------------------------------------------------

def analyze_qid(
    qid: str,
    gold: dict,
    traj: dict,
    rel_tol: float,
    abs_tol: float,
    gold_tol_abs: float,
    gold_tol_rel: float,
    corpus: Optional[Dict[str, str]] = None,
    judge_client=None,
) -> Dict[str, Any]:
    """Score a single query: resolve gold, compute coverage, attribute a stage."""
    steps = traj.get("steps", [])
    thinks = " ".join(str(s.get("think", "")) for s in steps)
    queries = " ".join(str(s.get("search_query", "")) for s in steps)
    generation = traj.get("generation", "")
    question = traj.get("question", "")

    resolved = resolve_gold(gold, question, gold_tol_abs, gold_tol_rel)

    retrieved_text = None
    if corpus is not None:
        retrieved_text = " ".join(corpus.get(d, "") for d in traj.get("seen_docs", set()))

    coverage = compute_coverage(resolved, thinks + " " + queries, retrieved_text)
    a_pred = extract_number_from_string(_extract_answer_text(generation))
    attr = attribute(resolved, coverage, a_pred, rel_tol, abs_tol)

    record: Dict[str, Any] = {"qid": qid, **attr, "num_steps": len(steps)}

    if judge_client is not None:
        entities = resolved["entities"]
        if attr["primary_stage"] == "planning" or (
            not attr["answer_match"]["answer"] and attr["plan_recall"] < 1.0
        ):
            missing = [entities[i]["label"] for i in (coverage["E_star"] - coverage["E_plan"])]
            first_think = next((str(s.get("think", "")) for s in steps if s.get("think")), "")
            record["planning_judge"] = planning_judge(judge_client, gold, question, first_think, missing)
        if attr["primary_stage"] == "reading_or_aggregation":
            record["aggregation_judge"] = aggregation_judge(judge_client, gold, generation, attr)

    return record


# ---------------------------------------------------------------------------
# Summary + output
# ---------------------------------------------------------------------------

def _template(qid: str) -> str:
    """The property suffix from the qid (e.g. '13_p569' -> 'p569')."""
    parts = qid.rsplit("_", 1)
    return parts[1] if len(parts) == 2 else "?"


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage_counts = Counter(r["primary_stage"] for r in records)
    n = len(records) or 1

    by_agg: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_agg[r.get("aggregation", "?")][r["primary_stage"]] += 1

    downstream_subtypes = Counter(r["downstream_subtype"] for r in records if r.get("downstream_subtype"))
    ret_recalls = [r["retrieval_recall"] for r in records if r.get("retrieval_recall") is not None]

    return {
        "num_queries": len(records),
        "num_failed": sum(1 for r in records if r["primary_stage"] != "success"),
        "num_gold_inconsistent": sum(1 for r in records if not r.get("gold_consistent")),
        "stage_distribution": dict(stage_counts),
        "stage_distribution_pct": {k: round(v / n, 4) for k, v in stage_counts.items()},
        "downstream_subtypes": dict(downstream_subtypes),
        "mean_plan_recall": round(sum(r["plan_recall"] for r in records) / n, 4),
        "mean_retrieval_recall": (round(sum(ret_recalls) / len(ret_recalls), 4) if ret_recalls else None),
        "by_aggregation": {k: dict(v) for k, v in by_agg.items()},
    }


def format_summary_table(summary: Dict[str, Any]) -> str:
    """Render the summary as a compact text table (for stdout, not a file)."""
    dist = summary.get("stage_distribution", {}) or {}
    n = summary.get("num_queries", 0) or 1
    stages = [s for s in STAGE_ORDER if s in dist]
    stages += [s for s in dist if s not in STAGE_ORDER]  # any unexpected keys

    lines = ["", f"reasoning-error attribution — {summary.get('num_queries', 0)} queries", "-" * 46]
    lines.append(f"{'stage':<24}{'count':>8}{'pct':>10}")
    for s in stages:
        c = dist.get(s, 0)
        lines.append(f"{s:<24}{c:>8}{100.0 * c / n:>9.1f}%")
    lines.append("-" * 46)
    lines.append(f"{'failed':<24}{summary.get('num_failed', 0):>8}")
    lines.append(f"{'gold inconsistent (excl.)':<24}{summary.get('num_gold_inconsistent', 0):>8}")
    mpr = summary.get("mean_plan_recall")
    mrr = summary.get("mean_retrieval_recall")
    if mpr is not None:
        lines.append(f"{'mean plan recall':<24}{mpr:>8.3f}")
    lines.append(f"{'mean retrieval recall':<24}" + ("n/a (no corpus)".rjust(8) if mrr is None else f"{mrr:>8.3f}"))
    lines.append("")
    return "\n".join(lines)


def write_outputs(records: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q = out_dir / "reasoning_error_per_query.jsonl"
    with open(per_q, "w", encoding="utf-8") as f:
        f.write(json.dumps({"record": "meta", **summary}, default=str) + "\n")
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    with open(out_dir / "reasoning_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  wrote {per_q}")
    print(f"  wrote {out_dir / 'reasoning_error_summary.json'}")


def load_records(jsonl_path: Path) -> tuple:
    """Read a ``reasoning_error_per_query.jsonl`` back into ``(records, summary)``.

    The file's first line is the ``record == "meta"`` header (the summary); every
    other line is one per-query record. This is the reader the plotter uses so
    that plotting is a standalone step that depends only on the written file, not
    on the run that produced it.
    """
    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record") == "meta":
                summary = {k: v for k, v in rec.items() if k != "record"}
                continue
            records.append(rec)
    return records, summary


# ===========================================================================
# Visualisation — plain matplotlib figures + tidy CSV, read from the jsonl
# ===========================================================================
# The plot step reads only a written ``reasoning_error_per_query.jsonl`` (via
# ``load_records`` above), so it is decoupled from attribution: you can re-plot
# without re-computing, or point it at any past run's file. Deliberately
# un-styled — matplotlib defaults, no custom palette — for fast exploration, not
# paper polish. Figures written to ``out_dir``:
#     stage_distribution.png     counts per failure stage (the headline)
#     stage_by_aggregation.png   stage mix per aggregation operator
#     stage_by_complexity.png    stage mix per #gold-entities bin (hop count)
#     coverage_scatter.png       plan-recall vs retrieval-recall (needs a corpus)
#     reasoning_error_per_query.csv   tidy per-query rows (for cross-run later)


def _retrieval_assessed(records: List[Dict[str, Any]]) -> bool:
    return any(r.get("retrieval_recall") is not None for r in records)


def _present_stages(counts_by_row: Dict[str, Counter]) -> List[str]:
    """Stages that actually occur, in pipeline order."""
    return [s for s in STAGE_ORDER if any(c.get(s) for c in counts_by_row.values())]


def plot_stage_distribution(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """Headline figure — counts per failure stage."""
    import matplotlib.pyplot as plt

    counts = Counter(r.get("primary_stage", "gold_incomplete") for r in records)
    stages = [s for s in STAGE_ORDER if counts.get(s)]
    if not stages:
        return None
    vals = [counts[s] for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([STAGE_LABELS[s] for s in stages], vals)
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("# queries")
    ax.set_title(f"Failure stage distribution (n={len(records)})")
    fig.tight_layout()
    p = out / "stage_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _plot_stacked(counts_by_row: Dict[str, Counter], rows: List[str],
                  title: str, xlabel: str, path: Path) -> Optional[Path]:
    """Plain stacked bars (raw counts): one bar per row, stacked by stage."""
    import matplotlib.pyplot as plt

    stages = _present_stages(counts_by_row)
    if not rows or not stages:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = [0] * len(rows)
    for s in stages:
        vals = [counts_by_row[r].get(s, 0) for r in rows]
        ax.bar(rows, vals, bottom=bottom, label=STAGE_LABELS[s])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("# queries")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _bin_entities(n: Optional[int]) -> str:
    if not n:
        return "?"
    return "5+" if n >= 5 else str(n)


def plot_stage_by_aggregation(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    by_agg: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_agg[str(r.get("aggregation", "?"))][r.get("primary_stage", "gold_incomplete")] += 1
    rows = sorted(by_agg, key=lambda a: -sum(by_agg[a].values()))
    return _plot_stacked(by_agg, rows, "Stage mix by aggregation operator",
                         "aggregation", out / "stage_by_aggregation.png")


def plot_stage_by_complexity(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        buckets[_bin_entities(r.get("n_gold_entities"))][r.get("primary_stage", "gold_incomplete")] += 1
    rows = [b for b in ["2", "3", "4", "5+", "?"] if b in buckets]
    return _plot_stacked(buckets, rows, "Stage mix by # gold entities (hop count)",
                         "# gold entities", out / "stage_by_complexity.png")


def plot_coverage_scatter(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """plan-recall vs retrieval-recall scatter (only when a corpus was supplied)."""
    if not _retrieval_assessed(records):
        return None
    import matplotlib.pyplot as plt

    by_stage: Dict[str, list] = defaultdict(list)
    for r in records:
        x, y = r.get("plan_recall"), r.get("retrieval_recall")
        if x is None or y is None:
            continue
        by_stage[r.get("primary_stage", "gold_incomplete")].append((x, y))
    if not by_stage:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for s in STAGE_ORDER:
        pts = by_stage.get(s)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=30, alpha=0.6, label=STAGE_LABELS[s])
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("plan recall (entities enumerated / gold)")
    ax.set_ylabel("retrieval recall (entities surfaced / gold)")
    ax.set_title("Where is entity coverage lost?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "coverage_scatter.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


_CSV_FIELDS = [
    "qid", "primary_stage", "aggregation", "downstream_subtype", "gold_consistent",
    "n_gold_entities", "n_planned", "n_retrieved", "plan_recall", "retrieval_recall",
    "num_steps", "a_pred", "answer",
]


def export_csv(records: List[Dict[str, Any]], out: Path,
               run_key: Optional[Dict[str, str]] = None) -> Path:
    """Tidy per-query rows for cross-run stacking (optionally keyed by the run)."""
    run_key = run_key or {}
    key_fields = list(run_key)
    p = out / "reasoning_error_per_query.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(key_fields + _CSV_FIELDS)
        for r in records:
            w.writerow([run_key[k] for k in key_fields] + [r.get(k) for k in _CSV_FIELDS])
    return p


def render_all(jsonl_path: Path, out_dir: Path,
               run_key: Optional[Dict[str, str]] = None) -> None:
    """Read ``reasoning_error_per_query.jsonl`` and render CSV + plain PNGs.

    The file is the only input, so plotting is decoupled from attribution.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, _summary = load_records(Path(jsonl_path))
    if not records:
        logger.warning("[viz] no records in %s", jsonl_path)
        return

    # CSV first — it never depends on matplotlib being importable.
    written = [export_csv(records, out_dir, run_key)]

    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception as e:  # pragma: no cover
        logger.warning("[viz] matplotlib unavailable (%s) — wrote CSV only", e)
        for p in written:
            logger.info("[viz] wrote %s", p)
        return

    for fn in (plot_stage_distribution,
               plot_stage_by_aggregation,
               plot_stage_by_complexity,
               plot_coverage_scatter):
        try:
            p = fn(records, out_dir)
            if p:
                written.append(p)
        except Exception as e:  # one bad figure must not sink the run
            logger.warning("[viz] figure %s failed: %s", fn.__name__, e)

    for p in written:
        logger.info("[viz] wrote %s", p)
