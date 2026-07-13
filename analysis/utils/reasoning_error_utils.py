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


STAGES = ("success", "planning", "retrieval", "reading_or_aggregation", "gold_incomplete")

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


def load_qrels(path: str) -> Dict[str, Set[str]]:
    """Load TREC qrels (``qid 0 docid rel``) -> {qid: {docid}} for rel>0."""
    qrels: Dict[str, Set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels[str(qid)].add(str(docid))
    return dict(qrels)


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
    retrieval_subtype = None
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
        "retrieval_subtype": retrieval_subtype,
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


# ---------------------------------------------------------------------------
# Self-test  (validates resolve/attribution on the fixed intermediate schema)
# ---------------------------------------------------------------------------

def self_test() -> None:
    rel, ab = 1e-3, 1e-6
    gtab, gtrel = 0.5, 1e-2

    def analyze(gold, question, think_text, retrieved_text, a_pred):
        resolved = resolve_gold(gold, question, gtab, gtrel)
        cov = compute_coverage(resolved, think_text, retrieved_text)
        return resolved, attribute(resolved, cov, a_pred, rel, ab)

    # --- COUNT: "how many ... located in Europe" over 4 states, answer 2 ---
    count_gold = {
        "qid": "q_count", "aggregation": "COUNT", "answer": 2,
        "property": {"label": "continent"},
        "entity_values": [
            {"entity": "Alpha", "value": ["Europe"]},
            {"entity": "Beta", "value": ["Asia"]},
            {"entity": "Gamma", "value": ["Europe"]},
            {"entity": "Delta", "value": ["North America"]},
        ],
    }
    q = "How many states are located in Europe?"
    allt = "Alpha Beta Gamma Delta"

    res, r = analyze(count_gold, q, allt, allt, 2)  # success
    assert res["count_polarity"] == "positive" and res["gold_consistent"], res
    assert r["primary_stage"] == "success", r

    _, r = analyze(count_gold, q, "Alpha Beta Delta", allt, 1)  # never planned Gamma (Europe)
    assert r["primary_stage"] == "planning", r
    assert r["answer_match"]["a_plan"], r

    _, r = analyze(count_gold, q, allt, "Alpha Beta Delta", 1)  # Gamma planned, not retrieved
    assert r["primary_stage"] == "retrieval", r

    _, r = analyze(count_gold, q, allt, allt, 4)  # had all, returned entity count
    assert r["primary_stage"] == "reading_or_aggregation", r
    assert r["downstream_subtype"].startswith("aggregation_logic"), r

    _, r = analyze(count_gold, q, allt, None, 4)  # no corpus -> retrieval unassessed
    assert r["primary_stage"] == "reading_or_aggregation", r
    assert "retrieval_unassessed" in r["secondary"], r

    # --- AVG over years, answer 1886 ---
    avg_gold = {
        "qid": "q_avg", "aggregation": "AVG", "answer": 1886,
        "property": {"label": "date of birth"},
        "entity_values": [{"entity": "Stalin", "value": 1878}, {"entity": "Khrushchev", "value": 1894}],
    }
    res, r = analyze(avg_gold, "avg birth year", "Stalin Khrushchev", "Stalin Khrushchev", 1886)
    assert res["gold_consistent"] and r["primary_stage"] == "success", (res, r)

    # --- gold_incomplete: entity_values cannot reach the answer ---
    bad_gold = {
        "qid": "q_bad", "aggregation": "SUM", "answer": 40501688,
        "property": {"label": "population"},
        "entity_values": [{"entity": "X", "value": 17494398}, {"entity": "Y", "value": 18676605}],
    }
    res, r = analyze(bad_gold, "sum pop", "X Y", "X Y", 999)
    assert not res["gold_consistent"], res
    assert r["primary_stage"] == "gold_incomplete", r

    print("self-test: all assertions passed")
