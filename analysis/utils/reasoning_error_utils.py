"""Reasoning-error analysis for TRQA agent trajectories.

The whole method is a **two-sided comparison**. For each query we take the gold
decomposition and ask, for every gold entity, whether its counterpart shows up in
the agent's trajectory.

Gold side (per query, from ``*_intermediate_info.jsonl`` + qrels)::

    E*  entities            [{"entity": "Joseph Stalin", "value": 1878}, ...]
    V*  their property values
    f   the aggregation operator ("AVG")
    a*  the stored answer (1886)
    D*  gold documents, split per entity via the qrels page ids

Trajectory side. A trajectory is ``{(t, q, o)}_{i=1..M}`` — think, search query,
observation. We collapse it into exactly two channels::

    A = concat of all (t, q)     what the agent said
    O = union of all seen doc ids  what retrieval delivered

and compute, per gold entity, three independent booleans plus one query-level
arithmetic check::

    e_in_A     entity name occurs in A                    -> planning
    retrieved  D*_e intersects O                          -> retrieval
    v_in_A     the gold value occurs in A                 -> reading
    agg_ok     f(values found in A) matches the answer    -> aggregation

There is deliberately **no cascade** and no earliest-failing rule: the four
columns are marginals, not stages, so the interesting joint cases fall out of the
signature instead of being encoded as rules (e.g. ``retrieved and not v_in_A`` =
surfaced but never articulated; ``all v_in_A and not agg_ok`` = had the facts,
fumbled the arithmetic).

``gold_ok`` (does ``f(V*)`` reproduce ``a*``?) is a *filter*, not a stage: the
intermediate-info gold is occasionally incomplete and those rows cannot support
the aggregation check.

The same trajectory is scored twice, at two granularities::

    entity rows  micro — every gold entity of every query, pooled
    query rows   macro — one score per query (plan / retrieval / reading
                 coverage), so the *distribution over queries* is plottable

They do not agree: a 50-entity query contributes 50 entity rows but a single
query row. That divergence is a result, not a bug, so :func:`summarize` reports
both side by side.

Note the one dependence between the columns: ``v_in_A`` is searched only within
``VALUE_WINDOW`` of a mention of its own entity, so ``v_in_A`` implies ``e_in_A``
and the raw reading coverage is a *product* of planning and reading failure. The
per-query ``read_cov_given_named`` conditions that away.

Layout: matching -> aggregators -> loading -> analysis -> output.
"""

import importlib.util
import json
import logging
import re
import statistics
import unicodedata
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _load_module(name: str, path: Path):
    """Import a single module file without executing its parent package's ``__init__``."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reused verbatim from the accuracy evaluator so `agg_ok` and the reported
# accuracy share one notion of "this number matches that number".
#
# Loaded straight from the file rather than via ``from evaluation.generation
# .trqa_match import ...``: that import runs ``evaluation/__init__``, which pulls
# in the whole retriever stack (torch, pyserini/JVM) and costs ~65s of startup
# this analysis has no use for. trqa_match itself has no heavy dependencies.
_trqa_match = _load_module(
    "trqa_match", Path(__file__).resolve().parents[2] / "src" / "evaluation" / "generation" / "trqa_match.py"
)
_NUMBER_RE = _trqa_match._NUMBER_RE
_extract_prediction = _trqa_match._extract_prediction
soft_exact_match = _trqa_match.soft_exact_match

logger = logging.getLogger("reasoning_error_analysis")


# ===========================================================================
# 1. Matching — does this gold item occur in the agent's text?
# ===========================================================================
# Containment, not comparison. The evaluator's matcher answers "do these two
# scalars agree"; here we search one known value in a long haystack, so the
# first-number extraction and the wide tolerance ladder it uses are both wrong.

_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Inline citations the agent writes into its think text — "[15641-0002]", "[6]".
# Stripped from A before matching: normalisation would otherwise turn the doc id
# into the bare tokens "15641" and "0002", and nine wiki2 qrels page ids are
# year-shaped (1930, 1966-1971, ...), so a citation can spoof a gold `Time` value
# and every passage index pollutes the `Quantity` candidate set.
_CITATION_RE = re.compile(r"\[\d+(?:-\d+)?\]")
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014"), "-")

# How far from a mention of the entity a value still counts as "stated for it".
# Values are searched only inside these windows: in a 50-entity question about
# mean age every value sits in the same narrow band, so a global search matches
# whichever entity you ask about (measured: 10 false positives out of 109 pairs).
VALUE_WINDOW = 400


def norm_text(s: Any) -> str:
    """Casefold to a punctuation-free token string — for text containment tests.

    NFKC first, because agent generations carry narrow no-break spaces and
    non-breaking hyphens (``Leonid Brezhnev``, ``1940\u20111953``) that
    defeat a naive substring match. Everything non-word then becomes a space, so
    matching is on token boundaries without needing a regex per lookup.
    """
    s = unicodedata.normalize("NFKC", str(s))
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", s)).casefold().strip()


def norm_soft(s: Any) -> str:
    """Casefold but *keep* punctuation — for locating mentions and reading numbers.

    :func:`norm_text` would turn ``-76.79`` into ``76 79``, destroying the sign and
    the decimal point, so numeric work runs on this string instead.
    """
    s = unicodedata.normalize("NFKC", str(s)).translate(_DASHES)
    return _WS_RE.sub(" ", s).casefold()


def pad(s: str) -> str:
    """Wrap a normalised string in spaces so ``in`` tests hit token boundaries."""
    return f" {s} "


def numbers_in(text: str) -> List[float]:
    """Every number in *text*, sorted — the candidate set for numeric matching."""
    out = []
    for m in _NUMBER_RE.finditer(text):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    out.sort()
    return out


def _near(nums: Sequence[float], target: float, tol_rel: float) -> bool:
    """Is some number in the sorted *nums* within ``tol_rel`` of *target*?"""
    if not nums:
        return False
    tol = max(abs(target) * tol_rel, 1e-9)
    i = bisect_left(nums, target)
    for j in (i - 1, i):
        if 0 <= j < len(nums) and abs(nums[j] - target) <= tol:
            return True
    return False


def name_variants(name: str, siblings: Sequence[str] = ()) -> List[str]:
    """Normalised forms of *name* worth searching for.

    Adds the last token ("Stalin" for "Joseph Stalin") because agents routinely
    drop the given name — but only when that token is distinctive and unambiguous
    among *siblings* (the query's other gold entities), so two gold entities
    sharing a surname never collapse into one.
    """
    n = norm_text(name)
    if not n:
        return []
    variants = [n]
    toks = n.split()
    if len(toks) >= 2 and len(toks[-1]) > 3:
        last = toks[-1]
        clashes = sum(
            1 for s in siblings if (t := norm_text(s).split()) and t[-1] == last
        )
        if clashes <= 1:
            variants.append(last)
    return variants


def entity_in(name: str, hay: str, siblings: Sequence[str] = ()) -> bool:
    """Does *name* occur anywhere in the padded, normalised haystack *hay*?"""
    return any(pad(v) in hay for v in name_variants(name, siblings))


def local_context(soft: str, variants: Sequence[str], window: int = VALUE_WINDOW) -> str:
    r"""The parts of *soft* within *window* characters of any mention of *variants*.

    Tokens are joined with ``\W+`` so a normalised variant ("st petersburg") still
    finds its punctuated form in the soft text ("st. petersburg").
    """
    spans = []
    for v in variants:
        pattern = r"(?<!\w)" + r"\W+".join(re.escape(t) for t in v.split()) + r"(?!\w)"
        spans += [m.span() for m in re.finditer(pattern, soft)]
    if not spans:
        return ""
    # Merge the windows before slicing: mentions of the same entity cluster, so
    # naive per-span slicing emits the same sentences several times over (measured
    # 4x on a 12k-char trajectory) and makes the cost quadratic in mention count.
    windows = sorted((max(0, a - window), min(len(soft), b + window)) for a, b in spans)
    merged = [list(windows[0])]
    for a, b in windows[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return " ".join(soft[a:b] for a, b in merged)


def value_in(value: Any, ctx: str, datatype: str, tol_rel: float) -> bool:
    """Does the gold *value* occur in *ctx* (the entity's local context)?

    Dispatched on the gold ``property.datatype``, because the value shapes differ
    completely: ``WikibaseItem`` values are label lists (``["Eastern Orthodoxy"]``
    — 79% of wiki2, matched as text), ``Time`` is a year (exact token), and
    ``Quantity`` needs a relative tolerance since rounding and units vary
    (``8270.79237550612`` may surface as "8,270.8 km").
    """
    items = value if isinstance(value, list) else [value]
    if not items or not ctx:
        return False
    hay = pad(norm_text(ctx))

    if datatype == "GlobeCoordinate":
        # A single [lat, lon] pair: require both components to be present.
        nums = numbers_in(ctx)
        try:
            return all(_near(nums, float(c), tol_rel) for c in items)
        except (TypeError, ValueError):
            return False

    if datatype == "WikibaseItem":
        return any((label := norm_text(v)) and pad(label) in hay for v in items)

    nums = numbers_in(ctx)
    for v in items:
        try:
            f = float(v)
        except (TypeError, ValueError):
            if (label := norm_text(v)) and pad(label) in hay:
                return True
            continue
        if datatype == "Time" and f == int(f):
            if pad(str(int(f))) in hay:
                return True
        elif _near(nums, f, tol_rel):
            return True
    return False


# ===========================================================================
# 2. Aggregators — recompute f over a set of values
# ===========================================================================
# Used twice: over V* (the gold-consistency filter) and over the values the agent
# actually surfaced (the aggregation check). Operators whose parameter lives only
# in the question text (SUM_TOP_K, NTH_LATEST, COUNT_LT_X, ... — wiki1) are out of
# scope and yield None, which marks the check "not assessed" rather than failed.

def _haversine(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle distance in km between two [lat, lon] pairs."""
    from math import asin, cos, radians, sin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(h))


def _pairwise(values, reduce_fn):
    pairs = [v for v in values if isinstance(v, (list, tuple)) and len(v) == 2]
    if len(pairs) < 2:
        return None
    return reduce_fn(
        _haversine(pairs[i], pairs[j])
        for i in range(len(pairs)) for j in range(i + 1, len(pairs))
    )


# Numeric reducers, keyed by the gold's ``aggregation`` string (keys copied
# verbatim from the data — note the U+2212 minus sign in DIFFERENCE).
_NUMERIC = {
    "AVG": lambda v: sum(v) / len(v),
    "SUM": sum,
    "MAX": max,
    "MIN": min,
    "LATEST": max,
    "EARLIEST": min,
    "MEDIAN": statistics.median,
    "DIFFERENCE(MAX−MIN)": lambda v: max(v) - min(v),
    "RATIO(MAX/MIN)": lambda v: max(v) / min(v) if min(v) else None,
    "TIME_BETWEEN_FIRST_LAST": lambda v: max(v) - min(v),
}
_COORD = {
    "PAIRWISE_MAX_DISTANCE": lambda v: _pairwise(v, max),
    "PAIRWISE_MIN_DISTANCE": lambda v: _pairwise(v, min),
}

SUPPORTED_AGGREGATIONS = set(_NUMERIC) | set(_COORD) | {"COUNT"}


def _as_float(value: Any) -> Optional[float]:
    """Coerce a gold value (scalar, or a single-item list) to a float."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_f(aggregation: str, values: Sequence[Any], question_norm: str) -> Optional[float]:
    """Reduce *values* with the gold operator; ``None`` when not computable.

    ``COUNT`` is the odd one out: it counts the entities whose value matches a
    target *named in the question* (e.g. "Eastern Orthodoxy"), so it needs the
    padded normalised question text. Positive polarity only — that reproduces the
    stored answer on 87% of wiki2 COUNT rows, and the rest are caught by
    ``gold_ok``, so the old polarity search buys nothing.
    """
    # Guard COUNT as much as the numeric operators: it would happily return 0
    # over an empty set, so a query where the agent surfaced nothing scored as an
    # aggregation *error* under COUNT and as "not assessed" under AVG. COUNT is
    # 80% of wiki2, so that asymmetry biased agg_ok_rate and made its denominator
    # incomparable across operators.
    if not values:
        return None

    if aggregation == "COUNT":
        n = 0
        for v in values:
            items = v if isinstance(v, list) else [v]
            if any((lbl := norm_text(i)) and pad(lbl) in question_norm for i in items):
                n += 1
        return float(n)

    if aggregation in _COORD:
        return _COORD[aggregation](values)

    fn = _NUMERIC.get(aggregation)
    if fn is None:
        return None
    nums = [f for f in (_as_float(v) for v in values) if f is not None]
    if len(nums) != len(values) or not nums:
        return None
    try:
        return fn(nums)
    except (ValueError, ZeroDivisionError, statistics.StatisticsError):
        return None


def extract_answer(generation: str) -> tuple:
    """``(prediction, structured)`` for the agent's final answer.

    ``structured`` is False when :func:`_extract_prediction` found no
    ``\\boxed{}`` / ``<answer>`` / ``Exact Answer:`` marker and fell through to the
    raw prose. That matters because the comparison downstream takes the *first*
    number in the string: on a generation opening "…served before Brezhnev (1966
    onward)" the answer is read as 1966 rather than the 1922 the agent committed
    to. The row is still scored — dropping it would discard every prose answer —
    but the flag is carried into the output so those rows can be audited.
    """
    pred = _extract_prediction(generation)
    if not pred:
        return "", False
    # Reproduce trqa_match's fallback so "did it fall through?" is answerable
    # without re-running the (optional, exception-swallowing) parsers.
    raw = generation
    idx = raw.find("\n\n## References\n")
    if idx != -1:
        raw = raw[:idx]
    return pred, pred != raw.strip()


def close(computed: Optional[float], target: Any, tol_pct: float) -> Optional[bool]:
    """Compare a recomputed value against a stored answer / agent prediction.

    Here — two committed scalars — the evaluator's matcher is exactly right, so
    it is reused as-is and ``agg_ok`` stays commensurable with ``accuracy.jsonl``.
    """
    if computed is None or target in (None, ""):
        return None
    return bool(
        soft_exact_match(computed, target, tolerance_pct=tol_pct).get("soft_match")
    )


# ===========================================================================
# 3. Loading — gold (+ per-entity gold docs) and trajectories
# ===========================================================================

def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _page_of(docid: str) -> str:
    """``"15641-0012"`` -> ``"15641"`` (the Wikipedia page the passage came from)."""
    return str(docid).rsplit("-", 1)[0]


def _title_matches(title_norm: str, entity_norm: str) -> bool:
    """Page title identifies the entity (exact, or disambiguated: "Georgia (U.S. state)")."""
    return title_norm == entity_norm or title_norm.startswith(entity_norm + " ")


def load_qrels(path: str) -> Dict[str, Set[str]]:
    """TREC qrels -> ``qid -> {docid}`` (relevant judgements only)."""
    out: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[3] != "0":
                out.setdefault(parts[0], set()).add(parts[2])
    return out


def load_gold(args) -> Dict[str, dict]:
    """Assemble the gold side: entities, values, operator, answer, per-entity D*.

    The intermediate-info file has no gold documents, but the qrels do — and
    because doc ids are ``pageid-passage``, grouping a query's qrels by page and
    matching the page *title* to the entity name splits the query-level D* into
    per-entity D*_e. Titles come from a cached one-pass scan of the corpus
    (``page_titles.json``); pages that match no gold entity are the roster/list
    documents and are simply left out.
    """
    questions = {str(r["id"]): r.get("text", "") for r in _read_jsonl(args.queries)}
    qrels = load_qrels(args.qrels) if Path(args.qrels).is_file() else {}
    titles: Dict[str, str] = {}
    if Path(args.page_titles).is_file():
        titles = json.loads(Path(args.page_titles).read_text(encoding="utf-8"))
    else:
        logger.warning("no page-title cache at %s — 'retrieved' will be unassessed",
                       args.page_titles)

    gold: Dict[str, dict] = {}
    for rec in _read_jsonl(args.gold):
        qid = str(rec["qid"])
        docs = qrels.get(qid, set())
        by_page: Dict[str, Set[str]] = {}
        for d in docs:
            by_page.setdefault(_page_of(d), set()).add(d)

        entities = []
        for ev in rec.get("entity_values", []):
            name = str(ev.get("entity", ""))
            n = norm_text(name)
            gold_docs = {
                d
                for page, ds in by_page.items()
                if _title_matches(norm_text(titles.get(page) or ""), n)
                for d in ds
            }
            entities.append({"entity": name, "value": ev.get("value"), "gold_docs": gold_docs})

        gold[qid] = {
            "question": questions.get(qid, ""),
            "answer": rec.get("answer"),
            "aggregation": rec.get("aggregation"),
            "datatype": (rec.get("property") or {}).get("datatype"),
            "entities": entities,
        }
    return gold


def load_trajectory(path: Path) -> Optional[dict]:
    """Collapse one ``trajectory/{qid}.jsonl`` into the two channels A and O.

    Citations are stripped from A only. ``generation`` is left verbatim so
    ``agg_ok`` keeps reading the answer exactly the way ``accuracy.jsonl`` does.
    """
    if not path.exists():
        return None
    generation = ""
    agent_text: List[str] = []
    seen: Set[str] = set()
    n_steps = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record") == "meta":
                generation = rec.get("generation", "") or ""
                continue
            n_steps += 1
            agent_text.append(_CITATION_RE.sub(" ", rec.get("think") or ""))
            agent_text.append(_CITATION_RE.sub(" ", rec.get("search_query") or ""))
            seen.update(str(d) for d in (rec.get("seen_docs") or []))
    return {
        "generation": generation,
        "agent_text": " ".join(agent_text),
        "seen_docs": seen,
        "n_steps": n_steps,
    }


def load_trajectories(args, gold: Dict[str, dict]) -> Dict[str, dict]:
    """Load every trajectory in the run that has a matching gold row."""
    traj_dir = Path(args.run_dir) / "trajectory"
    if not traj_dir.is_dir():
        raise SystemExit(f"no trajectory/ under {args.run_dir}")

    qids = args.qids or sorted(p.stem for p in traj_dir.glob("*.jsonl"))
    missing = [q for q in qids if q not in gold]
    qids = [q for q in qids if q in gold]
    if args.limit:
        qids = qids[: args.limit]
    if missing:
        logger.warning("%d trajectories had no gold row (skipped)", len(missing))

    trajs = {}
    for qid in qids:
        t = load_trajectory(traj_dir / f"{qid}.jsonl")
        if t is None:
            logger.warning("no trajectory for %s", qid)
            continue
        trajs[qid] = t
    return trajs


# ===========================================================================
# 4. Analysis — one row per (query, gold entity), plus one row per query
# ===========================================================================

def analyze_qid(qid: str, gold: dict, traj: dict, tol_pct: float,
                value_tol_rel: float) -> Tuple[List[dict], dict]:
    """Score one query.

    Returns ``(entity_rows, query_row)``: the per-entity table (micro view,
    query-level fields repeated onto every row) and the per-query scores (macro
    view, one point per query for the distribution figures).
    """
    hay = pad(norm_text(traj["agent_text"]))
    soft = norm_soft(traj["agent_text"])
    question_norm = pad(norm_text(gold["question"]))
    datatype = gold["datatype"]
    aggregation = gold["aggregation"]
    entities = gold["entities"]
    names = [e["entity"] for e in entities]

    rows = []
    for e in entities:
        # No gold docs resolved for this entity (title lookup failed, or the
        # query has no qrels) -> unassessable, not a retrieval failure.
        gold_docs = e["gold_docs"]
        seen_docs = gold_docs & traj["seen_docs"]
        rows.append({
            "qid": qid,
            "entity": e["entity"],
            "aggregation": aggregation,
            "e_in_A": entity_in(e["entity"], hay, names),
            "v_in_A": value_in(
                e["value"], local_context(soft, name_variants(e["entity"], names)),
                datatype, value_tol_rel,
            ),
            "retrieved": bool(seen_docs) if gold_docs else None,
            # Not metrics — diagnostics for "did the title resolution work, and
            # how much of D*_e did the run actually touch".
            "n_gold_docs": len(gold_docs),
            "n_gold_docs_seen": len(seen_docs),
        })

    values = [e["value"] for e in entities]
    gold_ok = close(apply_f(aggregation, values, question_norm), gold["answer"], tol_pct)

    found = [e["value"] for e, r in zip(entities, rows) if r["v_in_A"]]
    f_found = apply_f(aggregation, found, question_norm)
    a_pred, a_pred_structured = extract_answer(traj["generation"])
    agg_ok = close(f_found, a_pred, tol_pct)
    if agg_ok is not None and not a_pred_structured:
        logger.warning(
            "%s: generation carries no structured answer — agg_ok compares against "
            "the first number in the prose", qid
        )

    for r in rows:
        r["agg_ok"] = agg_ok
        r["gold_ok"] = gold_ok
        r["a_pred_structured"] = a_pred_structured

    return rows, _query_row(qid, gold, traj, rows, f_found, a_pred,
                            a_pred_structured, agg_ok, gold_ok, tol_pct)


def _query_row(qid, gold, traj, rows, f_found, a_pred, a_pred_structured,
               agg_ok, gold_ok, tol_pct) -> dict:
    """Collapse the entity rows of one query into its per-query scores.

    Each score carries its own denominator, and ``None`` always means *not
    assessed* rather than zero — a query where no gold page title resolved has no
    retrieval recall at all, and plotting it as 0.0 would report a title-lookup
    miss as a total retrieval failure.
    """
    n = len(rows)
    named = sum(1 for r in rows if r["e_in_A"])
    read = sum(1 for r in rows if r["v_in_A"])
    ret_assessed = [r for r in rows if r["retrieved"] is not None]

    return {
        "qid": qid,
        "aggregation": gold["aggregation"],
        "datatype": gold["datatype"],
        "n_entities": n,
        "n_ret_assessed": len(ret_assessed),
        # P(q) — over every gold entity; always defined.
        "plan_cov": round(named / n, 6) if n else None,
        # R(q) — over the entities whose gold docs resolved.
        "ret_recall": (
            round(sum(1 for r in ret_assessed if r["retrieved"]) / len(ret_assessed), 6)
            if ret_assessed else None
        ),
        # G(q) — bounded above by plan_cov by construction (see module docstring).
        "read_cov": round(read / n, 6) if n else None,
        # G|pi(q) — the version that actually isolates reading.
        "read_cov_given_named": round(read / named, 6) if named else None,
        "agg_ok": agg_ok,
        "gold_ok": gold_ok,
        "f_found": f_found,
        "a_pred": a_pred,
        "answer": gold["answer"],
        # `close` is order-sensitive: soft_exact_match takes the tolerance
        # relative to its second argument, so the gold answer goes second — the
        # same way accuracy.jsonl scores it.
        "answer_correct": close(a_pred, gold["answer"], tol_pct) if a_pred else False,
        "a_pred_structured": a_pred_structured,
        "n_steps": traj.get("n_steps", 0),
        "n_seen_docs": len(traj["seen_docs"]),
    }


def _rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    """Mean over the entries that were actually assessed (``None`` = skipped)."""
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _dist(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """``mean`` / ``median`` / ``n`` / ``n_null`` over one per-query score."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "median": None, "n": 0, "n_null": len(values)}
    return {
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "n": len(vals),
        "n_null": len(values) - len(vals),
    }


def summarize(rows: List[dict], qrows: Sequence[dict] = ()) -> Dict[str, Any]:
    """Micro rates over the entity table, plus macro stats over the query table.

    The two blocks answer different questions and routinely disagree: micro is
    entity-weighted, so a 50-entity query counts 50x, while macro weights every
    query equally. Both are reported so the tables and the figures agree.
    """
    per_query = {}
    for r in rows:
        per_query[r["qid"]] = (
            r["agg_ok"], r["gold_ok"], r["aggregation"], r["a_pred_structured"]
        )

    summary = {
        "n_queries": len(per_query),
        "n_entities": len(rows),
        "named_rate": _rate([r["e_in_A"] for r in rows]),
        "retrieved_rate": _rate([r["retrieved"] for r in rows]),
        "value_rate": _rate([r["v_in_A"] for r in rows]),
        "n_retrieval_assessed": sum(1 for r in rows if r["retrieved"] is not None),
        "agg_ok_rate": _rate([v[0] for v in per_query.values()]),
        "n_agg_assessed": sum(1 for v in per_query.values() if v[0] is not None),
        "gold_ok_rate": _rate([v[1] for v in per_query.values()]),
        "n_unsupported_aggregation": sum(
            1 for v in per_query.values() if v[2] not in SUPPORTED_AGGREGATIONS
        ),
        "n_agg_pred_unstructured": sum(
            1 for v in per_query.values() if v[0] is not None and not v[3]
        ),
    }
    if qrows:
        summary["macro"] = {
            key: _dist([r[key] for r in qrows])
            for key in ("plan_cov", "ret_recall", "read_cov", "read_cov_given_named")
        }
        summary["answer_correct_rate"] = _rate([r["answer_correct"] for r in qrows])
    return summary


# ===========================================================================
# 5. Output — the two jsonl tables (figures live in reasoning_error_plots)
# ===========================================================================

def format_summary_table(summary: Dict[str, Any]) -> str:
    """Compact stdout report: the micro marginals, then the macro distributions."""
    def fmt(x):
        return "  n/a" if x is None else f"{x:5.3f}"

    n_e, n_q = summary["n_entities"], summary["n_queries"]
    lines = [
        "",
        "=" * 72,
        f"REASONING-ERROR ANALYSIS  ({n_q} queries, {n_e} gold entities)",
        "=" * 72,
        "  MICRO  (pooled over entities)",
        f"    named in A     (planning)     {fmt(summary['named_rate'])}   {n_e} entities",
        f"    gold doc seen  (retrieval)    {fmt(summary['retrieved_rate'])}   "
        f"{summary['n_retrieval_assessed']} assessed",
        f"    value in A     (reading)      {fmt(summary['value_rate'])}   {n_e} entities",
        f"    f(V_found)=ans (aggregation)  {fmt(summary['agg_ok_rate'])}   "
        f"{summary['n_agg_assessed']} assessed",
    ]

    macro = summary.get("macro")
    if macro:
        lines += ["-" * 72, "  MACRO  (per query, then averaged — every query weighs the same)",
                  f"    {'score':<26}{'mean':>7}{'median':>9}{'n':>7}{'n/a':>6}"]
        for key, label in (
            ("plan_cov", "plan coverage"),
            ("ret_recall", "retrieval recall"),
            ("read_cov", "reading coverage"),
            ("read_cov_given_named", "reading | named"),
        ):
            d = macro[key]
            lines.append(
                f"    {label:<26}{fmt(d['mean'])}{fmt(d['median']):>9}"
                f"{d['n']:>7}{d['n_null']:>6}"
            )
        if summary.get("answer_correct_rate") is not None:
            lines.append(
                f"    {'end-to-end correct':<26}{fmt(summary['answer_correct_rate'])}"
                f"{'':>9}{n_q:>7}{0:>6}"
            )

    lines += [
        "-" * 72,
        f"  gold recomputable             {fmt(summary['gold_ok_rate'])}   "
        f"({summary['n_unsupported_aggregation']} queries: unsupported operator)",
        f"  unstructured final answer     {summary['n_agg_pred_unstructured']:5d}   "
        f"of the assessed (agg_ok read the first number in prose)",
        "=" * 72,
    ]
    return "\n".join(lines)


def _write_table(records: List[dict], meta: Dict[str, Any], out_path: Path,
                 label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, separators=(",", ":"), default=str) + "\n")
        for r in records:
            f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
    print(f"  Saved {label:<26} {out_path}")


def write_jsonl(rows: List[dict], summary: Dict[str, Any], run_key: Dict[str, Any],
                out_path: Path) -> None:
    """The per-entity table: meta header line, then one row per (qid, entity)."""
    _write_table(rows, {"record": "meta", **run_key, **summary}, out_path,
                 "reasoning-error rows:")


def write_query_jsonl(qrows: List[dict], summary: Dict[str, Any],
                      run_key: Dict[str, Any], out_path: Path) -> None:
    """The per-query table — the only input the figures read."""
    _write_table(qrows, {"record": "meta", **run_key, **summary}, out_path,
                 "per-query rows:")
