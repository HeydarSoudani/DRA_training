"""Matplotlib rendering for reasoning-error attribution (PNG figures + CSV).

The driver (:mod:`analysis.reasoning_error_analysis`) hands us the per-query
*records* and the *summary* produced by :mod:`reasoning_error_utils`; we turn
them into a small, self-consistent set of figures whose shared job is to answer
one question — **where do the agents fail?**

Design (from the data model, not taste):

  * Attribution is a strictly ordered, mutually-exclusive, earliest-failing
    assignment over three gates — ``planning -> retrieval -> reading/aggregation``
    — plus a ``success`` sink and a ``gold_incomplete`` side-channel (a data
    problem, never blamed on the agent). That ordering is *literally a funnel*,
    so the headline figure is a funnel, and every other chart reuses the same
    stage colour legend so the set reads as one system.

Figures written to ``out_dir``:
    reasoning_error_funnel.png          headline drop-off funnel
    reasoning_error_by_complexity.png   stage mix vs #gold entities (hop count)
    reasoning_error_by_aggregation.png  stage mix vs aggregation operator
    reasoning_error_downstream.png      reading vs which-wrong-operator split
    reasoning_error_coverage.png        plan-recall vs retrieval-recall scatter
    reasoning_error_per_query.csv       tidy per-query rows (for cross-run later)
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("reasoning_error_analysis")

# ---------------------------------------------------------------------------
# Palette (validated reference palette, light surface — PNGs are for papers).
# Stage colours are assigned once, in fixed pipeline order, and reused by every
# chart so identity is never colour-alone-inconsistent across figures.
# ---------------------------------------------------------------------------
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"

# pipeline order: a query "flows" success <- past every gate; failures fall out
# at the earliest broken gate. gold_incomplete is excluded from the funnel.
STAGE_ORDER = ["success", "planning", "retrieval", "reading_or_aggregation", "gold_incomplete"]
STAGE_LABELS = {
    "success": "Success",
    "planning": "Planning",
    "retrieval": "Retrieval",
    "reading_or_aggregation": "Reading / aggregation",
    "gold_incomplete": "Gold incomplete (excluded)",
}
STAGE_COLORS = {
    "success": "#0ca30c",                 # status: good
    "planning": "#2a78d6",                # categorical blue
    "retrieval": "#eb6834",               # categorical orange
    "reading_or_aggregation": "#e34948",  # categorical red
    "gold_incomplete": _MUTED,            # excluded / not attributable
}

# ordinal blue ramp for the funnel survivor bars (light-surface safe steps)
_FUNNEL_RAMP = ["#5598e7", "#3987e5", "#256abf", "#184f95"]


def _apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": _SURFACE,
        "axes.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
        "axes.edgecolor": _BASELINE,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "grid.color": _GRID,
        "grid.linewidth": 1.0,
        "text.color": _INK,
        "axes.labelcolor": _INK_2,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "font.family": "sans-serif",
        "svg.fonttype": "none",
    })


def _despine(ax, keep=("bottom",)) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def _stage_legend(fig, stages: List[str]) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=STAGE_COLORS[s], edgecolor=_SURFACE, label=STAGE_LABELS[s]) for s in stages]
    fig.subplots_adjust(bottom=0.26)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(stages), 3),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))


def _retrieval_assessed(records: List[Dict[str, Any]]) -> bool:
    return any(r.get("retrieval_recall") is not None for r in records)


# ---------------------------------------------------------------------------
# 1. Headline funnel — drop-off through the pipeline gates
# ---------------------------------------------------------------------------

def plot_funnel(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """Centred funnel over gold-consistent queries: how many survive each gate.

    Survivors are reconstructed from the earliest-failing attribution:
        entered            = gold-consistent queries
        past planning       = entered            - #planning
        past retrieval      = past planning       - #retrieval   (if assessed)
        success             = past retrieval      - #reading/agg
    The between-bar annotations name each gate's loss in that stage's colour.
    """
    import matplotlib.pyplot as plt

    counts = {s: 0 for s in STAGE_ORDER}
    for r in records:
        counts[r.get("primary_stage", "gold_incomplete")] = counts.get(r.get("primary_stage"), 0) + 1

    n_incomplete = counts["gold_incomplete"]
    entered = len(records) - n_incomplete
    if entered <= 0:
        logger.warning("[viz] funnel skipped: no gold-consistent queries")
        return None

    assessed = _retrieval_assessed(records)
    # levels: (label, survivors, loss_count, loss_stage)
    past_plan = entered - counts["planning"]
    if assessed:
        past_ret = past_plan - counts["retrieval"]
        levels = [
            ("Gold-consistent queries", entered, None, None),
            ("Plan enumerates all entities", past_plan, counts["planning"], "planning"),
            ("Retrieval surfaces them", past_ret, counts["retrieval"], "retrieval"),
            ("Answer correct", counts["success"], counts["reading_or_aggregation"], "reading_or_aggregation"),
        ]
    else:
        # no corpus: retrieval folded into reading/aggregation — collapse the gate
        levels = [
            ("Gold-consistent queries", entered, None, None),
            ("Plan enumerates all entities", past_plan, counts["planning"], "planning"),
            ("Answer correct", counts["success"], counts["reading_or_aggregation"], "reading_or_aggregation"),
        ]

    fig, ax = plt.subplots(figsize=(8.6, 0.95 * len(levels) + 1.6))
    ramp = _FUNNEL_RAMP[-len(levels):]
    y_positions = list(range(len(levels)))[::-1]  # first level on top

    for (label, surv, loss, loss_stage), y, base in zip(levels, y_positions, ramp):
        left = (entered - surv) / 2.0
        color = STAGE_COLORS["success"] if label == "Answer correct" else base
        ax.barh(y, surv, left=left, height=0.62, color=color,
                edgecolor=_SURFACE, linewidth=1.5, zorder=3)
        pct = 100.0 * surv / entered
        ax.text(entered / 2.0, y, f"{label}\n{surv}  ({pct:.0f}%)",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
        if loss:
            lpct = 100.0 * loss / entered
            ax.text(entered + entered * 0.015, y + 0.5, f"−{loss}  {STAGE_LABELS[loss_stage]}  ({lpct:.0f}%)",
                    ha="left", va="center", color=STAGE_COLORS[loss_stage], fontsize=9, fontweight="bold")

    ax.set_xlim(-entered * 0.02, entered * 1.30)
    ax.set_ylim(-0.6, len(levels) - 0.4)
    ax.axis("off")
    title = "Where do the agents fall out? — reasoning-error funnel"
    sub = f"{entered} gold-consistent queries"
    if n_incomplete:
        sub += f"   •   {n_incomplete} excluded (gold incomplete)"
    if not assessed:
        sub += "   •   retrieval unassessed (no corpus)"
    ax.set_title(title, loc="left", pad=22)
    ax.text(0, 1.0, sub, transform=ax.transAxes, ha="left", va="bottom",
            color=_MUTED, fontsize=9.5)

    p = out / "reasoning_error_funnel.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# helper: 100% stacked horizontal bars sharing the stage colour legend
# ---------------------------------------------------------------------------

def _stacked_by(ax, rows: List[str], counts_by_row: Dict[str, Dict[str, int]], stages: List[str]) -> None:
    y = list(range(len(rows)))[::-1]
    for yi, row in zip(y, rows):
        total = sum(counts_by_row[row].get(s, 0) for s in stages) or 1
        left = 0.0
        for s in stages:
            c = counts_by_row[row].get(s, 0)
            if c == 0:
                continue
            frac = c / total
            ax.barh(yi, frac, left=left, height=0.66, color=STAGE_COLORS[s],
                    edgecolor=_SURFACE, linewidth=1.5, zorder=3)
            if frac > 0.06:
                ax.text(left + frac / 2.0, yi, str(c), ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold", zorder=4)
            left += frac
    ax.set_yticks(y)
    ax.set_yticklabels(rows, color=_INK, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    _despine(ax, keep=("bottom",))


def _bin_entities(n: Optional[int]) -> str:
    if not n:
        return "?"
    return "5+" if n >= 5 else str(n)


def plot_by_complexity(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """Stage mix vs #gold entities (a proxy for hop-count / question difficulty)."""
    import matplotlib.pyplot as plt

    from collections import Counter, defaultdict

    buckets: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        buckets[_bin_entities(r.get("n_gold_entities"))][r.get("primary_stage", "gold_incomplete")] += 1
    order = [b for b in ["2", "3", "4", "5+", "?"] if b in buckets]
    if not order:
        return None
    stages = [s for s in STAGE_ORDER if any(buckets[b].get(s) for b in order)]

    fig, ax = plt.subplots(figsize=(8.6, 0.7 * len(order) + 2.0))
    rows = [f"{b}  (n={sum(buckets[b].values())})" for b in order]
    _stacked_by(ax, rows, {r: buckets[b] for r, b in zip(rows, order)}, stages)
    ax.set_ylabel("# gold entities")
    ax.set_title("Does it get harder with more entities?", loc="left", pad=26)
    _stage_legend(fig, stages)

    p = out / "reasoning_error_by_complexity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_by_aggregation(summary: Dict[str, Any], out: Path) -> Optional[Path]:
    """Stage mix vs aggregation operator (COUNT/AVG/SUM/... ) from ``by_aggregation``."""
    import matplotlib.pyplot as plt

    by_agg = summary.get("by_aggregation") or {}
    if not by_agg:
        return None
    aggs = sorted(by_agg, key=lambda a: -sum(by_agg[a].values()))
    stages = [s for s in STAGE_ORDER if any(by_agg[a].get(s) for a in aggs)]

    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(aggs) + 2.0))
    rows = [f"{a}  (n={sum(by_agg[a].values())})" for a in aggs]
    _stacked_by(ax, rows, {r: by_agg[a] for r, a in zip(rows, aggs)}, stages)
    ax.set_ylabel("aggregation")
    ax.set_title("Which operators are hardest?", loc="left", pad=26)
    _stage_legend(fig, stages)

    p = out / "reasoning_error_by_aggregation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_downstream(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """Reading vs which-wrong-operator split within the reading/aggregation bucket."""
    import matplotlib.pyplot as plt
    from collections import Counter

    subs = Counter(r["downstream_subtype"] for r in records
                   if r.get("primary_stage") == "reading_or_aggregation" and r.get("downstream_subtype"))
    if not subs:
        return None
    items = subs.most_common()
    labels = [k.replace("aggregation_logic:", "wrong op: ") for k, _ in items]
    vals = [v for _, v in items]
    # reading errors get the red stage colour; aggregation-logic get orange
    colors = [STAGE_COLORS["reading_or_aggregation"] if k.startswith("reading") else "#eb6834"
              for k, _ in items]

    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(items) + 1.8))
    y = list(range(len(items)))[::-1]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor=_SURFACE, linewidth=1.5, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yi, str(v), va="center", ha="left",
                color=_INK_2, fontsize=9, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=_INK, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="x", zorder=0)
    _despine(ax, keep=("bottom",))
    ax.set_xlabel("# queries")
    ax.set_title("Reading vs aggregation-logic errors", loc="left", pad=12)

    p = out / "reasoning_error_downstream.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_coverage(records: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    """plan-recall (x) vs retrieval-recall (y), coloured by stage. Needs a corpus."""
    if not _retrieval_assessed(records):
        return None
    import matplotlib.pyplot as plt

    # deterministic jitter (no RNG): spread by index within each (x,y) cell
    from collections import defaultdict
    cell = defaultdict(list)
    pts = []
    for r in records:
        x, yv = r.get("plan_recall"), r.get("retrieval_recall")
        if x is None or yv is None:
            continue
        key = (round(x, 3), round(yv, 3))
        cell[key].append(r)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    stages_present = set()
    for (x, yv), rs in cell.items():
        k = len(rs)
        for i, r in enumerate(rs):
            ang = (i / max(k, 1)) * 6.28318
            rad = 0.012 * (i > 0) * ((i % 5) + 1)
            jx = x + rad * __import__("math").cos(ang)
            jy = yv + rad * __import__("math").sin(ang)
            s = r.get("primary_stage", "gold_incomplete")
            stages_present.add(s)
            size = 24 + 10 * (r.get("n_gold_entities") or 0)
            ax.scatter(jx, jy, s=size, color=STAGE_COLORS.get(s, _MUTED),
                       edgecolor=_SURFACE, linewidth=0.8, alpha=0.85, zorder=3)
    ax.plot([0, 1], [0, 1], color=_BASELINE, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("plan recall  (entities enumerated / gold)")
    ax.set_ylabel("retrieval recall  (entities surfaced / gold)")
    ax.grid(True, zorder=0)
    _despine(ax, keep=("bottom", "left"))
    ax.set_title("Where is entity coverage lost?", loc="left", pad=12)
    stages = [s for s in STAGE_ORDER if s in stages_present]
    _stage_legend(fig, stages)

    p = out / "reasoning_error_coverage.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CSV export — tidy per-query rows so a thin cross-run aggregator can stack
# multiple configs without re-running attribution.
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "qid", "primary_stage", "aggregation", "downstream_subtype", "gold_consistent",
    "n_gold_entities", "n_planned", "n_retrieved", "plan_recall", "retrieval_recall",
    "num_steps", "a_pred", "answer",
]


def export_csv(records: List[Dict[str, Any]], summary: Dict[str, Any], out: Path,
               run_key: Optional[Dict[str, str]] = None) -> Path:
    run_key = run_key or {}
    key_fields = list(run_key)
    p = out / "reasoning_error_per_query.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(key_fields + _CSV_FIELDS)
        for r in records:
            w.writerow([run_key[k] for k in key_fields]
                       + [r.get(k) for k in _CSV_FIELDS])
    return p


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def render_all(records: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path,
               run_key: Optional[Dict[str, str]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        logger.warning("[viz] no records to plot")
        return
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
    except Exception as e:  # pragma: no cover
        logger.warning("[viz] matplotlib unavailable (%s) — skipping figures", e)
        export_csv(records, summary, out_dir, run_key)
        return

    _apply_style()
    written = []
    for fn in (lambda: plot_funnel(records, out_dir),
               lambda: plot_by_complexity(records, out_dir),
               lambda: plot_by_aggregation(summary, out_dir),
               lambda: plot_downstream(records, out_dir),
               lambda: plot_coverage(records, out_dir)):
        try:
            p = fn()
            if p:
                written.append(p)
        except Exception as e:  # a single bad figure must not sink the run
            logger.warning("[viz] figure failed: %s", e)
    written.append(export_csv(records, summary, out_dir, run_key))
    for p in written:
        logger.info("[viz] wrote %s", p)
