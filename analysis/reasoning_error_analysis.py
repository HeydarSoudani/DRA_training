"""Reasoning-error attribution for TRQA agent trajectories (entry point).

Given a TRQA run (agent trajectories under ``run_outputs/``) and the per-query
**intermediate-info** gold, this script localises *why* each trajectory failed
by assigning it a primary root-cause stage:

    planning  ->  retrieval  ->  reading / aggregation

The method is deliberately **rule-based**; two small LLM judges are optional
(``--judge``) and only run on the slices where the rules are provably blind.

Usage mirrors ``experiments/dra_inference.py``: pass the same run-selecting
knobs (``--agentic-model``, ``--dataset``, ``--retriever``, ``--controller``,
``--controller-prompt-variant``) and the run directory, gold file and corpus are
derived automatically.  Everything else lives in the YAML config
(``--config``, default experiments/configs/dra_analysis.yaml) and any of it can be
overridden with the matching ``--flag`` (e.g. ``--subset wiki2``, ``--gold /p``).

The scoring machinery lives in :mod:`utils.reasoning_error_utils`; path/config
resolution in :mod:`utils.config`.  This file just parses the CLI, loops over
queries to produce a per-query *score*, and hands the scores to visualisation.

Run::

    python analysis/reasoning_error_analysis.py --agentic-model glm --controller off
    python analysis/reasoning_error_analysis.py --agentic-model glm --subset wiki2 --judge
    python analysis/reasoning_error_analysis.py --self-test   # validate the rules
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Import as ``analysis.utils.*`` off the repo root so the analysis subpackage
# (also named ``utils``) never shadows the repo-root ``utils`` package. Drop the
# script's own dir from sys.path — that is what would cause the shadowing when
# launched as ``python analysis/reasoning_error_analysis.py``.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_SELF_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SELF_DIR)]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.utils import config as cfg  # noqa: E402
from analysis.utils import reasoning_error_utils as reu  # noqa: E402
from analysis.utils import reasoning_error_viz as viz  # noqa: E402
from utils.config import AGENTIC_MODEL_TO_LLM  # noqa: E402

logger = logging.getLogger("reasoning_error_analysis")


# ============================================================================
# CLI
# ============================================================================
def _parse_args():
    """Parse the frequently-varied knobs; merge the file-backed config on top.

    Only the run-selecting knobs are real CLI arguments (same set as
    dra_inference).  The mostly-fixed variables come from ``--config`` and can
    still be overridden by passing the matching ``--flag`` (handled via
    ``parse_known_args`` -> ``apply_config_to_args``).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── File-backed config ─────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=cfg.CONFIG_DEFAULT, help="YAML holding the mostly-fixed analysis variables; any value can be overridden with the matching --flag.")

    # ── Frequently-varied knobs (select which run to analyse) ──────────────
    parser.add_argument("--agentic-model", type=str, default="glm", choices=list(AGENTIC_MODEL_TO_LLM), help="Agent whose run to analyse (selects the run dir).")
    parser.add_argument("--dataset", type=str, default="trqa", choices=["trqa", "browsecomp_plus", "neuclir"], help="Dataset. Reasoning-error attribution needs TRQA intermediate-info gold, so only 'trqa' is supported.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever used for the run (selects the run dir).")
    parser.add_argument("--controller", type=str, default="off", choices=["off", "monitor", "action"], help="Controller mode of the run (selects the run dir).")
    parser.add_argument("--controller-prompt-variant", type=str, default="nov_cov_sim", choices=["nov", "nov_cov", "nov_sim", "nov_cov_sim", "sim", "cov_sim"], help="Controller policy prompt variant of the run (only affects the dir name when --controller action).")

    # ── Run-control flags ──────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="analyze at most N queries")
    parser.add_argument("--qids", nargs="*", default=None, help="restrict to these query ids")
    parser.add_argument("--judge", action="store_true", help="enable the gated LLM judges")
    parser.add_argument("--self-test", action="store_true", help="run the built-in rule checks and exit")
    parser.add_argument("-v", "--verbose", action="store_true")

    args, extras = parser.parse_known_args()

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    config = cfg.load_config(args.config)
    overrides = cfg.parse_cli_overrides(extras)
    cfg.apply_config_to_args(args, config, overrides)

    return args


# ============================================================================
# Loading
# ============================================================================
def _load_inputs(args) -> tuple:
    """Load gold, trajectories and (optionally) the retrieved-doc corpus + judge.

    Returns ``(gold, trajs, corpus, judge_client, n_missing_gold)``.
    """
    gold = reu.load_gold(args.gold)
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
        t = reu.load_trajectory(traj_dir / f"{qid}.jsonl")
        if t is None:
            logger.warning("no trajectory for %s", qid)
            continue
        trajs[qid] = t

    corpus = None
    if args.corpus:
        needed = set().union(*(t["seen_docs"] for t in trajs.values())) if trajs else set()
        corpus = reu.load_corpus_subset(args.corpus, needed)
        logger.info("loaded %d/%d retrieved docs from corpus", len(corpus), len(needed))
    else:
        logger.warning("no --corpus: per-entity retrieval (E_ret) will be left unassessed")

    judge_client = reu.get_judge_client(args.judge_model) if args.judge else None

    return gold, trajs, corpus, judge_client, n_missing_gold


# ============================================================================
# Visualization (stubs — to be implemented later)
# ============================================================================
def visualize(records: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path,
              run_key: Dict[str, str] | None = None) -> None:
    """Render the per-query scores + summary into PNG figures (+ tidy CSV).

    The figures share one stage-colour legend so the set reads as one system:
      * reasoning_error_funnel.png          — headline drop-off through the gates
      * reasoning_error_by_complexity.png   — stage mix vs #gold entities
      * reasoning_error_by_aggregation.png  — stage mix vs aggregation operator
      * reasoning_error_downstream.png      — reading vs aggregation-logic split
      * reasoning_error_coverage.png        — plan- vs retrieval-recall scatter
      * reasoning_error_per_query.csv       — one row/query, keyed by the run
    See :mod:`analysis.utils.reasoning_error_viz` for the rendering logic.
    """
    logger.info("[viz] %d per-query scores -> %s", len(records), out_dir)
    viz.render_all(records, summary, out_dir, run_key=run_key)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.self_test:
        reu.self_test()
        return

    # Derive run_dir / gold / corpus / out from the run-selecting knobs.
    cfg.resolve_paths(args)

    gold, trajs, corpus, judge_client, n_missing_gold = _load_inputs(args)

    # ── Loop over queries: each produces one score (root-cause record) ──────
    records: List[Dict[str, Any]] = []
    for qid, traj in trajs.items():
        score = reu.analyze_qid(
            qid, gold[qid], traj,
            args.rel_tol, args.abs_tol, args.gold_tol_abs, args.gold_tol_rel,
            corpus, judge_client,
        )
        records.append(score)

    if n_missing_gold:
        logger.warning("%d queries had no gold decomposition (skipped)", n_missing_gold)

    summary = reu.summarize(records)
    print(json.dumps(summary, indent=2))

    out_dir = Path(args.out)
    reu.write_outputs(records, summary, out_dir)

    # ── Hand the scores to visualisation ────────────────────────────────────
    run_key = {
        "agentic_model": args.agentic_model,
        "retriever": args.retriever,
        "controller": args.controller,
        "controller_prompt_variant": args.controller_prompt_variant,
    }
    visualize(records, summary, out_dir, run_key=run_key)


if __name__ == "__main__":
    main()
