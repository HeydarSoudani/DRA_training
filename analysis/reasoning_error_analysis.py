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

Everything except the CLI lives in :mod:`analysis.utils.reasoning_error_utils`
(loading, scoring, summary, plotting) and :mod:`analysis.utils.config` (path /
config resolution).  This file just parses the CLI, loops over queries to produce
a per-query *score*, writes the jsonl, and re-plots it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_SELF_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SELF_DIR)]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

from analysis.utils import config as cfg  # noqa: E402
from analysis.utils import reasoning_error_utils as reu  # noqa: E402
from utils.config import AGENTIC_MODEL_TO_LLM  # noqa: E402

# Mirror experiments/dra_inference.py: load .env so OPENROUTER_API_KEY is present
# and resolve_agent_backend() derives the same `_api_` run-dir the run was
# written under (without it the backend falls back to `_vllm_` and the path mismatches).
load_dotenv()

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
    parser.add_argument("--subset", type=str, default="wiki2", help="Dataset subset/collection (null = auto-selected from --dataset). trqa: wiki1|wiki2|ecommerce; neuclir: news|technical|report; browsecomp_plus: test.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever used for the run (selects the run dir).")
    parser.add_argument("--controller", type=str, default="off", choices=["off", "monitor", "action"], help="Controller mode of the run (selects the run dir).")
    parser.add_argument("--controller-prompt-variant", type=str, default="nov_cov_sim", choices=["nov", "nov_cov", "nov_sim", "nov_cov_sim", "sim", "cov_sim"], help="Controller policy prompt variant of the run (only affects the dir name when --controller action).")

    # ── Run-control flags ──────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="analyze at most N queries")
    parser.add_argument("--qids", nargs="*", default=None, help="restrict to these query ids")
    parser.add_argument("--judge", action="store_true", help="enable the gated LLM judges")
    parser.add_argument("--plot-only", type=str, default=None, metavar="PATH", help="skip attribution: re-plot an existing reasoning_error_per_query.jsonl (or the dir containing it) and exit")
    parser.add_argument("-v", "--verbose", action="store_true")

    args, extras = parser.parse_known_args()

    cli_subset = args.subset
    config = cfg.load_config(args.config)
    overrides = cfg.parse_cli_overrides(extras)
    cfg.apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset

    return args


# ============================================================================
# Plot-only mode: re-plot a written jsonl without re-computing attribution
# ============================================================================
def _plot_only(path: Path, out: str | None) -> None:
    """Re-render figures + CSV from a written ``reasoning_error_per_query.jsonl``.

    Because plotting reads only the jsonl (see ``reu.render_all``), you can
    iterate on the figures — or point at any past run — without re-attributing.
    Accepts either the jsonl file or the directory that holds it.
    """
    jsonl_path = path / "reasoning_error_per_query.jsonl" if path.is_dir() else path
    if not jsonl_path.is_file():
        raise SystemExit(f"no such jsonl: {jsonl_path}")
    out_dir = Path(out) if out else jsonl_path.parent

    _records, summary = reu.load_records(jsonl_path)
    print(reu.format_summary_table(summary))
    reu.render_all(jsonl_path, out_dir)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.plot_only:
        _plot_only(Path(args.plot_only), args.out)
        return

    # Derive run_dir / gold / corpus / out from the run-selecting knobs.
    cfg.resolve_paths(args)

    gold, trajs, corpus, judge_client, n_missing_gold = reu.load_inputs(args)

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

    # ── Write jsonl first: it is the single source of truth for plotting ────
    out_dir = Path(args.out)
    reu.write_outputs(records, summary, out_dir)

    # Compact stage table to stdout (full summary is written to summary.json).
    print(reu.format_summary_table(summary))

    # ── Plot from the written jsonl (decoupled from attribution) ────────────
    run_key = {
        "agentic_model": args.agentic_model,
        "retriever": args.retriever,
        "controller": args.controller,
        "controller_prompt_variant": args.controller_prompt_variant,
    }
    jsonl_path = out_dir / "reasoning_error_per_query.jsonl"
    logger.info("[viz] %s -> %s", jsonl_path, out_dir)
    reu.render_all(jsonl_path, out_dir, run_key=run_key)


if __name__ == "__main__":
    main()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
# python analysis/reasoning_error_analysis.py --limit 1
# python analysis/reasoning_error_analysis.py --agentic-model glm --subset wiki2 --judge
# python analysis/reasoning_error_analysis.py --plot-only path/to/run/analysis/  # re-plot only
