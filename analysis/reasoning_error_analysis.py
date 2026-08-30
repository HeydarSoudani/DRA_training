"""Reasoning-error analysis for TRQA agent trajectories (entry point).

Compares the gold decomposition of each query against the agent's trajectory and
reports, per gold entity, where the correspondence breaks:

    e_in_A     entity named in think+query text        -> planning
    retrieved  a gold doc for that entity was seen     -> retrieval
    v_in_A     the gold value stated in think+query    -> reading
    agg_ok     f(values found) matches the answer      -> aggregation

These are marginals, not ordered stages — see
:mod:`analysis.utils.reasoning_error_utils` for the method.

Two tables come out: one row per (query, entity) for the pooled/micro view, and
one row per query holding that query's plan / retrieval / reading coverage, which
is what the distribution figures are drawn from.

Usage mirrors ``experiments/dra_inference.py``: pass the same run-selecting knobs
(``--agentic-model``, ``--dataset``, ``--retriever``, ``--controller``,
``--controller-prompt-variant``) and the run directory, gold, queries and qrels
are derived automatically. Everything else lives in the YAML config (``--config``,
default experiments/configs/dra_analysis.yaml) and can be overridden with the
matching ``--flag`` (e.g. ``--subset wiki2``).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SELF_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SELF_DIR)]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402

from analysis.utils import config as cfg  # noqa: E402
from analysis.utils import reasoning_error_plots as rep  # noqa: E402
from analysis.utils import reasoning_error_utils as reu  # noqa: E402
from utils.config import AGENTIC_MODEL_TO_LLM  # noqa: E402

# Mirror experiments/dra_inference.py: load .env so resolve_agent_backend()
# derives the same `_api_` run-dir the run was written under (without it the
# backend falls back to `_vllm_` and the path mismatches).
load_dotenv()

logger = logging.getLogger("reasoning_error_analysis")


# ============================================================================
# CLI
# ============================================================================
def _parse_args():
    """Parse the run-selecting knobs; merge the file-backed config on top.

    Only the run-selecting knobs are real CLI arguments (same set as
    dra_inference). The mostly-fixed variables come from ``--config`` and can
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
    parser.add_argument("--dataset", type=str, default="trqa", choices=["trqa", "browsecomp_plus", "neuclir"], help="Dataset. The analysis needs TRQA intermediate-info gold, so only 'trqa' is supported.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset; trqa: wiki1|wiki2. Omit to take the value from --config (shipped default: wiki2).")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever used for the run (selects the run dir).")
    parser.add_argument("--controller", type=str, default="off", choices=["off", "monitor", "action"], help="Controller mode of the run (selects the run dir).")
    parser.add_argument("--controller-prompt-variant", type=str, default="nov_cov_sim", choices=["nov", "nov_cov", "nov_sim", "nov_cov_sim", "sim", "cov_sim"], help="Controller policy prompt variant of the run (only affects the dir name when --controller action).")

    # ── Run-control flags ──────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="analyse at most N queries")
    parser.add_argument("--qids", nargs="*", default=None, help="restrict to these query ids")
    parser.add_argument("--figures", choices=["none", "core", "all"], default="core", help="which figures to render: core = funnel + per-query ECDF; all = adds the per-error detail, aggregation-vs-coverage and joint plots.")
    parser.add_argument("-v", "--verbose", action="store_true")

    args, extras = parser.parse_known_args()

    # `subset` is the one knob that exists both as a real CLI flag and as a
    # config key, so apply_config_to_args() would overwrite it.  Its argparse
    # default is None precisely so "was it passed?" is answerable here and the
    # documented precedence (fallback < YAML < flag) actually holds.
    cli_subset = args.subset
    config = cfg.load_config(args.config)
    overrides = cfg.parse_cli_overrides(extras)
    cfg.apply_config_to_args(args, config, overrides)
    if cli_subset is not None:
        args.subset = cli_subset

    return args


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    # Derive run_dir / gold / queries / qrels / page_titles / out from the knobs.
    cfg.resolve_paths(args)

    gold = reu.load_gold(args)
    trajs = reu.load_trajectories(args, gold)

    rows: List[Dict[str, Any]] = []
    qrows: List[Dict[str, Any]] = []
    for qid, traj in trajs.items():
        entity_rows, query_row = reu.analyze_qid(
            qid, gold[qid], traj, args.agg_tol_pct, args.value_tol_rel
        )
        rows.extend(entity_rows)
        qrows.append(query_row)

    summary = reu.summarize(rows, qrows)
    print(reu.format_summary_table(summary))

    run_key = {
        "agentic_model": args.agentic_model,
        "subset": args.subset,
        "retriever": args.retriever,
        "controller": args.controller,
        "controller_prompt_variant": args.controller_prompt_variant,
    }
    out_dir = Path(args.out)
    reu.write_jsonl(rows, summary, run_key, out_dir / "reasoning_errors.jsonl")
    reu.write_query_jsonl(qrows, summary, run_key, out_dir / "reasoning_errors_by_query.jsonl")
    rep.make_figures(rows, qrows, summary, out_dir / "figures", args.figures)


if __name__ == "__main__":
    main()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
# python analysis/reasoning_error_analysis.py --agentic-model glm --subset wiki2
# python analysis/reasoning_error_analysis.py --subset wiki2 --figures all
# python analysis/reasoning_error_analysis.py --subset wiki2 --limit 20 -v
