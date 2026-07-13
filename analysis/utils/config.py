"""CLI config + path derivation for the reasoning-error analysis.

Mirrors ``experiments/dra_inference.py``: a few frequently-varied knobs stay on
the CLI (``--agentic-model``, ``--dataset``, ``--retriever``, ``--controller``,
``--controller-prompt-variant``), while the mostly-fixed variables live in a YAML
file (``experiments/configs/dra_analysis.yaml``) and are merged onto ``args``.

Crucially, the run directory, the gold intermediate-info file and the corpus are
then **derived** from those knobs exactly the way the inference pipeline lays
them out, so you never hand-type a run path. Any derived value can still be
overridden by putting an explicit path in the YAML or passing the matching
``--flag`` on the CLI.

Precedence (lowest -> highest):  built-in fallback  <  YAML file  <  CLI flag.
"""

import logging
import os
from pathlib import Path

import yaml

# Generic, pipeline-agnostic override helpers reused from the inference layer.
from utils.cli_setup import parse_cli_overrides, _coerce_scalar
from utils.config import _IR_ROOT, AGENTIC_MODEL_TO_LLM, AGENTIC_MODEL_ALIAS
from utils.io_utils import build_run_name_for_pipeline, build_controller_config_name
from indexing_corpus_dataset.dataset_loaders import resolve_split_id
from indexing_corpus_dataset.layout import trqa_corpus_name

logger = logging.getLogger("reasoning_error_analysis")

_OUTPUT_PREFIX = os.environ.get(
    "DRA_OUTPUT_ROOT", "/projects/0/prjs0834/heydars/DRA_training/run_outputs"
)
# analysis/utils/config.py -> repo_root/experiments/configs/dra_analysis.yaml
CONFIG_DEFAULT = str(
    Path(__file__).resolve().parents[2] / "experiments" / "configs" / "dra_analysis.yaml"
)


# ---------------------------------------------------------------------------
# File-backed defaults (the mostly-fixed knobs; keep in sync with the shipped
# experiments/configs/dra_analysis.yaml).  ``None`` means "auto-derive".
# ---------------------------------------------------------------------------
FILE_BACKED_DEFAULTS = {
    # Dataset resolution (null -> auto-selected from --dataset)
    "subset": None,
    "dataset_year": None,
    "query_key": None,
    # Roots (null -> built-in: _IR_ROOT for data, DRA_OUTPUT_ROOT for runs)
    "data_root": None,
    "output": None,
    # Run-dir derivation — must match how the run was produced so the path resolves
    "use_plan": False,
    "ensure_novel_seen_docs": False,
    "llm_controller": "claude-sonnet-4-6",
    # Explicit path overrides (null -> derive from the knobs above)
    "run_dir": None,
    "gold": None,
    "corpus": None,
    "qrels": None,
    "out": None,
    # Numeric-matching tolerances
    "rel_tol": 1e-3,
    "abs_tol": 1e-6,
    "gold_tol_abs": 0.5,
    "gold_tol_rel": 1e-2,
    # LLM judge (only used with --judge)
    "judge_model": "openrouter/qwen/qwen3-32b",
}


def load_config(config_path) -> dict:
    """Load the YAML run-config, overlaid on the built-in fallbacks."""
    merged = dict(FILE_BACKED_DEFAULTS)
    if not config_path:
        return merged
    path = Path(config_path)
    if not path.exists():
        print(f"WARNING: config file not found: {config_path} — using built-in defaults")
        return merged
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - set(FILE_BACKED_DEFAULTS)
    if unknown:
        logger.warning("Ignoring unknown keys in %s: %s", config_path, sorted(unknown))
    merged.update({k: v for k, v in data.items() if k in FILE_BACKED_DEFAULTS})
    return merged


def apply_config_to_args(args, config: dict, overrides: dict) -> None:
    """Set every config key on ``args``, then apply CLI overrides on top.

    An override is coerced to the type of its config default, so ``--rel-tol
    1e-4`` lands as a float and ``--use-plan true`` as a bool.
    """
    for key, value in config.items():
        setattr(args, key, value)

    for key, raw_vals in overrides.items():
        if key not in config:
            print(f"WARNING: unknown override --{key.replace('_', '-')} ignored")
            continue
        ref = config[key]
        if isinstance(ref, list):
            flat = [tok for v in raw_vals for tok in v.split()]
            coerced = [_coerce_scalar(tok, ref[0] if ref else "") for tok in flat]
        elif not raw_vals:  # bare flag, e.g. --use-plan
            coerced = True
        else:
            coerced = _coerce_scalar(raw_vals[0], ref)
        setattr(args, key, coerced)


def resolve_paths(args) -> None:
    """Derive run_dir / gold / corpus / out from the CLI knobs (TRQA layout).

    Reproduces the directory names the inference pipeline builds:
        {output}/{dataset}_{split}_{retriever}/{agent}_{backend}_{model}/{ctrl}/
    and the canonical data layout under ``data_root``.
    """
    if args.dataset != "trqa":
        raise SystemExit(
            "reasoning-error analysis needs TRQA intermediate-info gold; "
            f"--dataset={args.dataset!r} is not supported."
        )

    data_root = Path(args.data_root) if args.data_root else _IR_ROOT
    output_root = args.output or _OUTPUT_PREFIX

    # Dataset defaults (mirror cli_setup.resolve_dataset_defaults for trqa).
    if args.dataset_year is None:
        args.dataset_year = "test"
    if args.subset is None:
        args.subset = "wiki1"
    if args.query_key is None:
        args.query_key = "text"

    file_data_set = resolve_split_id(args.dataset, args.dataset_year, args.subset)

    # Reproduce the agent/controller directory names exactly like inference.
    llm_model = AGENTIC_MODEL_TO_LLM[args.agentic_model]
    agentic_model = AGENTIC_MODEL_ALIAS.get(args.agentic_model, args.agentic_model)
    run_name = build_run_name_for_pipeline(
        agentic_model=agentic_model, llm_model=llm_model, use_plan=args.use_plan
    )
    controller_config_name = build_controller_config_name(
        controller=args.controller,
        llm_controller=args.llm_controller,
        llm_intervene=args.llm_controller,
        controller_prompt_variant=args.controller_prompt_variant,
        ensure_novel_seen_docs=args.ensure_novel_seen_docs,
    )
    qk_part = f"_{args.query_key}" if args.query_key and args.query_key != "text" else ""
    dataset_dir = f"{args.dataset}_{file_data_set}{qk_part}_{args.retriever}"

    if args.run_dir is None:
        args.run_dir = str(Path(output_root) / dataset_dir / run_name / controller_config_name)
        print(f"Auto-derived run dir: {args.run_dir}")
    if args.gold is None:
        args.gold = str(
            data_root / "trqa" / "queries" / f"queries_{file_data_set}_intermediate_info.jsonl"
        )
        print(f"Auto-derived gold: {args.gold}")
    if args.corpus is None:
        cand = data_root / "trqa" / "corpus" / f"{trqa_corpus_name(args.subset)}.jsonl"
        if cand.exists():
            args.corpus = str(cand)
            print(f"Auto-derived corpus: {args.corpus}")
        else:
            print(f"WARNING: corpus not found at {cand} — per-entity retrieval (E_ret) unassessed")
    if args.out is None:
        args.out = str(Path(args.run_dir) / "analysis")
        print(f"Auto-derived out: {args.out}")
