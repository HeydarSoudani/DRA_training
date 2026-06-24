"""CLI argument resolution and pipeline-kwargs assembly.

Leaf-layer helpers that turn parsed argparse args into the dataset defaults,
worker config, and pipeline kwargs consumed by the orchestration layer. Depends
only on stdlib and ``utils`` — no heavy component imports.
"""

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from indexing_corpus_dataset.layout import corpus_name, trqa_corpus_name
from utils.config import _IR_ROOT

logger = logging.getLogger(__name__)


# ============================================================================
# File-backed run configuration
# ============================================================================
# The mostly-fixed pipeline variables live in a YAML config file
# (experiments/configs/default.yaml) instead of as argparse `add_argument`
# calls.  The frequently-varied knobs stay on the CLI; everything below is
# sourced from the file but can still be overridden by an explicit CLI flag.
#
# Precedence (lowest -> highest):
#   FILE_BACKED_DEFAULTS  <  YAML config file  <  explicit CLI flag

# Built-in fallbacks, used only when a key is absent from the YAML file so that
# downstream ``args.<attr>`` access never raises.  Keep in sync with the
# shipped experiments/configs/default.yaml.
FILE_BACKED_DEFAULTS = {
    # LLM
    "llm_temperature": 0.0,
    "llm_max_tokens_per_call": 10000,
    "llm_top_p": 1.0,
    "max_output_tokens_total": 20000,
    "request_timeout": 300,
    # Dataset (null -> auto-resolved from --dataset)
    "data_path": None,
    "subset": None,
    "dataset_year": None,
    "query_key": None,
    "qrels_data_path": None,
    "min_relevance_score": None,
    # Retriever
    "index_dir": None,
    "corpus_path": None,
    "top_k": 100,
    "seen_top_k": 5,
    # Searcher
    "post_retrieval_reranker": "null",
    "post_fusion_reranker": "null",
    "rerank_top_k": 100,
    "retrieval_input": "subquery",
    "post_fusion_reranker_input": "original_query",
    "ensure_novel_seen_docs": False,
    # Agent-specific
    "use_plan": False,
    "max_extend_steps": 5,
    "with_oracle_outline": False,
    "hard_mode": True,
    # Controller (the rest)
    "llm_controller": "claude-sonnet-4-6",
    "llm_intervene": "claude-sonnet-4-6",
    "controller_history_window": None,
    "criteria_coverage_mode": None,
    "criteria_coverage_max_criteria": 8,
    "llm_criteria_coverage": "claude-sonnet-4-6",
    # Evaluation
    "k_values": [1, 3, 5, 10, 25, 50, 75, 100, 500, 1000],
    "interleaving_window": 3,
    "rrf_k": 60,
    "fusion_k": 100,
    "fusion_methods": ["interleaving"],
    "judge_api_url": None,
    # Execution & output
    "output": None,
    "max_iteration": 100,
    "max_retries": 3,
    "verbose": True,
    "gpu_ids": None,
    "report_eval": False,
}

# Allowed values for the file-backed enum knobs (argparse ``choices=``
# equivalents) so a typo in the YAML or a CLI override fails fast.
_RERANKER_CHOICES = ["null", "batched_reranker", "rankllama", "rank1",
                     "qwen3_reranker", "listwise", "rank_r1", "monot5"]
_FILE_BACKED_CHOICES = {
    "post_retrieval_reranker": _RERANKER_CHOICES,
    "post_fusion_reranker": _RERANKER_CHOICES,
    "retrieval_input": ["subquery", "original_query+subquery", "reasoning+subquery"],
    "post_fusion_reranker_input": ["original_query", "original_query+subqueries",
                                   "original_query+reasoning", "reasoning+subqueries"],
    "criteria_coverage_mode": [None, "static", "dynamic"],
}


def _sm_bool(v):
    """SageMaker-compatible boolean coercion.

    Accepts real booleans and the string forms SageMaker forwards when
    hyperparameters are passed as ``--key value`` pairs.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "t"):
        return True
    if v.lower() in ("false", "0", "no", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def load_run_config(config_path) -> dict:
    """Load the YAML run-config of file-backed (mostly-fixed) variables.

    Returns a dict over the FILE_BACKED_DEFAULTS keys: built-in fallbacks
    overlaid with whatever the YAML file provides.  Unknown keys in the file
    are ignored with a warning.
    """
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
    _validate_choices(merged)
    return merged


def parse_cli_overrides(extras) -> dict:
    """Turn argparse ``parse_known_args`` leftovers into an overrides dict.

    Handles ``--key value``, ``--key=value``, repeated/space-separated list
    values (``--k-values 1 3 5``), and bare boolean flags (``--report-eval``).
    Dashes in flag names are normalised to underscores to match attr names.
    """
    overrides: dict = {}
    i = 0
    while i < len(extras):
        tok = extras[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if "=" in tok:
            key, val = tok[2:].split("=", 1)
            overrides[key.replace("-", "_")] = [val]
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        vals = []
        i += 1
        while i < len(extras) and not extras[i].startswith("--"):
            vals.append(extras[i])
            i += 1
        overrides[key] = vals  # empty list = bare flag
    return overrides


def _coerce_scalar(value_str, reference):
    """Coerce a CLI string to the type of its config default ``reference``."""
    if isinstance(reference, bool):
        return _sm_bool(value_str)
    if isinstance(reference, int):  # bool already handled above
        return int(value_str)
    if isinstance(reference, float):
        return float(value_str)
    return value_str


def _validate_choices(values: dict) -> None:
    """Validate file-backed enum knobs against their allowed values."""
    for key, allowed in _FILE_BACKED_CHOICES.items():
        if key in values and values[key] not in allowed:
            raise ValueError(
                f"Invalid value for '{key}': {values[key]!r}. Choose from: {allowed}"
            )


def apply_config_to_args(args, config: dict, overrides: dict) -> None:
    """Merge file-backed config and CLI overrides onto the ``args`` namespace.

    Sets every FILE_BACKED_DEFAULTS key on ``args`` from ``config``, then
    applies CLI ``overrides`` (coerced to each key's config type) so an
    explicit flag wins over the file for a single run.
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
            elem_ref = ref[0] if ref else ""
            coerced = [_coerce_scalar(tok, elem_ref) for tok in flat]
        elif not raw_vals:  # bare flag, e.g. --report-eval
            coerced = True
        else:
            coerced = _coerce_scalar(raw_vals[0], ref)
        setattr(args, key, coerced)

    _validate_choices(vars(args))


def resolve_dataset_defaults(args) -> None:
    """Auto-select dataset-related CLI arguments that were left as None."""
    if args.data_path is None:
        data_paths = {
            "trqa":            _IR_ROOT / "trqa",
            "neuclir":         _IR_ROOT / "neuclir",
            "browsecomp_plus": _IR_ROOT / "browsecomp_plus",
        }
        args.data_path = str(data_paths[args.dataset])
        print(f"Auto-selected data path: {args.data_path}")

    if args.dataset_year is None and args.dataset == "neuclir":
        args.dataset_year = "2024"
        print(f"Auto-selected dataset year: {args.dataset_year}")

    # TRQA carries the evaluation split (test/validation) in dataset_year.
    if args.dataset_year is None and args.dataset == "trqa":
        args.dataset_year = "test"
        print(f"Auto-selected eval split: {args.dataset_year}")

    if args.subset is None:
        if args.dataset == "neuclir":
            args.subset = "news"
            print(f"Auto-selected subset: {args.subset}")
        elif args.dataset == "trqa":
            args.subset = "wiki1"
            print(f"Auto-selected subset: {args.subset}")
        elif args.dataset == "browsecomp_plus":
            args.subset = "test"
            print(f"Auto-selected subset: {args.subset}")

    if args.min_relevance_score is None and args.dataset == "neuclir":
        args.min_relevance_score = 3
    if args.min_relevance_score is None and args.dataset == "trqa":
        args.min_relevance_score = 1

    if args.query_key is None:
        # Downloaders flatten each dataset's best query text into the "text"
        # field (NeuCLIR's topic_* fields are collapsed at download time), so
        # "text" is the correct key for every dataset.
        args.query_key = "text"
        print(f"Auto-selected query key: {args.query_key}")

    if args.dataset in {"trqa", "neuclir", "browsecomp_plus"}:
        if args.index_dir is None:
            args.index_dir = str(_IR_ROOT / args.dataset / "indices")
            print(f"Auto-selected index directory: {args.index_dir}")
        if args.corpus_path is None:
            if args.dataset == "neuclir":
                _corpus_file = f"{corpus_name(args.subset)}.jsonl"
                args.corpus_path = str(_IR_ROOT / "neuclir" / "corpus" / _corpus_file)
            elif args.dataset == "trqa":
                _corpus_file = f"{trqa_corpus_name(args.subset)}.jsonl"
                args.corpus_path = str(_IR_ROOT / "trqa" / "corpus" / _corpus_file)
            else:  # browsecomp_plus
                args.corpus_path = str(_IR_ROOT / "browsecomp_plus" / "corpus" / "corpus.jsonl")
            print(f"Auto-selected corpus path: {args.corpus_path}")

    if getattr(args, "qrels_data_path", None) is None:
        args.qrels_data_path = args.data_path


def detect_num_gpus(args_num_gpus: int) -> int:
    """Resolve the number of GPU workers."""
    if args_num_gpus != 0:
        return args_num_gpus
    try:
        import torch
        detected = torch.cuda.device_count()
        num_gpus = max(1, detected)
        if detected > 1:
            print(f"Auto-detected {detected} GPUs — enabling parallel mode ({num_gpus} workers)")
        else:
            print(f"Auto-detected {detected} GPU(s) — running single-GPU / sequential mode")
        return num_gpus
    except ImportError:
        return 1


def assemble_pipeline_kwargs(args, llm_client, retriever, num_gpus: int, verbose: bool, gpu_ids=None) -> dict:
    """Build the kwargs dict to pass to run_pipeline() (includes worker_config)."""
    pipeline_kwargs = {
        "retriever":                  retriever,
        "llm_client":                 llm_client,
        "llm_model":                  args.llm_model,
        "top_k":                      args.top_k,
        "seen_top_k":                 args.seen_top_k,
        "dataset":                    args.dataset,
        "min_relevance_score":        args.min_relevance_score,
        "k_values":                   args.k_values,
        "interleaving_window":        args.interleaving_window,
        "rrf_k":                      args.rrf_k,
        "max_iteration":              args.max_iteration,
        "max_retries":                args.max_retries,
    }

    if args.dataset in ["trqa", "neuclir", "browsecomp_plus"]:
        pipeline_kwargs["retriever_name"] = args.retriever

    pipeline_kwargs["qrels_data_path"] = getattr(args, "qrels_data_path", None)
    pipeline_kwargs["temperature"] = args.llm_temperature
    pipeline_kwargs["max_output_tokens_total"] = args.max_output_tokens_total
    pipeline_kwargs["use_plan"] = getattr(args, "use_plan", False)
    pipeline_kwargs["max_extend_steps"] = args.max_extend_steps
    pipeline_kwargs["hard_mode"] = args.hard_mode
    pipeline_kwargs["oracle_outline_path"] = getattr(args, "with_oracle_outline", None)

    pipeline_kwargs["controller"] = getattr(args, "controller", "monitor")
    pipeline_kwargs["llm_intervene"] = getattr(args, "llm_intervene", None)
    pipeline_kwargs["llm_controller"] = getattr(args, "llm_controller", None)
    pipeline_kwargs["controller_history_window"] = getattr(args, "controller_history_window", None)
    pipeline_kwargs["controller_prompt_variant"] = getattr(args, "controller_prompt_variant", "nov_cov_sim")
    pipeline_kwargs["criteria_coverage_mode"] = getattr(args, "criteria_coverage_mode", "dynamic")
    pipeline_kwargs["criteria_coverage_max_criteria"] = getattr(args, "criteria_coverage_max_criteria", 8)
    pipeline_kwargs["llm_criteria_coverage"] = getattr(args, "llm_criteria_coverage", None)

    worker_config = {
        "agentic_model":           args.agentic_model,
        "llm_model":               args.llm_model,
        "llm_temperature":         args.llm_temperature,
        "llm_max_tokens_per_call":          args.llm_max_tokens_per_call,
        "llm_top_p":               args.llm_top_p,
        "request_timeout":         args.request_timeout,
        "dataset":                 args.dataset,
        "retriever_type":          args.retriever,
        "index_dir":               getattr(args, "index_dir", None),
        "corpus_path":             getattr(args, "corpus_path", None),
        "top_k":                   args.top_k,
        "seen_top_k":              args.seen_top_k,
        "max_iteration":           args.max_iteration,
        "max_extend_steps":        args.max_extend_steps,
        "max_retries":             args.max_retries,
        "hard_mode":               args.hard_mode,
        "temperature":             args.llm_temperature,
        "verbose":                 verbose,
        "gpu_ids":                 gpu_ids if gpu_ids is not None else list(range(num_gpus)),
        "post_retrieval_reranker_type": args.post_retrieval_reranker,
        "post_fusion_reranker_type":    args.post_fusion_reranker,
        "rerank_top_k":                 args.rerank_top_k,
        "retrieval_input":              args.retrieval_input,
        "post_fusion_reranker_input":   args.post_fusion_reranker_input,
        "max_output_tokens_total":       getattr(args, "max_output_tokens_total", 40000),
        "use_plan":                      getattr(args, "use_plan", False),
        "oracle_outline_path":           getattr(args, "with_oracle_outline", None),
        "controller":                    getattr(args, "controller", "monitor"),
        "llm_intervene":        getattr(args, "llm_intervene", None),
        "llm_controller":           getattr(args, "llm_controller", None),
        "controller_history_window": getattr(args, "controller_history_window", None),
        "controller_prompt_variant":             getattr(args, "controller_prompt_variant", "nov_cov_sim"),
        "criteria_coverage_mode":          getattr(args, "criteria_coverage_mode", "dynamic"),
        "criteria_coverage_max_criteria":   getattr(args, "criteria_coverage_max_criteria", 8),
        "llm_criteria_coverage":           getattr(args, "llm_criteria_coverage", None),
        "ensure_novel_seen_docs":        getattr(args, "ensure_novel_seen_docs", False),
        "quiet":                         getattr(args, "quiet", False),
    }

    pipeline_kwargs["num_gpus"]      = num_gpus
    pipeline_kwargs["worker_config"] = worker_config
    return pipeline_kwargs


def cleanup_event_loop() -> None:
    """Cancel any pending async tasks and close the current event loop."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            if not loop.is_closed():
                loop.close()
    except Exception:
        pass
