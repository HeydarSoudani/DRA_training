"""Training configuration — dataclasses + YAML loader.

Mirrors the inference convention (a YAML file holding mostly-fixed knobs, with
CLI overrides merged on top) but is self-contained: it does NOT import
``utils.cli_setup`` so the training path stays decoupled from inference-specific
argument logic.

Everything has a default so a bare ``TrainingConfig()`` is a valid CPU smoke-test
config (all components default to ``mock``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Section configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    source: str = "synthetic"                 # "synthetic" (CPU test) | "dataset" (indexing_corpus_dataset)
    dataset: str = "trqa"                     # used when source == "dataset"
    dataset_year: Optional[str] = None
    subset: Optional[str] = None
    query_key: str = "text"
    data_path: Optional[str] = None
    qrels_data_path: Optional[str] = None
    min_relevance_score: Optional[int] = None
    limit: Optional[int] = None
    num_synthetic: int = 8                    # size of the synthetic dataset


@dataclass
class RolloutConfig:
    policy: str = "mock"                      # "mock" | "server"
    tool: str = "mock"                        # "mock" | "retrieval"
    server_url: Optional[str] = None          # policy server (SGLang/vLLM) base url, when policy == "server"
    model: Optional[str] = None               # served model name
    max_turns: int = 5
    max_gen_tokens: int = 512
    temperature: float = 1.0
    top_k_docs: int = 5                       # docs surfaced per search turn
    group_size: int = 4                       # rollouts per prompt (GRPO group)
    concurrency: int = 8                      # max concurrent rollouts (async)


@dataclass
class RewardConfig:
    outcome_metric: str = "f1"                # TODO(reward): choose/define
    lambda_process: float = 0.2               # weight on process reward (keep outcome dominant)
    clamp_process: bool = True                # bound process shaping so it can't dominate outcome
    signals: tuple = ("doc_novelty", "marginal_recall")  # TODO(reward): which controller signals to use


@dataclass
class BufferConfig:
    staleness_k: int = 1                      # max weight-version lag tolerated (async)
    capacity: int = 4096


@dataclass
class TrainerConfig:
    backend: str = "mock"                     # "mock" | "grpo" (AReaL/veRL adapter, TODO)
    lr: float = 1e-6
    kl_coef: float = 0.001
    save_dir: Optional[str] = None
    save_every: int = 50
    # backend-specific extras (areal/verl config path, parallelism, ...) go here
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTConfig:
    backend: str = "mock"                     # "mock" | "llama_factory" | "trl" | "verl_fsdp" (TODO)
    data_out: Optional[str] = None            # where sft_dataset writes formatted trajectories
    base_model: Optional[str] = None
    epochs: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    mode: str = "sync"                        # "sync" (validate first) | "async" (production)
    total_steps: int = 4                      # trainer updates
    prompts_per_step: int = 2                 # prompts sampled per sync step
    seed: int = 0


@dataclass
class TrainingConfig:
    stage: str = "rl"                         # "rl" | "sft"
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    output_dir: Optional[str] = None
    verbose: bool = True


# ---------------------------------------------------------------------------
# (de)serialization + merge
# ---------------------------------------------------------------------------

def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Uses ``get_type_hints`` so ``from __future__ import annotations`` (which turns
    field types into strings) still resolves nested dataclass fields.
    """
    if not is_dataclass(cls):
        return data
    from typing import get_type_hints
    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = type_hints.get(f.name)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """Load a TrainingConfig from YAML (optional) with dict overrides merged on top."""
    data: Dict[str, Any] = {}
    if path:
        import yaml  # local import: not needed for the pure-Python smoke path
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        _deep_update(data, overrides)
    return _from_dict(TrainingConfig, data)
