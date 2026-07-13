"""Train the DRA search agent (SFT cold-start + async GRPO).

This is the single INPUT to the training pipeline — the sibling of
``experiments/dra_inference.py``.  All logic lives in ``src/training`` (consumed
read-only alongside the finalized inference components); this file only parses
args, merges the YAML config, and dispatches to ``training.pipeline``.

Design (see MEMORY: agent-sft-rl-training-plan):
  * INFRASTRUCTURE is implemented; DESIGNABLE parts (reward, SFT data-gen, the
    GRPO/FSDP update, weight-sync transport) are TODO stubs behind interfaces.
  * ``--smoke`` runs the ENTIRE loop on CPU with mock policy/tool/trainer — no
    GPU, server, dataset, or index required.

Examples
--------
  # CPU smoke test (mock everything), sync loop:
  python experiments/dra_train.py --smoke
  # CPU smoke test, async loop:
  python experiments/dra_train.py --smoke --mode async
  # Real-ish RL once backends are wired (dataset + server + grpo backend):
  python experiments/dra_train.py --stage rl --mode async \
      --config experiments/configs/dra_train.yaml
"""

import argparse
import sys
from pathlib import Path

# ── Ensure src/ is importable even without scripts/_activate.sh (defensive) ──
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging

from training.config import load_config
from training import pipeline

_CONFIG_DEFAULT = str(Path(__file__).resolve().parent / "configs" / "dra_train.yaml")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Train the DRA search agent (SFT + async GRPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="YAML config with mostly-fixed knobs; CLI flags override it.")
    p.add_argument("--stage", type=str, default=None, choices=["rl", "sft"], help="Training stage (overrides config.stage).")
    p.add_argument("--mode", type=str, default=None, choices=["sync", "async"], help="RL loop mode (overrides config.rl.mode).")
    p.add_argument("--trainer-backend", type=str, default=None, choices=["mock", "grpo"], help="Trainer backend (overrides config.trainer.backend).")
    p.add_argument("--policy", type=str, default=None, choices=["mock", "server"], help="Policy client (overrides config.rollout.policy).")
    p.add_argument("--tool", type=str, default=None, choices=["mock", "retrieval"], help="Tool env (overrides config.rollout.tool).")
    p.add_argument("--total-steps", type=int, default=None, help="Trainer updates (overrides config.rl.total_steps).")
    p.add_argument("--group-size", type=int, default=None, help="Rollouts per prompt (overrides config.rollout.group_size).")
    p.add_argument("--limit", type=int, default=None, help="Cap number of prompts.")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test: force mock policy/tool/trainer + synthetic data.")
    p.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return p.parse_args()


def _overrides_from_args(args) -> dict:
    """Translate CLI flags into a nested-dict override for load_config."""
    ov: dict = {}

    def setpath(section, key, val):
        if val is not None:
            ov.setdefault(section, {})[key] = val

    if args.stage is not None:
        ov["stage"] = args.stage
    setpath("rl", "mode", args.mode)
    setpath("rl", "total_steps", args.total_steps)
    setpath("trainer", "backend", args.trainer_backend)
    setpath("rollout", "policy", args.policy)
    setpath("rollout", "tool", args.tool)
    setpath("rollout", "group_size", args.group_size)
    setpath("data", "limit", args.limit)

    if args.smoke:
        ov.setdefault("rollout", {}).update({"policy": "mock", "tool": "mock"})
        ov.setdefault("trainer", {})["backend"] = "mock"
        ov.setdefault("data", {})["source"] = "synthetic"
    return ov


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config, overrides=_overrides_from_args(args))
    print(f"[dra_train] stage={cfg.stage} rl.mode={cfg.rl.mode} "
          f"trainer={cfg.trainer.backend} policy={cfg.rollout.policy} tool={cfg.rollout.tool}")
    pipeline.run(cfg)


if __name__ == "__main__":
    main()
