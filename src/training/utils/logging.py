"""Lightweight stats logging for the training loops.

Console by default; W&B is optional and guarded so nothing here pulls a heavy
dependency into the CPU smoke path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("training")

_WANDB = None


def init_wandb(project: str, name: str, config: Dict[str, Any]) -> None:  # pragma: no cover
    global _WANDB
    try:
        import wandb  # noqa
        _WANDB = wandb
        wandb.init(project=project, name=name, config=config)
    except Exception:
        logger.warning("wandb unavailable — console logging only")
        _WANDB = None


def log_stats(prefix: str, stats: Dict[str, Any]) -> None:
    kv = "  ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in stats.items()
    )
    logger.info("%s %s", prefix, kv)
    print(f"{prefix} {kv}")
    if _WANDB is not None:  # pragma: no cover
        _WANDB.log(stats)
