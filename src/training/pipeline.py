"""Top-level orchestrators: build components from config and run SFT / RL.

Called by ``experiments/dra_train.py``.  All the wiring (which policy / tool /
reward / trainer to instantiate) lives here so the entrypoint stays thin.

Everything defaults to ``mock`` so ``run_rl(TrainingConfig())`` is a full CPU
smoke test with no GPUs, server, or dataset.
"""

from __future__ import annotations

import asyncio
import logging

from .config import TrainingConfig
from .data.prompt_dataset import load_prompts
from .rollout.policy_client import MockPolicyClient, ServerPolicyClient, PolicyClient
from .rollout.tool_env import MockToolEnv, RetrievalToolEnv, ToolEnv
from .rollout.sampler import Sampler
from .reward.base import CompositeReward, RewardModel
from .reward.outcome_reward import OutcomeReward, MockOutcomeReward
from .reward.process_reward import ProcessReward, MockProcessReward
from .trainer.base import TrainerBackend
from .trainer.mock_trainer import MockTrainer
from .trainer.weight_sync import LocalVersionSync
from .buffer.trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component builders (each honors the "mock" default)
# ---------------------------------------------------------------------------

def _build_policy(cfg: TrainingConfig) -> PolicyClient:
    if cfg.rollout.policy == "mock":
        return MockPolicyClient(search_turns=min(2, cfg.rollout.max_turns - 1))
    if cfg.rollout.policy == "server":
        if not cfg.rollout.server_url or not cfg.rollout.model:
            raise ValueError("rollout.server_url and rollout.model required for policy='server'")
        return ServerPolicyClient(cfg.rollout.server_url, cfg.rollout.model)
    raise ValueError(f"unknown rollout.policy '{cfg.rollout.policy}'")


def _build_tool(cfg: TrainingConfig) -> ToolEnv:
    if cfg.rollout.tool == "mock":
        return MockToolEnv(top_k=cfg.rollout.top_k_docs)
    if cfg.rollout.tool == "retrieval":
        # READ-ONLY reuse of the finalized searcher. Built exactly as inference does.
        from searcher_component.searcher import RetrievalSearchTool  # noqa
        from orchestration import setup_retriever_from_args  # noqa
        raise NotImplementedError(
            "tool='retrieval' wiring: construct RetrievalSearchTool from the "
            "retriever (as dra_inference does) and wrap in RetrievalToolEnv. "
            "Left as a thin TODO so no GPU/index is needed for CPU validation."
        )
    raise ValueError(f"unknown rollout.tool '{cfg.rollout.tool}'")


def _build_reward(cfg: TrainingConfig, mock: bool) -> RewardModel:
    if mock:
        outcome = MockOutcomeReward(cfg.reward)
        process = MockProcessReward(cfg.reward)
    else:
        outcome = OutcomeReward(cfg.reward)      # TODO(reward): real metric
        process = ProcessReward(cfg.reward)      # TODO(reward): real signals
    return CompositeReward(outcome, process, cfg.reward)


def _build_trainer(cfg: TrainingConfig) -> TrainerBackend:
    if cfg.trainer.backend == "mock":
        return MockTrainer(save_dir=cfg.trainer.save_dir)
    if cfg.trainer.backend == "grpo":
        from .trainer.grpo_backend import GRPOBackend
        return GRPOBackend(cfg.trainer)
    raise ValueError(f"unknown trainer.backend '{cfg.trainer.backend}'")


# ---------------------------------------------------------------------------
# Stage entry points
# ---------------------------------------------------------------------------

def run_rl(cfg: TrainingConfig) -> None:
    """Run the RL stage (sync or async) with the configured components."""
    prompts = load_prompts(cfg.data)
    if not prompts:
        raise RuntimeError("no prompts loaded")

    mock_components = (cfg.rollout.policy == "mock" and cfg.rollout.tool == "mock"
                       and cfg.trainer.backend == "mock")
    policy = _build_policy(cfg)
    tool = _build_tool(cfg)
    reward = _build_reward(cfg, mock=mock_components)
    trainer = _build_trainer(cfg)
    sampler = Sampler(policy, tool, cfg.rollout)

    print(f"[dra_train] RL stage | mode={cfg.rl.mode} | prompts={len(prompts)} "
          f"| policy={cfg.rollout.policy} tool={cfg.rollout.tool} trainer={cfg.trainer.backend}")

    if cfg.rl.mode == "sync":
        from .loop.sync_rl_loop import run_sync_rl
        asyncio.run(run_sync_rl(cfg, prompts, sampler, reward, trainer))
    elif cfg.rl.mode == "async":
        from .loop.async_rl_loop import run_async_rl
        buffer = TrajectoryBuffer(
            group_size=cfg.rollout.group_size,
            staleness_k=cfg.buffer.staleness_k,
            capacity=cfg.buffer.capacity,
        )
        weight_sync = LocalVersionSync()  # TODO(weight-sync): BroadcastWeightSync for real backend
        asyncio.run(run_async_rl(cfg, prompts, sampler, reward, trainer, buffer, weight_sync))
    else:
        raise ValueError(f"unknown rl.mode '{cfg.rl.mode}'")

    print("[dra_train] RL stage complete")


def run_sft(cfg: TrainingConfig) -> None:
    """Run the SFT cold-start stage. TODO(sft): data-gen + backend are designable."""
    prompts = load_prompts(cfg.data)
    from .data.sft_dataset import build_sft_dataset
    from .trainer.sft_backend import run_sft_backend
    data_path = build_sft_dataset(cfg.sft, prompts)   # raises NotImplementedError (stub)
    run_sft_backend(cfg.sft, data_path)


def run(cfg: TrainingConfig) -> None:
    if cfg.stage == "rl":
        run_rl(cfg)
    elif cfg.stage == "sft":
        run_sft(cfg)
    else:
        raise ValueError(f"unknown stage '{cfg.stage}'")
