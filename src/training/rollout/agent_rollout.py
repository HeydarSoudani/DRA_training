"""Async multi-turn agent rollout.

Reimplements a minimal think->search->observe->answer loop (mirroring the
inference agents' protocol) but records TOKEN-LEVEL data (ids + loss mask +
logprobs) needed for RL.  It deliberately does NOT reuse ``deep_research_agents``
(those are API/text-only) so the finalized inference agents stay untouched.

Reuses READ-ONLY: the tool env (wrapping ``searcher_component``).
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import RolloutConfig
from ..data.schema import PromptRecord, Trajectory, Turn
from .policy_client import PolicyClient
from .tool_env import ToolEnv
from .protocol import parse_action
from .masking import make_observation_text, validate_trajectory

logger = logging.getLogger(__name__)


def _build_prompt(prompt: PromptRecord, history: str, turn_idx: int) -> str:
    """Assemble the context fed to the policy for the next step.

    The trailing ``[[turn=N]]`` marker is consumed by MockPolicyClient; a real
    ServerPolicyClient ignores it (it's inside the rendered context but harmless).
    TODO(prompting): replace with the finalized chat template once the policy
    server + tokenizer are wired.
    """
    return (
        f"Question: {prompt.question}\n"
        f"{history}"
        f"[[turn={turn_idx}]]"
    )


async def rollout_once(
    prompt: PromptRecord,
    policy: PolicyClient,
    tool: ToolEnv,
    cfg: RolloutConfig,
    *,
    group_id: str,
) -> Trajectory:
    """Run a single trajectory for ``prompt`` and return it (reward unset)."""
    tool.reset()
    traj = Trajectory(prompt_id=prompt.id, group_id=group_id, policy_version=policy.version)
    history = ""

    for turn_idx in range(cfg.max_turns):
        prompt_text = _build_prompt(prompt, history, turn_idx)
        gen = await policy.generate(
            prompt_text, max_tokens=cfg.max_gen_tokens, temperature=cfg.temperature
        )
        action = parse_action(gen.text)

        turn = Turn(
            index=turn_idx,
            text=gen.text,
            action_kind=action.kind,
            query=action.query,
            gen_token_ids=gen.token_ids,
            gen_logprobs=gen.logprobs,
        )

        if action.kind == "answer":
            traj.final_answer = action.answer or ""
            traj.stop_reason = "answer"
            traj.turns.append(turn)
            break

        if action.kind == "stop":
            traj.stop_reason = "error"          # malformed generation
            traj.turns.append(turn)
            break

        # --- search action: call the tool, inject a masked observation ---
        docs = tool.search(action.query, original_query=prompt.question, top_k=cfg.top_k_docs)
        obs_text = make_observation_text(docs, max_docs=cfg.top_k_docs)
        turn.obs_text = obs_text
        turn.obs_doc_ids = [d.get("doc_id") or d.get("id") or "" for d in docs]
        # Observation tokens are masked. TODO(policy-server): tokenize obs with the
        # SAME tokenizer as the policy; the mock uses the toy tokenizer for parity.
        from .policy_client import _toy_tokenize
        turn.obs_token_ids = _toy_tokenize(obs_text)

        traj.turns.append(turn)
        history += f"{gen.text}\n{obs_text}\n"
    else:
        traj.stop_reason = "max_turns"

    validate_trajectory(traj)
    return traj
