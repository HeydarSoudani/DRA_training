#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --mem=240GB
#SBATCH --output=script_logging/slurm_%A.out
# vLLM manager auto-splits GPUs: model server takes the leftmost GPU(s),
# the rest become pipeline workers. SBU is billed on actual runtime, not --time.

# sbatch runs non-interactively; scripts/_activate.sh loads the Python/3.13.5 module
# stack, activates the project venv (torch/faiss/vllm), and exports the project
# env vars (HF_HOME, DRA_DATA_ROOT, DRA_OUTPUT_ROOT, ...) — same env as interactive.
# Sourced CWD-relative: sbatch preserves the submission dir (the repo root), which
# is also why the python/log paths below are relative.
source scripts/_activate.sh
mkdir -p script_logging

# DATASET + RETRIEVER must match the built index (see scripts/run_index_builder.sh).
DATASET=browsecomp_plus            # trqa | neuclir | browsecomp_plus
RETRIEVER=qwen3_emb_4b
AGENT=glm                          # glm | oss_20b | oss_120b | tongyi | react | cpm_report | ...
CONTROLLER=action                  # off | monitor | action
CONTROLLER_PROMPT_VARIANT=nov_cov_sim

python experiments/dra_inference.py \
    --dataset "$DATASET" \
    --retriever "$RETRIEVER" \
    --agentic-model "$AGENT" \
    --controller "$CONTROLLER" \
    --controller-prompt-variant "$CONTROLLER_PROMPT_VARIANT" \
    --num-gpus 0

# Smoke test: python experiments/dra_inference.py --dataset browsecomp_plus --limit 1 --num-gpus 0
# Eval only:  python experiments/dra_inference.py --dataset browsecomp_plus --eval-only --num-gpus 0
