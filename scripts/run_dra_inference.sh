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

# sbatch runs non-interactively, so activate the base env (with torch/faiss) explicitly.
source /sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate base
mkdir -p script_logging

export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE=/projects/0/prjs0834/heydars/.cache/huggingface
export HF_HOME=/projects/0/prjs0834/heydars/.cache/huggingface
export DRA_DATA_ROOT=/projects/0/prjs0834/heydars/DRA_training/data
export DRA_OUTPUT_ROOT=/home/hsoudani/DRA_training/run_outputs

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
