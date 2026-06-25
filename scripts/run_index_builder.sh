#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --mem=240GB
#SBATCH --output=script_logging/slurm_%A.out
# Sizing notes (measured on H100, qwen3_emb_4b, browsecomp_plus, max_length=4096):
#   - Encoding is compute-bound at ~5.6 docs/s/GPU; throughput is FLAT from
#     batch_size 16..64, so 16 (lowest memory, peak ~23GB) is optimal. 96 OOMs.
#   - SBU ~= node-fraction x walltime is ~identical for 1/2/4 GPUs (startup is
#     negligible vs ~294 GPU-min encode), so 4 GPUs = same cost, ~75 min wall.
#   - With --gpus=4 you take the whole node; request all 64 cores. mem<=480GB
#     keeps the header valid on gpu_a100 too (cores there: 72).
#   - SBU is billed on ACTUAL runtime, not --time, so 02:00:00 only affects
#     scheduling, not cost.

# torch / faiss live in the Anaconda3/2024.06 base env (auto-activated via conda
# in ~/.bashrc for interactive shells, but sbatch runs non-interactively so we
# activate it explicitly). The 2025 Python/3.13.5 module has no torch.
source /sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate base
mkdir -p script_logging

# Stream print() output to the slurm log in real time (stdout is block-buffered
# to files otherwise, so the config banner would appear after the progress bars).
export PYTHONUNBUFFERED=1

export HF_DATASETS_CACHE=/projects/0/prjs0834/heydars/.cache/huggingface
export HF_HOME=/projects/0/prjs0834/heydars/.cache/huggingface

# Read/write dataset root: corpus is read from and indices are written under here.
# Set explicitly so the job is independent of the submitting shell's environment.
export DRA_DATA_ROOT=/projects/0/prjs0834/heydars/DRA_training/data

RETRIEVER=qwen3_emb_4b        # bm25 | spladepp | bge | qwen3_emb_4b
DATASET=browsecomp_plus       # trqa | neuclir | browsecomp_plus

case "$RETRIEVER" in
    bm25)              ARGS=() ;;
    spladepp|spladev3) ARGS=(--use_fp16 --max_length 256 --batch_size 512 --save_embedding) ;;
    # qwen3 embedding models: 4B params; browsecomp_plus auto-raises max_length
    # to 4096. Measured: throughput is flat 16..64, so batch 16 (peak ~23GB) is
    # optimal — larger batches only add memory (96 OOMs at ~98GB), no speedup.
    qwen3_emb_*)       ARGS=(--use_fp16 --max_length 4096 --batch_size 16 --faiss_type Flat --save_embedding) ;;
    *)                 ARGS=(--use_fp16 --max_length 512 --batch_size 512 --faiss_type Flat --save_embedding) ;;
esac

python -m indexing_corpus_dataset.index_builder \
    --retriever "$RETRIEVER" \
    --dataset "$DATASET" \
    "${ARGS[@]}"
