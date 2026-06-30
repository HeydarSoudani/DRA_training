#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=05:00:00
#SBATCH --mem=240GB
#SBATCH --output=script_logging/slurm_%A.out
# Sizing notes (measured on H100, qwen3_emb_4b, 4 GPUs):
#   browsecomp_plus (100K long docs, max_length=4096): compute-bound ~5.6 docs/s/GPU,
#     throughput FLAT batch 16..64 so batch 16 is optimal (96 OOMs). ~75 min wall.
#   trqa (5.9M short wiki docs, max_length=512): ~131 docs/s/GPU; SMALLER batch wins
#     (less padding waste) so batch 32 is optimal. ~3.3 h wall + ~60GB index I/O tail.
#   --time=05:00:00 covers the slower (trqa) run; browsecomp finishes well inside it.
#   --gpus=4 takes the whole node -> request all 64 cores. mem<=480GB keeps the
#     header valid on gpu_a100 too (cores there: 72); peak host RAM ~120GB (trqa).
#   SBU is billed on ACTUAL runtime, not --time, so a generous --time is free.

# torch / faiss / vllm live in the project venv on the Python/3.13.5 module
# stack. scripts/_activate.sh loads the modules, activates the venv, and exports the
# project env vars (PYTHONUNBUFFERED, HF_HOME, HF_DATASETS_CACHE, DRA_DATA_ROOT,
# ...) — same env as interactive runs. Sourced CWD-relative: sbatch preserves
# the submission dir (the repo root), which is why the paths below are relative.
source scripts/_activate.sh
mkdir -p script_logging

RETRIEVER=qwen3_emb_4b        # bm25 | spladepp | bge | qwen3_emb_4b
DATASET=trqa                  # trqa | neuclir | browsecomp_plus

case "$RETRIEVER" in
    bm25)              ARGS=() ;;
    spladepp|spladev3) ARGS=(--use_fp16 --max_length 256 --batch_size 512 --save_embedding) ;;
    # qwen3 embedding models (4B): max_length / batch_size are corpus-dependent
    # (measured on H100, see sizing notes above). Long-doc corpora want 4096 +
    # small batch (compute-bound); short-doc corpora want 512 + small batch
    # (padding-bound, smaller batch = less wasted compute).
    qwen3_emb_*)
        case "$DATASET" in
            browsecomp_plus) ARGS=(--use_fp16 --max_length 4096 --batch_size 16 --faiss_type Flat --save_embedding) ;;
            *)               ARGS=(--use_fp16 --max_length 512  --batch_size 32 --faiss_type Flat --save_embedding) ;;
        esac
        ;;
    *)                 ARGS=(--use_fp16 --max_length 512 --batch_size 512 --faiss_type Flat --save_embedding) ;;
esac

python -m indexing_corpus_dataset.index_builder \
    --retriever "$RETRIEVER" \
    --dataset "$DATASET" \
    "${ARGS[@]}"
