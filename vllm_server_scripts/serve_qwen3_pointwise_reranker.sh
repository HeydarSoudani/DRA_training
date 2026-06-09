#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Qwen3-Reranker (pointwise reranker).
#
# Unlike serve_qwen3_reranker.sh which serves a general Qwen3-8B model for
# *listwise* reranking, this script serves the dedicated Qwen3-Reranker model
# family for *pointwise* (yes/no) reranking.
#
# Models: Qwen/Qwen3-Reranker-0.6B, Qwen/Qwen3-Reranker-4B, Qwen/Qwen3-Reranker-8B
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_qwen3_pointwise_reranker.sh
#   MODEL_SIZE=8B bash experiments/deep_research_agents/vllm_server_scripts/serve_qwen3_pointwise_reranker.sh
#   PORT=8001 MODEL_SIZE=0.6B bash experiments/deep_research_agents/vllm_server_scripts/serve_qwen3_pointwise_reranker.sh
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): 0.6B fits on any GPU, 4B on 1× 40 GB, 8B on 1× 80 GB GPU (TP=1).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
MODEL_SIZE="${MODEL_SIZE:-8B}"
MODEL="${MODEL:-Qwen/Qwen3-Reranker-${MODEL_SIZE}}"
TP_SIZE="${TP_SIZE:-1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/mnt/sagemaker-nvme/huggingface/hub}"

# Ensure HF caches land on NVMe too
export HF_HOME="${HF_HOME:-/mnt/sagemaker-nvme/huggingface}"

# ── Check vLLM is installed ───────────────────────────────────────────────────
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none")
if [[ "$VLLM_VERSION" == "none" ]]; then
    echo "ERROR: vLLM is not installed. Run: pip install vllm"
    exit 1
fi
echo "Using vLLM ${VLLM_VERSION}"

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "Starting vLLM pointwise reranker server:"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  TP    : ${TP_SIZE}"
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --enable-prefix-caching
