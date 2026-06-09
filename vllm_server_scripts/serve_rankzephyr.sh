#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for RankZephyr used as a listwise reranker.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_rankzephyr.sh
#   PORT=8000 bash experiments/deep_research_agents/vllm_server_scripts/serve_rankzephyr.sh
#
# The decomposition pipeline connects to this server via --listwise-api-url
# (default http://localhost:8000/v1).
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): RankZephyr-7B fits on 1× 80 GB GPU (TP=1). Use TP=2 for smaller GPUs.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
MODEL="${MODEL:-castorini/rank_zephyr_7b_v1_full}"
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
echo "Starting vLLM reranker server (RankZephyr):"
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
