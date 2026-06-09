#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Rank1 (JHU CLSP) used as a pointwise reranker.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_rank1.sh
#   PORT=8000 bash experiments/deep_research_agents/vllm_server_scripts/serve_rank1.sh
#
# The pipeline connects to this server via --rank1-api-url
# (default http://localhost:8000/v1).
#
# Rank1 is a pointwise reranker that scores (query, passage) pairs via
# P(true)/P(false) logprobs.  It uses the /v1/completions endpoint (NOT chat).
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): rank1-7b fits on 1× 80 GB GPU (TP=1).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
MODEL="${MODEL:-jhu-clsp/rank1-7b}"
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
echo "Starting vLLM reranker server (Rank1 - JHU CLSP pointwise):"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  TP    : ${TP_SIZE}"
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --dtype float16 \
    --gpu-memory-utilization 0.9
