#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for the OpenAI gpt-oss-20b model.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_gpt_oss.sh              # defaults: gpt-oss-20b, port 6008
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_gpt_oss.sh 120b         # serve gpt-oss-120b instead
#   PORT=8000 bash experiments/deep_research_agents/vllm_server_scripts/serve_gpt_oss.sh    # custom port
#
# Requirements:
#   - vLLM installed with gpt-oss support.
#   - GPU(s): 20b fits on 1× L4 24 GB GPU (mxfp4 quantized MoE);
#             120b needs TP=8 across 8 GPUs.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
VARIANT="${1:-20b}"                         # "20b" or "120b"
PORT="${PORT:-6008}"
MODEL="openai/gpt-oss-${VARIANT}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/mnt/sagemaker-nvme/huggingface/hub}"

# Ensure HF caches land on NVMe too
export HF_HOME="${HF_HOME:-/mnt/sagemaker-nvme/huggingface}"

if [[ "$VARIANT" == "120b" ]]; then
    TP_SIZE="${TP_SIZE:-8}"                  # 120b: tensor-parallel across GPUs
else
    TP_SIZE="${TP_SIZE:-1}"                  # 20b:  single GPU is enough (mxfp4)
fi

# Pin vLLM to the first TP_SIZE GPUs (0..TP_SIZE-1) so the remaining GPUs
# stay free for retrieval workers.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TP_SIZE - 1)))}"

# ── Check vLLM is installed ───────────────────────────────────────────────────
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none")
if [[ "$VLLM_VERSION" == "none" ]]; then
    echo "ERROR: vLLM is not installed. Run: pip install vllm"
    exit 1
fi
echo "Using vLLM ${VLLM_VERSION}"

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "Starting vLLM server:"
echo "  Model : ${MODEL}"
echo "  Port  : ${PORT}"
echo "  TP    : ${TP_SIZE}"
echo "  GPUs  : ${CUDA_VISIBLE_DEVICES}"
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager
