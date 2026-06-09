#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for the Tongyi-DeepResearch-30B-A3B model.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_tongyi.sh
#   PORT=8000 bash experiments/deep_research_agents/vllm_server_scripts/serve_tongyi.sh
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): MoE model (30B total, 3B active), ~61 GB weights (FP16).
#     TP=4 on 4× 24 GB GPUs (uses GPUs 0-3, leaves 4-7 free for pipeline).
#     TP must divide num_kv_heads=4 (valid: 1, 2, 4).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-6008}"
MODEL="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
TP_SIZE="${TP_SIZE:-4}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/mnt/sagemaker-nvme/huggingface/hub}"

# Pin vLLM to the first TP_SIZE GPUs (0..TP_SIZE-1) so the remaining GPUs
# stay free for retrieval workers.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TP_SIZE - 1)))}"

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
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
