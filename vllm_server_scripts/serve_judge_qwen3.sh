#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Qwen3-32B used as the accuracy-evaluation judge.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_judge_qwen3.sh
#   PORT=6009 bash experiments/deep_research_agents/vllm_server_scripts/serve_judge_qwen3.sh
#
# The pipeline connects to this server via --judge-api-url
# (default http://localhost:6009/v1).
#
# When started externally with this script, the pipeline skips its own
# automatic judge-server lifecycle (start_judge_server / shutdown_judge_server).
#
# Parameters match the original AgentIR evaluation setup
# (https://github.com/texttron/AgentIR/blob/main/evaluation/evaluate_bcp.py).
# AgentIR uses vLLM offline with default TP=1; override TP_SIZE as needed.
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): Qwen3-32B fits on 1× 80 GB GPU (TP=1, default).
#             Use TP_SIZE=4 to spread across multiple GPUs for faster inference.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-6009}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/mnt/sagemaker-nvme/huggingface/hub}"

# Ensure HF caches land on NVMe too
export HF_HOME="${HF_HOME:-/mnt/sagemaker-nvme/huggingface}"

# Pin vLLM to the first TP_SIZE GPUs (0..TP_SIZE-1) so the remaining GPUs
# stay free for pipeline workers.
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
echo "Starting vLLM accuracy-evaluation judge server (Qwen3-32B):"
echo "  Model        : ${MODEL}"
echo "  Port         : ${PORT}"
echo "  TP           : ${TP_SIZE}"
echo "  Max seq len  : ${MAX_MODEL_LEN}"
echo "  GPUs         : ${CUDA_VISIBLE_DEVICES}"
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "${MAX_NUM_SEQS:-4}" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90
