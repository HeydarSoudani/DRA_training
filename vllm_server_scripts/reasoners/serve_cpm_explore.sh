#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for the AgentCPM-Explore model.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_cpm_explore.sh
#   PORT=8000 bash experiments/deep_research_agents/vllm_server_scripts/serve_cpm_explore.sh
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): ~8B; fits on a single H100/A100 (TP=1).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-6008}"
MODEL="${MODEL:-openbmb/AgentCPM-Explore}"
TP_SIZE="${TP_SIZE:-$(auto_tp 16 "1,2")}"

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
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 64
