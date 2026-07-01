#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for the GLM-4.7-Flash model.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_glm.sh
#   PORT=8000 bash experiments/deep_research_agents/vllm_server_scripts/serve_glm.sh
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): 30B fp16 (~60 GB).  TP auto-sized from GPU memory: TP=1 on a
#     single 80+ GB H100/A100, TP=2 only on smaller cards.  20 attention heads
#     → TP must divide 20.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-6008}"
MODEL="zai-org/GLM-4.7-Flash"
TP_SIZE="${TP_SIZE:-$(auto_tp 60 "1,2,4")}"
PP_SIZE="${PP_SIZE:-1}"

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
echo "  PP    : ${PP_SIZE}"
echo ""

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --pipeline-parallel-size "$PP_SIZE" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --max-model-len 202752 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser glm47
