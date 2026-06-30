#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for a Qwen3-*-Thinking-2507 reasoning model.
#
# Usage:
#   bash vllm_server_scripts/serve_qwen3_thinking.sh            # 30b (default)
#   bash vllm_server_scripts/serve_qwen3_thinking.sh 4b
#   PORT=8000 bash vllm_server_scripts/serve_qwen3_thinking.sh 30b
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): 4B fp16 (~9 GB, single GPU) or 30B-A3B MoE (~61 GB).  TP auto-sized
#     from GPU memory: TP=1 on a 94 GB H100, TP=2 on a 40 GB A100.  TP must
#     divide num_kv_heads (valid: 1, 2, 4).
#   - Tool calling uses the hermes parser (same as Tongyi).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

# ── Configuration ─────────────────────────────────────────────────────────────
VARIANT="${1:-30b}"
case "$VARIANT" in
    4b)  MODEL="Qwen/Qwen3-4B-Thinking-2507";  GB=9  ;;
    30b) MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"; GB=61 ;;
    *)
        echo "ERROR: unknown variant '$VARIANT' (expected '4b' or '30b')"
        exit 1
        ;;
esac

PORT="${PORT:-6008}"
TP_SIZE="${TP_SIZE:-$(auto_tp "$GB" "1,2,4")}"

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
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
# NOTE: --enable-reasoning was removed in vLLM >=0.9; reasoning is enabled
# implicitly by passing --reasoning-parser (verified against vllm 0.24).
