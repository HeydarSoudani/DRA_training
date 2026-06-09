#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start a vLLM server for Rank-R1 (ielab) used as a setwise reranker.
#
# Usage:
#   bash experiments/deep_research_agents/vllm_server_scripts/serve_rank_r1.sh
#   PORT=8001 bash experiments/deep_research_agents/vllm_server_scripts/serve_rank_r1.sh
#
# The pipeline connects to this server via --rank-r1-api-url
# (default http://localhost:8000/v1).
#
# Rank-R1 is a setwise reranker that selects the most relevant document from
# a set of candidates.  It consists of LoRA adapters on top of
# Qwen/Qwen2.5-7B-Instruct, served via vLLM's --enable-lora flag.
# The pipeline calls the /v1/chat/completions endpoint.
#
# Available LoRA variants (override via LORA_MODEL env var):
#   ielabgroup/Rank-R1-3B-v0.1   (base: Qwen/Qwen2.5-3B-Instruct)
#   ielabgroup/Rank-R1-7B-v0.1   (base: Qwen/Qwen2.5-7B-Instruct)   [default]
#   ielabgroup/Rank-R1-14B-v0.1  (base: Qwen/Qwen2.5-14B-Instruct)
#   ielabgroup/Rank-R1-32B-v0.2  (base: Qwen3 series)
#
# Requirements:
#   - vLLM installed.
#   - GPU(s): 7B fits on 1× 80 GB GPU (TP=1). 14B may need TP=2.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LORA_MODEL="${LORA_MODEL:-ielabgroup/Rank-R1-7B-v0.1}"
LORA_ALIAS="${LORA_ALIAS:-rank-r1}"
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
echo "Starting vLLM reranker server (Rank-R1 - ielab setwise):"
echo "  Base model : ${BASE_MODEL}"
echo "  LoRA       : ${LORA_MODEL}"
echo "  Alias      : ${LORA_ALIAS}"
echo "  Port       : ${PORT}"
echo "  TP         : ${TP_SIZE}"
echo ""

exec vllm serve "$BASE_MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --download-dir "$DOWNLOAD_DIR" \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --enable-lora \
    --lora-modules "${LORA_ALIAS}=${LORA_MODEL}" \
    --max-lora-rank 32 \
    --gpu-memory-utilization 0.9
