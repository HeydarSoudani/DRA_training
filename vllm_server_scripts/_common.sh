# shellcheck shell=bash
# ──────────────────────────────────────────────────────────────────────────────
# Shared setup for the vLLM serve_*.sh scripts (Snellius-oriented, env-driven).
#
# Source it near the top of each serve script:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Provides:
#   - HF_HOME / DOWNLOAD_DIR     cache + download paths (env first, Snellius default)
#   - gpu_mem_gb                 per-GPU total memory in GB (0 if undetectable)
#   - auto_tp <model_gb> <csv>   smallest allowed TP that fits the model
# ──────────────────────────────────────────────────────────────────────────────

# ── Cache / download paths (env-driven; Snellius defaults) ────────────────────
# An explicitly-exported HF_HOME / DOWNLOAD_DIR (e.g. from the SBATCH script)
# always wins; otherwise fall back to the project cache on /projects.
export HF_HOME="${HF_HOME:-/projects/0/prjs0834/heydars/.cache/huggingface}"
export DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HF_HOME/hub}"

# Optional pre-staged weights dir (mirrors DRA_MODELS_DIR used by vllm_manager.py).
export DRA_MODELS_DIR="${DRA_MODELS_DIR:-/projects/0/prjs0834/heydars/DRA_training/models}"

# ── Per-GPU total memory in GB (GPU 0), or 0 if nvidia-smi is unavailable ──────
gpu_mem_gb() {
    local mb
    mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 \
         2>/dev/null | head -n1 | tr -d ' ')
    if [[ -z "$mb" ]]; then echo 0; else echo $(( mb / 1024 )); fi
}

# ── Pick the smallest allowed TP that fits a model ────────────────────────────
# Usage: auto_tp <model_gb> <allowed_csv> [gpu_mem_util]
#   model_gb     approximate on-GPU weight footprint
#   allowed_csv  comma-separated valid TP values, ascending (e.g. "1,2,4")
#   gpu_mem_util fraction of GPU memory vLLM may use (default 0.90)
# Chooses the smallest allowed TP with tp*usable >= model_gb*1.2.  When nothing
# fits, returns the largest allowed value; when GPU memory is undetectable,
# returns the first (smallest) allowed value.
auto_tp() {
    local model_gb="$1" allowed_csv="$2" util="${3:-0.90}"
    local per_gpu; per_gpu=$(gpu_mem_gb)
    if [[ "$per_gpu" -le 0 ]]; then
        echo "${allowed_csv%%,*}"
        return
    fi
    awk -v mg="$model_gb" -v allowed="$allowed_csv" -v util="$util" -v pg="$per_gpu" '
    BEGIN {
        usable = pg * util;
        need   = mg * 1.2;
        n = split(allowed, a, ",");
        chosen = a[n];                       # fallback: largest allowed
        for (i = 1; i <= n; i++) {
            if (a[i] * usable >= need) { chosen = a[i]; break }
        }
        print chosen
    }'
}
