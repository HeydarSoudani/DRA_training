#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Interactive node helper — gives an interactive GPU shell whose environment is
# IDENTICAL to what the SBATCH scripts use: the Snellius 2025 module stack
# (Python 3.13.5 + CUDA + Java) plus the project venv (torch / vllm / faiss).
# All of that lives in scripts/_activate.sh, the single source of truth.
#
# Two modes (auto-detected):
#
#   1. LAUNCH (run from a login node, outside any allocation):
#        bash scripts/run_interactive_node.sh            # h100 (default)
#        bash scripts/run_interactive_node.sh a100
#      Grabs an interactive 2-GPU node and drops you into a bash shell that has
#      already activated `base` and exported the project env vars.
#
#   2. SETUP (source it once you are already on a node):
#        source scripts/run_interactive_node.sh
#      Activates `base` + exports the project env vars in your current shell.
#      LAUNCH mode does this for you automatically, so you rarely call it byhand.
#
# Overridable via env: TIME=2:00:00 GPUS=4 bash scripts/run_interactive_node.sh a100
# ──────────────────────────────────────────────────────────────────────────────

# ── Shared env setup: delegate to the single source of truth (scripts/_activate.sh):
#    module stack (Python 3.13.5 + CUDA + Java) + project venv + env vars. ──────
_dra_setup_env() {
    source "$(dirname "${BASH_SOURCE[0]}")/_activate.sh"
}

# ── Detect: are we being sourced, or executed? ────────────────────────────────
# When sourced, $0 is the parent shell (e.g. -bash / bash), not this file.
_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then _sourced=1; fi

# ── SETUP mode: already inside an allocation, or explicitly sourced ───────────
if [[ -n "${SLURM_JOB_ID:-}" || "$_sourced" -eq 1 ]]; then
    _dra_setup_env
    [[ "$_sourced" -eq 1 ]] && return 0 || exit 0
fi

# ── LAUNCH mode: on a login node, grab an interactive node ────────────────────
set -euo pipefail

PARTITION="${1:-h100}"
TIME="${TIME:-0:30:00}"
GPUS="${GPUS:-1}"

case "$PARTITION" in
    a100|gpu_a100) PART=gpu_a100; CPUS_PER_GPU=18; MEM_PER_GPU=120 ;;
    h100|gpu_h100) PART=gpu_h100; CPUS_PER_GPU=16; MEM_PER_GPU=180 ;;
    *)
        echo "ERROR: unknown partition '$PARTITION' (expected 'a100' or 'h100')" >&2
        exit 1 ;;
esac

# cpu / mem scale with GPUs (standard Snellius per-GPU ratios). Changing only
# GPUS keeps the request balanced; GPUS=2 reproduces the original sizes
# (a100: 36c/240GB, h100: 32c/360GB).
CPUS=$(( GPUS * CPUS_PER_GPU ))
MEM="$(( GPUS * MEM_PER_GPU ))GB"

echo "[run_interactive_node] requesting: $PART  gpus=$GPUS  cpus=$CPUS  mem=$MEM  time=$TIME"

# Launch the allocation and set up the env inside the new shell. We start an
# interactive bash with an rcfile that first runs the normal ~/.bashrc, then
# sources this script in SETUP mode so `base` + the env vars are active the
# moment the prompt appears.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
# NB: the rcfile must live on a SHARED filesystem. srun launches bash on the
# compute node, where /tmp is node-local — a default `mktemp` (→ /tmp) file
# written here on the login node would be invisible there, so bash would start
# with no rcfile and the env would never get activated. $HOME is shared GPFS.
RCFILE="$(mktemp "${HOME}/.dra_interactive_rc.XXXXXX")"
cat > "$RCFILE" <<EOF
[ -f ~/.bashrc ] && source ~/.bashrc
source "$SELF"
rm -f "$RCFILE"
EOF

exec srun -p "$PART" -n 1 --ntasks-per-node 1 --gpus "$GPUS" \
    --cpus-per-task "$CPUS" -t "$TIME" --mem="$MEM" \
    --pty /bin/bash --rcfile "$RCFILE"
