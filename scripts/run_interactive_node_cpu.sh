#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# CPU-only interactive node helper — same design as run_interactive_node.sh but
# grabs a GPU-less shell on the `staging` partition. Use it for work that never
# touches CUDA (e.g. analysis/reasoning_error_analysis.py, corpus/gold munging).
# The environment is IDENTICAL to what the SBATCH scripts use: the Snellius 2025
# module stack (Python 3.13.5 + Java) plus the project venv. All of that lives
# in scripts/_activate.sh, the single source of truth.
#
# Two modes (auto-detected):
#
#   1. LAUNCH (run from a login node, outside any allocation):
#        bash scripts/run_interactive_node_cpu.sh
#      Grabs an interactive CPU-only node and drops you into a bash shell that
#      has already activated `base` and exported the project env vars.
#
#   2. SETUP (source it once you are already on a node):
#        source scripts/run_interactive_node_cpu.sh
#      Activates `base` + exports the project env vars in your current shell.
#      LAUNCH mode does this for you automatically, so you rarely call it byhand.
#
# Reproduces the reference request by default:
#   srun -p staging -n 1 --ntasks-per-node 1 --cpus-per-task 4 \
#        -t 3:00:00 --mem=180GB --pty /bin/bash
#
# Overridable via env:
#   PARTITION=staging CPUS=8 MEM=240GB TIME=6:00:00 bash scripts/run_interactive_node_cpu.sh
# ──────────────────────────────────────────────────────────────────────────────

# ── Shared env setup: delegate to the single source of truth (scripts/_activate.sh):
#    module stack (Python 3.13.5 + Java) + project venv + env vars. ──────────────
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

# ── LAUNCH mode: on a login node, grab an interactive CPU node ────────────────
set -euo pipefail

PARTITION="${PARTITION:-staging}"
TIME="${TIME:-1:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-180GB}"

echo "[run_interactive_node_cpu] requesting: $PARTITION  cpus=$CPUS  mem=$MEM  time=$TIME  (no GPU)"

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

exec srun -p "$PARTITION" -n 1 --ntasks-per-node 1 \
    --cpus-per-task "$CPUS" -t "$TIME" --mem="$MEM" \
    --pty /bin/bash --rcfile "$RCFILE"
