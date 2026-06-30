# shellcheck shell=bash
# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth for the DRA runtime environment.
#
#   source scripts/_activate.sh
#
# Loads the Snellius 2025 module stack (Python 3.13.5 + CUDA + Java), activates
# the project venv, and exports the project env vars. Used identically by the
# interactive helper (scripts/run_interactive_node.sh) and the SBATCH scripts
# (scripts/run_*.sh), so interactive and batch runs share one environment.
#
# Replaces the old Anaconda3/2024.06 `base` conda env (Python 3.12). The venv is
# built from the module Python — see scripts/build_venv.sh to (re)create it.
# ──────────────────────────────────────────────────────────────────────────────

# ── Project venv location (on /projects: big-file friendly, shared with batch) ─
export DRA_VENV="${DRA_VENV:-/projects/0/prjs0834/heydars/DRA_training/venvs/dra313}"

# ── Make `module` available in non-interactive (sbatch) shells ────────────────
if ! type module >/dev/null 2>&1; then
    for _init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$_init" ] && source "$_init" && break
    done
fi

# ── Deactivate any active conda env — it shadows the module Python / venv ─────
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

# ── Load the module stack (stepwise: a single combined `module load` line can
#    silently fail to swap Python on this stack) ───────────────────────────────
module purge 2>/dev/null
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.8.0
module load Java/21.0.7            # pyserini / Anserini JVM

# ── Activate the project venv (if it exists) ──────────────────────────────────
if [ -f "$DRA_VENV/bin/activate" ]; then
    source "$DRA_VENV/bin/activate"
else
    echo "[_activate] WARNING: venv not found at $DRA_VENV — run scripts/build_venv.sh first" >&2
fi

# ── Make the project packages importable ──────────────────────────────────────
# Two source roots (see pyproject.toml): the repo root (utils.*, orchestration.*)
# and src/ (indexing_corpus_dataset.*, deep_research_agents.*, ...). The project
# is NOT pip-installed into the venv, so put both on PYTHONPATH. Anchored on the
# script location, so it is correct regardless of the caller's CWD.
_DRA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_DRA_REPO_ROOT}:${_DRA_REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# ── Project env vars (paths previously set inline in each SBATCH script) ──────
export PYTHONUNBUFFERED=1
export HF_HOME=/projects/0/prjs0834/heydars/.cache/huggingface
export HF_DATASETS_CACHE=/projects/0/prjs0834/heydars/.cache/huggingface
export DRA_DATA_ROOT=/projects/0/prjs0834/heydars/DRA_training/data
export DRA_OUTPUT_ROOT=/home/hsoudani/DRA_training/run_outputs

echo "[_activate] ready: $(command -v python) ($(python --version 2>&1))"
