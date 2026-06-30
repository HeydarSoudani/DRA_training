#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Create (or rebuild) the DRA project venv on the Python/3.13.5 module stack and
# install requirements.txt. Run once on a node with internet (login node is OK):
#
#   bash scripts/build_venv.sh             # create venv + install
#   bash scripts/build_venv.sh --recreate  # wipe and rebuild from scratch
#
# After this, both interactive and batch runs use it via scripts/_activate.sh.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Make `module` available in non-interactive shells.
if ! type module >/dev/null 2>&1; then
    for _init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$_init" ] && source "$_init" && break
    done
fi

# conda must not be active — it shadows the module Python.
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

module purge 2>/dev/null
module load 2025
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.8.0
module load Java/21.0.7

DRA_VENV="${DRA_VENV:-/projects/0/prjs0834/heydars/DRA_training/venvs/dra313}"

if [[ "${1:-}" == "--recreate" && -d "$DRA_VENV" ]]; then
    echo "[build_venv] removing existing venv: $DRA_VENV"
    rm -rf "$DRA_VENV"
fi

if [[ ! -d "$DRA_VENV" ]]; then
    echo "[build_venv] creating venv with $(python3 --version) at $DRA_VENV"
    mkdir -p "$(dirname "$DRA_VENV")"
    python3 -m venv "$DRA_VENV"
fi

source "$DRA_VENV/bin/activate"
echo "[build_venv] using $(command -v python) ($(python --version 2>&1))"

python -m pip install --upgrade pip setuptools wheel

# Install order matters: torch trio must land before vllm/faiss so the resolver
# anchors on the pinned ABI-matched build (see requirements.txt header).
echo "[build_venv] installing requirements.txt ..."
python -m pip install -r requirements.txt

echo "[build_venv] done. Resolved key versions:"
python - <<'EOF'
for m in ("torch","vllm","faiss","transformers","datasets","pyserini","numpy"):
    try:
        mod=__import__(m); print(f"  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14s} IMPORT FAILED: {type(e).__name__}: {e}")
EOF
echo "[build_venv] If versions look right, re-pin requirements.txt to 'pip freeze' output."
