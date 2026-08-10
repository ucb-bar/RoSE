#!/usr/bin/env bash
# Reproduce the Isaac side (Prereq A) of the RoSE nav co-sim on a fresh machine:
# a pinned IsaacLab checkout + an `env_isaaclab` conda env with Isaac Sim, then apply
# the RoSE overlay and verify the co-sim gym envs register (GPU-free).
#
# This is a big external install (Isaac Sim wheels + torch+cu128 are multi-GB). The
# script is deliberately conservative: it prints each step and stops on error, and it
# will NOT clobber an existing conda env unless you pass FORCE=1.
#
# Parameters (env vars):
#   ISAACLAB_DIR   where to clone/find IsaacLab      (default: $SCRATCH_BASE/IsaacLab)
#   ENV_NAME       conda env name                    (default: env_isaaclab)
#   CONDA_BASE     conda install root                (default: autodetected / $SCRATCH_BASE/miniforge3)
#   SCRATCH_BASE   large-volume base for installs    (default: /scratch2/$USER)
#   ROSE_DIR       this repo root                    (default: resolved from this script)
#   PIP_ISAACSIM   1=pip install Isaac Sim wheels    (default: 1)
#   FORCE          1=recreate the conda env if it exists (default: 0)
#
# Usage:
#   SCRATCH_BASE=/scratch2/$USER bash experiments/rose_nav_cosim/setup_isaaclab.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSE_DIR="${ROSE_DIR:-$(cd "$HERE/../.." && pwd)}"
OVL="$HERE/isaaclab_overlay"

SCRATCH_BASE="${SCRATCH_BASE:-/scratch2/$USER}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$SCRATCH_BASE/IsaacLab}"
ENV_NAME="${ENV_NAME:-env_isaaclab}"
CONDA_BASE="${CONDA_BASE:-$SCRATCH_BASE/miniforge3}"
PIP_ISAACSIM="${PIP_ISAACSIM:-1}"
FORCE="${FORCE:-0}"

PIN_COMMIT="$(grep -oE '[0-9a-f]{40}' "$OVL/PINNED_VERSION.txt" | head -1)"
PIN_REPO="https://github.com/isaac-sim/IsaacLab.git"

echo "=============================================================="
echo " RoSE nav co-sim -- Isaac side setup"
echo "   ROSE_DIR     = $ROSE_DIR"
echo "   ISAACLAB_DIR = $ISAACLAB_DIR"
echo "   pinned commit= $PIN_COMMIT"
echo "   conda base   = $CONDA_BASE"
echo "   env name     = $ENV_NAME"
echo "=============================================================="

# --- 0. sanity: never build on a full root fs ---
echo "[0/6] disk sanity"
df -h "$SCRATCH_BASE" 2>/dev/null || true

# --- 1. conda ---
echo "[1/6] conda"
if [ ! -x "$CONDA_BASE/bin/conda" ]; then
  echo "  conda not found at $CONDA_BASE."
  echo "  Install miniforge there first, e.g.:"
  echo "    wget -O /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  echo "    bash /tmp/mf.sh -b -p $CONDA_BASE"
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 2. IsaacLab checkout at the pinned commit ---
echo "[2/6] IsaacLab checkout @ $PIN_COMMIT"
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  git clone "$PIN_REPO" "$ISAACLAB_DIR"
fi
git -C "$ISAACLAB_DIR" fetch --all --tags
git -C "$ISAACLAB_DIR" checkout "$PIN_COMMIT"

# --- 3. create env_isaaclab ---
echo "[3/6] conda env $ENV_NAME"
if conda env list | grep -qE "^\s*$ENV_NAME\s|/$ENV_NAME$"; then
  if [ "$FORCE" = "1" ]; then
    conda env remove -n "$ENV_NAME" -y
    conda env create -n "$ENV_NAME" -f "$OVL/environment_isaaclab.yml"
  else
    echo "  env $ENV_NAME already exists (pass FORCE=1 to recreate). Reusing."
  fi
else
  # Prefer the captured conda spec; falls back to a bare py3.11 + pip freeze.
  conda env create -n "$ENV_NAME" -f "$OVL/environment_isaaclab.yml" \
    || { conda create -n "$ENV_NAME" -y python=3.11.15; }
fi
conda activate "$ENV_NAME"
PYBIN="$(command -v python)"
echo "  python = $PYBIN ($(python --version 2>&1))"

# --- 4. Isaac Sim wheels + IsaacLab (editable) + exact pip set ---
echo "[4/6] pip packages"
if [ "$PIP_ISAACSIM" = "1" ]; then
  # The pinned manifest already lists isaacsim==5.1.0.0 + all isaacsim-* extensions and
  # the -e IsaacLab source subprojects. Replaying it reproduces the validated env.
  # (Large multi-GB download. Requires a CUDA-capable box for the wheels to be useful.)
  python -m pip install --upgrade pip
  python -m pip install -r "$OVL/pip_freeze_isaaclab.txt" || {
    echo "  NOTE: a full 'pip install -r pip_freeze' may need --extra-index-url for torch cu128"
    echo "        and the Isaac Sim wheel index. See docs/REPRODUCE_ISAACLAB.md."
  }
fi

# The IsaacLab editable subprojects resolve against $ISAACLAB_DIR. If the pip freeze's
# git URLs point elsewhere, install the local checkout editable instead:
"$ISAACLAB_DIR/isaaclab.sh" -i 2>/dev/null || {
  for sub in isaaclab isaaclab_assets isaaclab_rl isaaclab_contrib isaaclab_tasks; do
    python -m pip install -e "$ISAACLAB_DIR/source/$sub" 2>/dev/null || true
  done
}

# --- 5. apply the RoSE overlay onto the IsaacLab tree ---
echo "[5/6] apply RoSE overlay"
ISAACLAB_DIR="$ISAACLAB_DIR" bash "$OVL/apply_overlay.sh"

# --- 6. GPU-free registration check ---
echo "[6/6] headless registration check (no GPU)"
ROSE_DIR="$ROSE_DIR" CUDA_VISIBLE_DEVICES="" python "$OVL/check_registration.py"

echo "=============================================================="
echo " Isaac side ready. env python: $PYBIN"
echo " Run the co-sim with:  ISAAC_PY=$PYBIN experiments/rose_nav_cosim/run_cosim.sh"
echo "=============================================================="
