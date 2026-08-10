#!/usr/bin/env bash
# In-tree RoSE Zephyr build environment for the nav co-sim experiment.
#
# Resolves ROSE_DIR from this script's location (repo-root relative), so it works
# from any clone path. Requires the in-tree Zephyr toolchain to already be
# bootstrapped inside the zephyr-chipyard-sw submodule (conda env `zephyr` under
# tools/miniforge3 + the SDK under tools-manual/). See REPRODUCE_ROSE_NAV_COSIM.md
# ("Prereq B: Zephyr/RISC-V toolchain bootstrap").
#
# Usage:  source experiments/rose_nav_cosim/env.sh
set -u
_SELF="${BASH_SOURCE[0]:-$0}"
_HERE="$(cd "$(dirname "$_SELF")" && pwd)"
ROSE_DIR="$(cd "$_HERE/../.." && pwd)"
ZSW="$ROSE_DIR/soc/sw/xpu-rt/zephyr-chipyard-sw"

if [ ! -f "$ZSW/tools/miniforge3/etc/profile.d/conda.sh" ]; then
  echo "[rose-env] ERROR: in-tree conda not found at $ZSW/tools/miniforge3" >&2
  echo "[rose-env] Bootstrap the toolchain first (see REPRODUCE_ROSE_NAV_COSIM.md)." >&2
  return 1 2>/dev/null || exit 1
fi

source "$ZSW/tools/miniforge3/etc/profile.d/conda.sh"
conda activate zephyr
source "$ZSW/scripts/set_envvars_sdk.sh" >/dev/null
export ROSE_DIR
export ROSE_MODULE="$ROSE_DIR/soc/sw/zephyr-rose"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr

echo "[rose-env] ROSE_DIR=$ROSE_DIR"
echo "[rose-env] west=$(command -v west)"
echo "[rose-env] cmake=$(command -v cmake)"
echo "[rose-env] ZEPHYR_BASE=${ZEPHYR_BASE:-<unset>}"
echo "[rose-env] SDK=${ZEPHYR_SDK_INSTALL_DIR:-<unset>}"
