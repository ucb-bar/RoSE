#!/usr/bin/env bash
# Build the RoSE bridge Zephyr samples for Spike (board spike_riscv64), linking the
# zephyr-rose module. Outputs to soc/sim/zephyr_rose_builds/<app>/zephyr/zephyr.elf,
# which run_spike_rose*.sh consume. The same elfs also run on the FireSim metasim/FPGA.
#
# All toolchain deps are LOCAL to the zephyr-chipyard-sw submodule (no external SDK):
# conda+west in tools/miniforge3, the beta Zephyr SDK in tools-manual/, and the
# zephyr_ws workspace. One-time bootstrap (downloads conda + SDK into the submodule):
#   cd soc/sw/xpu-rt/zephyr-chipyard-sw
#   git submodule update --init --recursive      # zephyr_ws workspace
#   source scripts/install_conda.sh              # conda env 'zephyr' (provides west)
#   bash   scripts/install_toolchain_sdk.sh      # beta Zephyr SDK -> tools-manual/
# See soc/src/main/cc/rose_spike/README.md.
#
# Usage: build_zephyr_rose.sh [app ...]   (default: reqrsp dma protocol selftest)
set -eo pipefail
ROSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODULE="$ROSE_DIR/soc/sw/zephyr-rose"
ZEPHYR_SW="$ROSE_DIR/soc/sw/xpu-rt/zephyr-chipyard-sw"
SAMPLES="$ZEPHYR_SW/samples/rose"
OUT="$ROSE_DIR/soc/sim/zephyr_rose_builds"

if [ ! -d "$SAMPLES" ]; then
  echo "ERROR: samples not found at $SAMPLES" >&2
  echo "       Init (NOT --recursive; xpu-rt nests chipyard/llvm):" >&2
  echo "         git submodule update --init soc/sw/xpu-rt" >&2
  echo "         git -C soc/sw/xpu-rt submodule update --init zephyr-chipyard-sw" >&2
  exit 1
fi
if [ ! -f "$ZEPHYR_SW/tools/miniforge3/etc/profile.d/conda.sh" ]; then
  echo "ERROR: in-tree conda not installed. One-time bootstrap:" >&2
  echo "  cd $ZEPHYR_SW && source scripts/install_conda.sh && bash scripts/install_toolchain_sdk.sh" >&2
  exit 1
fi

# Activate the in-tree conda env (provides west) + point ZEPHYR_BASE / ZEPHYR_SDK at
# the submodule-local workspace and beta SDK (scripts resolve paths repo-root-relative).
set +e
source "$ZEPHYR_SW/scripts/activate_conda.sh"
source "$ZEPHYR_SW/scripts/set_envvars_sdk.sh"
set -e

if ! command -v west >/dev/null 2>&1; then
  echo "ERROR: 'west' not available after activating the in-tree zephyr conda env." >&2
  echo "       Re-run: cd $ZEPHYR_SW && source scripts/install_conda.sh" >&2
  exit 1
fi
if [ ! -d "${ZEPHYR_SDK_INSTALL_DIR:-}" ]; then
  echo "ERROR: Zephyr SDK not found at '${ZEPHYR_SDK_INSTALL_DIR:-<unset>}'." >&2
  echo "       Install it: cd $ZEPHYR_SW && bash scripts/install_toolchain_sdk.sh" >&2
  exit 1
fi

apps=("$@")
[ ${#apps[@]} -eq 0 ] && apps=(reqrsp dma protocol selftest)

for app in "${apps[@]}"; do
  src="$SAMPLES/$app"
  if [ ! -d "$src" ]; then echo "skip: no sample at $src"; continue; fi
  echo "=== building rose sample: $app ==="
  west build -p always -b spike_riscv64 --build-dir "$OUT/$app" "$src" \
    -- -DZEPHYR_EXTRA_MODULES="$MODULE"
  echo "built $OUT/$app/zephyr/zephyr.elf"
done
echo "ZEPHYR_ROSE_BUILD_EXIT=0"
