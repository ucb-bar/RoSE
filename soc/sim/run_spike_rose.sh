#!/usr/bin/env bash
# Run a Zephyr rose sample on Spike with the RoSE bridge plugin (--extlib).
# Fast single-core functional tier. Requires a synchronizer listening on :10001
# (deploy/hephaestus/run_sync_only.py). See soc/src/main/cc/rose_spike/README.md.
#
# Usage: run_spike_rose.sh [elf]   (default: zephyr_rose_builds/reqrsp)
set -eo pipefail
ROSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CY="$ROSE_DIR/soc/sim/chipyard"
S="$CY/toolchains/riscv-tools/riscv-isa-sim"
SO="$ROSE_DIR/soc/sim/librose_spike.so"
ELF="${1:-$ROSE_DIR/soc/sim/zephyr_rose_builds/reqrsp/zephyr/zephyr.elf}"
ELF="$(readlink -f "$ELF")"   # resolve before cd
cd "$CY"; source env.sh
timeout 30 "$S/build/spike" --extlib="$SO" \
  --device="rose_bridge,0x2000,3,0x88000000,2,1" "$ELF" 2>&1
echo "SPIKE_EXIT=$?"
