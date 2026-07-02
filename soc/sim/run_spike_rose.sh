#!/usr/bin/env bash
# Run a Zephyr rose sample on Spike with the RoSE bridge plugin.
# Usage: run_spike_rose.sh [elf]   (default: samples/rose/reqrsp)
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
S=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/toolchains/riscv-tools/riscv-isa-sim
SO=/scratch/dima/rose-infra/RoSE/soc/sim/librose_spike.so
ELF="${1:-/scratch/dima/rose-infra/RoSE/soc/sim/zephyr_rose_builds/reqrsp/zephyr/zephyr.elf}"
timeout 30 "$S/build/spike" --extlib="$SO" \
  --device="rose_bridge,0x2000,3,0x88000000,2,1" "$ELF" 2>&1
echo "SPIKE_EXIT=$?"
