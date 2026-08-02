#!/usr/bin/env bash
# Run a Zephyr rose sample on the RoSE lockstep Spike harness (rose_spike_sim).
# True two-sided cycle barrier, multicore-capable. Contrast run_spike_rose.sh
# (the passive --extlib functional tier).
#
# Usage: run_spike_rose_lockstep.sh [elf] [nprocs]
# Wall-clock guard: ROSE_SPIKE_TIMEOUT seconds (default 40). Raise it for slow physics
# backends — e.g. an Isaac Sim env steps at ~0.1-0.4 s/step, so 40 s caps a run at ~100
# steps mid-flight; set ROSE_SPIKE_TIMEOUT=600 for a full multi-second co-sim.
set -eo pipefail
ROSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ELF="${1:-$ROSE_DIR/soc/sim/zephyr_rose_builds/reqrsp/zephyr/zephyr.elf}"
ELF="$(readlink -f "$ELF")"   # resolve before cd
cd "$ROSE_DIR/soc/sim/chipyard"; source env.sh
BIN="$ROSE_DIR/soc/sim/rose_spike_sim"
NPROCS="${2:-1}"
timeout "${ROSE_SPIKE_TIMEOUT:-40}" "$BIN" -p "$NPROCS" \
  --rose-base=0x2000 --rose-irq=3 --rose-dma-base=0x88000000 \
  --rose-nreqrsp=2 --rose-ndma=1 "$ELF" 2>&1
echo "ROSE_SIM_EXIT=$?"
