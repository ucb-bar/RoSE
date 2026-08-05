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
# Sync port: ROSE_SYNC_PORT (default 10001) must match the synchronizer's port. The parallel
# stress harness sets this per cell so many co-sim cells share one host without colliding.
ROSE_SYNC_PORT="${ROSE_SYNC_PORT:-10001}"
# `timeout --foreground`: without it, GNU timeout (>=8.13) runs the command in its OWN process
# group, so a caller that kills THIS script's group (e.g. the parallel stress harness on cell
# completion) misses the rose_spike_sim grandchild -> it orphans and pins a CPU core. With
# --foreground the sim stays in this script's group and dies with it.
# DMA landing zone: default 0x90000000 — the first address ABOVE the Zephyr guest's 256 MB
# SRAM (0x80000000..0x90000000) yet inside spike's 2 GB physical RAM, so DMA'd camera frames
# never overwrite the kernel heap (0x88000000 sits mid-SRAM and corrupts it for large frames).
# The guest overlay's dma-base-address MUST match this. Override with ROSE_DMA_BASE.
ROSE_DMA_BASE="${ROSE_DMA_BASE:-0x90000000}"
timeout --foreground "${ROSE_SPIKE_TIMEOUT:-40}" "$BIN" -p "$NPROCS" \
  --rose-base=0x2000 --rose-irq=3 --rose-dma-base="$ROSE_DMA_BASE" \
  --rose-nreqrsp=2 --rose-ndma=1 --rose-port="$ROSE_SYNC_PORT" "$ELF" 2>&1
echo "ROSE_SIM_EXIT=$?"
