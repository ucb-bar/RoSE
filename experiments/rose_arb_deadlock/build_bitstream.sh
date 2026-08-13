#!/usr/bin/env bash
# Build the arbiter-deadlock-FIX bitstream: MMIO Saturn+RoSE @ 30 MHz with the deeper
# 256-deep per-channel rx FIFO + arbiter stall counters. Chisel elaboration runs first
# (fail-fast on any RTL error), then Vivado synth+PAR (~hours). Runs on garden localhost.
set -uo pipefail
FSIMDIR=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/sims/firesim
echo "=== BUILD START $(date) ==="
( set +u
  cd "$FSIMDIR" && source env.sh >/dev/null 2>&1 && source sourceme-manager.sh --skip-ssh-setup >/dev/null 2>&1 \
  && cd deploy \
  && firesim buildbitstream -c config_build.yaml -r config_build_recipes.yaml )
echo "=== BUILD EXIT=$? $(date) ==="
