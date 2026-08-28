#!/usr/bin/env bash
# Build one model's firesim ELF and park it under experiments/kopt/elfs/<tag>.elf
# usage: build.sh <model> <target rvv|rvv_f16> <tag>
set -euo pipefail
source /scratch/dima/rose-infra/RoSE/experiments/kopt/env.sh
M="$1"; T="$2"; TAG="$3"
K=/scratch/dima/rose-infra/RoSE/experiments/kopt
mkdir -p $K/elfs $K/buildlogs
cd $MB
if [ "$T" = "rvv_f16" ]; then TGT=rvv; else TGT="$T"; fi
timeout 3000 env TARGET=$TGT QUANT=int8 RUNNER=firesim STOP_AFTER=build \
    ./examples/$M/run.sh > $K/buildlogs/$TAG.log 2>&1 || { echo "BUILD FAIL $TAG"; tail -5 $K/buildlogs/$TAG.log; exit 1; }
ELF=$MB/examples/$M/int8/build/${TGT}_firesim/zephyr/zephyr.elf
[ -f "$ELF" ] || { echo "NO ELF for $TAG"; exit 1; }
cp "$ELF" $K/elfs/$TAG.elf
echo "$TAG -> $K/elfs/$TAG.elf  ($(md5sum $K/elfs/$TAG.elf | cut -c1-8))"
grep -c "curated verify PASS" $K/buildlogs/$TAG.log | xargs echo "  curated PASS:"
