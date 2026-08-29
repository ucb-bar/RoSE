#!/usr/bin/env bash
# build_one.sh <model_base> <quant> <target>
#   e.g. build_one.sh dronet int8 gemmini_q31
# Builds the ARM B (MB_DRIFT_ATOL=2, int8-drain conv) profile ELF for one
# (model, backend) in the ISOLATED examples/<model>_armB tree.
set -uo pipefail
MB_BASE="$1"; Q="$2"; T="$3"
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
MB="$ZCS/modelblaster"
DEST=/scratch/dima/rose-infra/RoSE/experiments/armB_int8drain
set +u
source "$ZCS/scripts/activate_conda.sh"  >/dev/null 2>&1
source "$ZCS/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBAL_CURATED_DIR="$MB/kernels"
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-2}"   # ARM B: accept N LSB, selects gemmini_tiled_conv

M="${MB_BASE}_armB"
LOG="$DEST/${MB_BASE}_${T}.build.log"
echo "=== [$M/$T] build start $(date -Is)  MB_DRIFT_ATOL=$MB_DRIFT_ATOL GLOBAL_CURATED_DIR=$GLOBAL_CURATED_DIR"
( cd "$MB" && \
  MODEL_NAME="$M" BACKEND=reference TARGET="$T" QUANT="$Q" \
  OPTIMIZE=0 STOP_AFTER=build RUNNER=firesim \
  bash "examples/$M/run.sh" ) > "$LOG" 2>&1
rc=$?
elf="$MB/examples/$M/$Q/build/${T}_firesim/zephyr/zephyr.elf"
echo "=== [$M/$T] rc=$rc elf=$( [[ -f $elf ]] && stat -c %s "$elf" || echo MISSING ) $(date -Is)"
