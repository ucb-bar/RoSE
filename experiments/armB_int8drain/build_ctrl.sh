#!/usr/bin/env bash
# Build an ARM-A CONTROL (MB_DRIFT_ATOL UNSET) binary purely to read kernel_picks.json.
set -uo pipefail
M="$1"; Q="$2"; T="$3"
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
MB="$ZCS/modelblaster"; DEST=/scratch/dima/rose-infra/RoSE/experiments/armB_int8drain
set +u; source "$ZCS/scripts/activate_conda.sh" >/dev/null 2>&1; source "$ZCS/scripts/set_envvars_sdk.sh" >/dev/null 2>&1; set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBAL_CURATED_DIR="$MB/kernels"
unset MB_DRIFT_ATOL
( cd "$MB" && MODEL_NAME="$M" BACKEND=reference TARGET="$T" QUANT="$Q" OPTIMIZE=0 \
  STOP_AFTER=build RUNNER=firesim bash "examples/$M/run.sh" ) > "$DEST/${M}_${T}.ctrl.log" 2>&1
echo "rc=$? $M/$T"
