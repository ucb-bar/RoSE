#!/usr/bin/env bash
# Build one BN-folding A/B arm for both backends (gemmini_q31 + rvv).
# usage: build_arm.sh <bnfold|nobnfold>
set -uo pipefail
ARM="$1"
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
MB="$ZCS/modelblaster"
DEST=/scratch/dima/rose-infra/RoSE/experiments/bnfold
set +u
source "$ZCS/scripts/activate_conda.sh"  >/dev/null 2>&1
source "$ZCS/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$ARM" == "nobnfold" ]]; then export MB_NO_BN_FOLDING=1; else unset MB_NO_BN_FOLDING || true; fi

for T in gemmini_q31 rvv; do
  echo "=== [$ARM/$T] build start $(date -Is)"
  ( cd "$MB" && \
    MODEL_NAME="yolov8_nano_${ARM}" BACKEND=reference TARGET="$T" QUANT=int8 \
    OPTIMIZE=0 STOP_AFTER=build RUNNER=firesim \
    bash "examples/yolov8_nano_${ARM}/run.sh" ) > "$DEST/${ARM}_${T}.build.log" 2>&1
  rc=$?
  elf="$MB/examples/yolov8_nano_${ARM}/int8/build/${T}_firesim/zephyr/zephyr.elf"
  echo "=== [$ARM/$T] rc=$rc elf=$( [[ -f $elf ]] && stat -c %s "$elf" || echo MISSING ) $(date -Is)"
done
