#!/usr/bin/env bash
# ARM A: build the 6 per-op profiling ELFs (3 nets x {gemmini_q31, rvv}).
# EXACT conv (int32 drain): no MB_DRIFT_ATOL, no MAX_ACCURACY_CLASS.
set -uo pipefail
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
MB="$ZCS/modelblaster"
DEST=/scratch/dima/rose-infra/RoSE/experiments/armA
set +u
source "$ZCS/scripts/activate_conda.sh"  >/dev/null 2>&1
source "$ZCS/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBAL_CURATED_DIR="$MB/kernels"
unset MB_DRIFT_ATOL MAX_ACCURACY_CLASS 2>/dev/null || true

declare -A Q=( [mlp_control]=fp32 [dronet]=int8 [yolov8_nano]=int8 )
MODELS="${MODELS:-mlp_control dronet yolov8_nano}"
TARGETS="${TARGETS:-gemmini_q31 rvv}"
for m in $MODELS; do
  q=${Q[$m]}
  for T in $TARGETS; do
    k="${m}_${T}"
    echo "=== [$k] build start $(date -Is)  quant=$q curated=$GLOBAL_CURATED_DIR"
    ( cd "$MB" && \
      MODEL_NAME="${m}_armA" BACKEND=reference TARGET="$T" QUANT="$q" \
      OPTIMIZE=0 STOP_AFTER=build RUNNER=firesim \
      GLOBAL_CURATED_DIR="$GLOBAL_CURATED_DIR" \
      bash "examples/${m}_armA/run.sh" ) > "$DEST/${k}.build.log" 2>&1
    rc=$?
    elf="$MB/examples/${m}_armA/$q/build/${T}_firesim/zephyr/zephyr.elf"
    sz=$( [[ -f $elf ]] && stat -c %s "$elf" || echo MISSING )
    echo "=== [$k] rc=$rc elf=$sz $(date -Is)"
  done
done
