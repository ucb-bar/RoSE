#!/usr/bin/env bash
# Build the ARM B xpurt harness ELF (STOP_AFTER=build; the F2/fq path owns the run).
set -uo pipefail
XPURT=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt
ZCS="$XPURT/zephyr-chipyard-sw"
MB="$ZCS/modelblaster"
DEST=/scratch/dima/rose-infra/RoSE/experiments/armB_int8drain
set +u
source "$ZCS/scripts/activate_conda.sh"  >/dev/null 2>&1
source "$ZCS/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBAL_CURATED_DIR="$MB/kernels"
export MB_DRIFT_ATOL="${MB_DRIFT_ATOL:-6}"
cd "$MB"
SCHEDULE_JSON="$XPURT/schedules/scheduled_networks_3net_armB_greedy_profiled.json" \
  MODELS=mlp_control,dronet,yolov8_nano \
  MODEL_EXDIRS=mlp_control_armB,dronet_armB,yolov8_nano_armB \
  QUANT=int8 QUANTS=fp32,int8,int8 \
  BACKENDS=gemmini_q31,rvv \
  REGISTRY="$MB/cores/chipyard_dual_rocket_gemmini_q31.json" \
  CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv \
  FORCE_REGEN=0 XPURT_TRACE=1 SCHED_NAME=xpurt_f2_3net_armB \
  STOP_AFTER=build RUNNER=firesim \
  bash "$MB/examples/xpurt_demo_armB/run.sh" > "$DEST/harness_armB.build.log" 2>&1
echo "rc=$?"
tail -4 "$DEST/harness_armB.build.log"
