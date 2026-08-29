#!/usr/bin/env bash
# ARM A: build the 3-net xpurt harness ELF from the armA schedule (build only;
# the F2 fq queue owns the run).
set -uo pipefail
ZCS=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
MB="$ZCS/modelblaster"
XPURT=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt
DEST=/scratch/dima/rose-infra/RoSE/experiments/armA
set +u
source "$ZCS/scripts/activate_conda.sh"  >/dev/null 2>&1
source "$ZCS/scripts/set_envvars_sdk.sh" >/dev/null 2>&1
set -u
export PYTHONPATH="${ZCS}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBAL_CURATED_DIR="$MB/kernels"
unset MB_DRIFT_ATOL MAX_ACCURACY_CLASS 2>/dev/null || true

cd "$MB"
SCHEDULE_JSON="$XPURT/schedules/scheduled_networks_3net_armA_greedy_profiled.json" \
MODELS=mlp_control,dronet,yolov8_nano \
MODEL_DIRS=mlp_control_armA,dronet_armA,yolov8_nano_armA \
QUANT=int8 QUANTS=fp32,int8,int8 \
BACKENDS=gemmini_q31,rvv \
REGISTRY="$MB/cores/chipyard_dual_rocket_gemmini_q31.json" \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv \
FORCE_REGEN=0 XPURT_TRACE=1 SCHED_NAME=xpurt_3net_armA \
STOP_AFTER=build RUNNER=firesim \
  bash examples/xpurt_demo_armA/run.sh
