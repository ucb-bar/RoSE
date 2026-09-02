#!/usr/bin/env bash
# Build ONE 3net benchmark ELF from a solver-produced schedule and stash it.
#   build_3net.sh <pair> <solver>     e.g. build_3net.sh hetero greedy_periodic
#
# The 3net workload is mlp_control x8 + dronet x4 + yolov8_nano x1, scheduled by
# run_xpurt_schedule.py. This turns the solver's PREDICTED makespan into an
# FPGA-measured one, which is the only way a solver comparison means anything.
set -uo pipefail
PAIR=$1; SOLVER=$2
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/3net; mkdir -p $OUT/logs $OUT/elf
# NOT "3net_..." -- the codegen builds a header guard from the schedule name,
# and `#define 3NET_..._H` is not a valid C identifier ("macro names must be
# identifiers", "invalid suffix"). Any leading digit breaks the build.
TAG="net3_${PAIR}_${SOLVER}"
exec > $OUT/logs/build_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB

case $SOLVER in
  milp) SJ=$R/soc/sw/xpu-rt/schedules/scheduled_networks_3net_${PAIR}_profiled.json ;;
  *)    SJ=$R/soc/sw/xpu-rt/schedules/scheduled_networks_3net_${PAIR}_${SOLVER}_profiled.json ;;
esac
[ -f "$SJ" ] || { echo "### ABORT no schedule at $SJ"; exit 1; }
N=$(python3 -c "import json;print(len(json.load(open('$SJ'))['dispatches']))")
P=$(python3 -c "
import json;d=json.load(open('$SJ'))['dispatches']
v=d.values() if isinstance(d,dict) else d
print(round(max(x['start_time']+x['duration'] for x in v),3))")
echo "### tag=$TAG dispatches=$N predicted_ms=$P sched=$SJ $(date -u +%FT%TZ)"

find $MB/examples -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
SCHEDULE_JSON=$SJ \
MODELS=mlp_control,dronet,yolov8_nano \
MODEL_EXDIRS=mlp_control_armB,dronet_armB,yolov8_nano_armB \
QUANTS=fp32,int8,int8 QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=build RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf \
FORCE_REGEN=0 XPURT_TRACE=1 bash examples/xpurt_demo_armB/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|undefined reference|No such" $OUT/logs/build_$TAG.log | tail -6; echo "### ABORT build"; exit 1; }
ELF=$(find $MB/examples -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "### ABORT no fresh elf"; exit 1; }
cp "$ELF" $OUT/elf/$TAG.elf
echo "#### stashed $OUT/elf/$TAG.elf"
echo "############ BUILDDONE $TAG ############"
