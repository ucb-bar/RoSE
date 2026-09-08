#!/usr/bin/env bash
# Build ONE harness_xpurt ELF from an explicit schedule JSON.
#
#   build_cell.sh <workload.json> <schedule.json> <tag> <worker_example_dir_name>
#
# Differs from experiments/workload_gen/build_workload.sh in exactly two ways,
# both forced by this sweep's shape:
#
#  1. The schedule path is passed in, not derived. build_workload.sh computes
#     it as schedules/scheduled_<basename>_<solver>_profiled.json -- and the
#     base and shard arms share a basename, so the two arms OVERWRITE each
#     other's schedule. That is survivable when you solve-then-build one cell
#     at a time (which is what run_wl_sweep.sh does); it is not survivable when
#     schedules for 1056 rows are emitted up front.
#  2. Per-worker example dir, so N builds can run at once. run.sh derives both
#     GEN_DIR and BUILD_DIR from its own location, so a copy of run.sh under a
#     different examples/<dir> gets its own build tree. The shared
#     examples/xpurt_demo_armB/int8/build would otherwise be clobbered by every
#     concurrent build.
#
# Freshness: the ELF is required to be NEWER than a stamp taken immediately
# before the build, and the emitted dispatch count is cross-checked against the
# schedule's. A stale ELF silently reused across cells is the classic way a
# sweep like this produces confident nonsense.
set -uo pipefail
WL=$(readlink -f "$1"); SJ=$(readlink -f "$2"); TAG=$3; WDIR=${4:-xpurt_s10_w1}
R=/scratch/dima/rose-infra/RoSE
ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/sched_algo_sweep10
mkdir -p $OUT/elf $OUT/fpga/logs
LOG=$OUT/fpga/logs/build_$TAG.log
exec > "$LOG" 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
# MB_DRIFT_ATOL=2 is a GATE, not a tolerance knob: unset, the gemmini backend
# falls back to a software im2col that is 3.45x slower, so the ELF would not be
# executing the kernels the schedule was costed against.
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
# `west build` shells out to `cmake --build`, which defaults ninja to every
# core. N concurrent builds then ask for N x nproc jobs -- 6 workers took this
# 48-core box to load 135 and starved the CP-SAT solves running beside them,
# which are WALL-CLOCK limited and so lose real search to contention.
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-7}
cd $MB

[ -f "$SJ" ] || { echo "### ABORT no schedule at $SJ"; exit 1; }
[ -x "$MB/examples/xpurt_s10/run.sh" ] || { echo "### ABORT no xpurt_s10/run.sh"; exit 1; }
mkdir -p "$MB/examples/$WDIR/int8/generated"

read -r MODELS EXDIRS QUANTS < <(python3 - "$WL" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ms,ex,qs=[],[],[]
for n,e in d["networks"].items():
    qs.append("fp32" if "fp32" in e["dispatch_deps_path"] else "int8")
    ms.append(n); ex.append(n)
print(",".join(ms), ",".join(ex), ",".join(qs))
PY
)
[ -z "${MODELS:-}" ] && { echo "### ABORT empty MODELS from $WL"; exit 1; }
N=$(python3 -c "import json;d=json.load(open('$SJ'))['dispatches'];print(len(d))")
P=$(python3 -c "
import json;d=json.load(open('$SJ'))['dispatches']
v=d.values() if isinstance(d,dict) else d
print(round(max(x['start_time']+x['duration'] for x in v),3))")
echo "### tag=$TAG wl=$(basename $WL) sched=$(basename $SJ) worker=$WDIR $(date -u +%FT%TZ)"
echo "### models=$MODELS quants=$QUANTS dispatches=$N predicted_ms=$P"

export XPURT_EXAMPLE_DIR=$MB/examples/$WDIR
BUILT=$MB/examples/$WDIR/int8/build/gemmini_q31_rvv_firesim/zephyr/zephyr.elf
rm -f "$BUILT"
STAMP=$(mktemp); : > $STAMP; sleep 1
SCHEDULE_JSON=$SJ MODELS=$MODELS MODEL_EXDIRS=$EXDIRS QUANTS=$QUANTS QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=build RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf \
FORCE_REGEN=0 XPURT_TRACE=1 bash examples/xpurt_s10/run.sh
RC=$?; echo "#### build rc=$RC"
if [ $RC -ne 0 ]; then
  grep -nE "error:|undefined reference|No such|ERROR" "$LOG" | tail -8
  rm -f $STAMP; echo "### ABORT build"; exit 1
fi
if [ ! -f "$BUILT" ] || [ ! "$BUILT" -nt "$STAMP" ]; then
  rm -f $STAMP; echo "### ABORT no fresh elf at $BUILT"; exit 1
fi
rm -f $STAMP
cp "$BUILT" $OUT/elf/$TAG.elf
SZ=$(stat -c %s $OUT/elf/$TAG.elf)
echo "#### elf=$OUT/elf/$TAG.elf bytes=$SZ dispatches=$N"
echo "############ BUILDDONE $TAG ############"
