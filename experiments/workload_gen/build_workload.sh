#!/usr/bin/env bash
# Build ONE workload ELF from a generated workload JSON, sharded or not.
#   build_workload.sh <workload.json> <solver> <arm>      arm = base | shard
#
# Generalises build_3net.sh, which hardcodes MODELS/MODEL_EXDIRS/QUANTS for the
# fixed mlp+dronet+yolo bundle. Here the model list is READ OUT OF THE WORKLOAD
# JSON, so any generated family works -- including the ViNT ones, and any
# future mix -- without editing the builder.
#
# arm=shard points every network at its SPLIT example tree (<net>_wls) and at
# the dispatch graph emitted from that split IR, so the scheduler places TILES
# rather than whole ops. arm=base uses the unsplit trees. Both otherwise share
# a build recipe, which is what makes the pair comparable.
set -uo pipefail
# Resolve the workload path BEFORE any cd: the script cds into $MB, and a
# relative path handed in from the xpu-rt dir silently stops resolving there.
# That failure was not silent for long -- it emptied MODELS, and the codegen
# then rejected the schedule with "references unknown network 'dronet_sc';
# known: ['dronet','mlp_control']", i.e. it had fallen back to defaults.
WL=$(readlink -f "$1"); SOLVER=$2; ARM=${3:-base}
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/wl_sweep; mkdir -p $OUT/logs $OUT/elf
FAM=$(basename "$WL" .json | sed 's/^networks_//')
# Leading digits break the codegen's C header guard (#define 3NET_..._H), hence
# the wl_ prefix rather than the bare family name.
TAG="wl_${FAM}_${SOLVER}_${ARM}"
exec > $OUT/logs/build_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB

read -r MODELS EXDIRS QUANTS < <(python3 - "$WL" "$ARM" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); arm=sys.argv[2]
ms,ex,qs=[],[],[]
for n,e in d["networks"].items():
    q="fp32" if "fp32" in e["dispatch_deps_path"] else "int8"
    ms.append(n); qs.append(q)
    # The network name IS the example dir: step 3 renamed the sharded ones to
    # <net>_wls and left models with nothing splittable alone, so appending
    # here would ask for a tree that was deliberately never built.
    ex.append(n)
print(",".join(ms), ",".join(ex), ",".join(qs))
PY
)
echo "### tag=$TAG arm=$ARM models=$MODELS exdirs=$EXDIRS quants=$QUANTS $(date -u +%FT%TZ)"

SJ=$R/soc/sw/xpu-rt/schedules/scheduled_$(basename "$WL" .json)_${SOLVER}_profiled.json
[ -f "$SJ" ] || { echo "### ABORT no schedule at $SJ -- run run_xpurt_schedule.py first"; exit 1; }
N=$(python3 -c "import json;d=json.load(open('$SJ'))['dispatches'];print(len(d))")
P=$(python3 -c "
import json;d=json.load(open('$SJ'))['dispatches']
v=d.values() if isinstance(d,dict) else d
print(round(max(x['start_time']+x['duration'] for x in v),3))")
echo "#### dispatches=$N predicted_ms=$P"

find $MB/examples -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
SCHEDULE_JSON=$SJ MODELS=$MODELS MODEL_EXDIRS=$EXDIRS QUANTS=$QUANTS QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=build RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf \
FORCE_REGEN=0 XPURT_TRACE=1 bash examples/xpurt_demo_armB/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|undefined reference|No such" $OUT/logs/build_$TAG.log|tail -6; echo "### ABORT build"; exit 1; }
ELF=$(find $MB/examples -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "### ABORT no fresh elf"; exit 1; }
cp "$ELF" $OUT/elf/$TAG.elf
echo "############ BUILDDONE $TAG ############"
