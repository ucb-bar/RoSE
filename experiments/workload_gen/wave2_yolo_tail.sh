#!/usr/bin/env bash
# Wave 2 of the workload sweep: fold yolov8_nano sf/sg/sh (224/256/320) in, so
# the two families gated on them can be generated.
#
#   perception_heavy  one large detector + a light control loop  (needs sf)
#   bimodal           extreme period ratio, slow model + 1 ms loop (needs sh)
#
# MUST NOT run while run_wl_sweep.sh is building: both drive the shared
# examples/xpurt_demo_armB build tree.  Run it after the current sweep exits.
set -uo pipefail
R=/scratch/dima/rose-infra/RoSE; W=$R/experiments/workload_gen
S=$R/experiments/shard_dim/scripts; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw
MB=$ZCS/modelblaster; XR=$R/soc/sw/xpu-rt
set +u; source $ZCS/scripts/activate_conda.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
PY=/scratch2/dima/miniforge3/envs/xpurt/bin/python
RUNGS="yolov8_nano_sf yolov8_nano_sg yolov8_nano_sh"

profile_one() {   # <model> <quant> -- single-hart cost table on both backends
  local M=$1 Q=$2
  for AP in serialE serialP; do
    local T=${M}_${AP}_base
    [ -d $R/experiments/sweep3net/res_$T ] && { echo "    $T exists"; continue; }
    SWEEP_FORCE_SERIAL_COSTS=1 bash $S/sweep_pair.sh $M $M $Q $AP base >/dev/null 2>&1
    [ "$(grep -oE 'BUILDDONE' $R/experiments/sweep3net/logs/$T.log 2>/dev/null|tail -1)" = BUILDDONE ] || {
      echo "    $T BUILD FAILED"; continue; }
    rm -rf $R/experiments/sweep3net/res_$T; bash $S/sweep_submit.sh $T >/dev/null 2>&1
    local UL=$(find $R/experiments/sweep3net/res_$T -name uartlog 2>/dev/null|head -1)
    [ -n "$UL" ] || { echo "    $T NO UARTLOG"; continue; }
    local BE=$([ $AP = serialP ] && echo gemmini_q31 || echo V256D128_rvv)
    (cd $XR && PYTHONPATH=$ZCS $PY scripts/uartlog_to_profile.py --uartlog "$UL" --model $M \
       --quant $Q --backend $BE --cpu firesim_f2_rocket_saturn --tag $M \
       --clock-mhz 1000 --out-root gen/profile >/dev/null 2>&1)
    echo "    $T profiled"
  done
}

echo "########## W2 STEP 1: unsplit trees + profiles ##########"
for M in $RUNGS; do echo "  $M"; profile_one $M int8; done

echo "########## W2 STEP 2: split trees + profiles ##########"
cd $MB
for M in $RUNGS; do
  if [ ! -f $MB/examples/${M}_wls/int8/generated/graph.json ]; then
    EC=$($PY $S/mk_ec_args.py $M $M int8 --pair hetero 2>/dev/null)
    CONV=$($PY - "$M" <<'PY'
import json,sys
m=sys.argv[1]
MB="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
g=json.load(open(f"{MB}/examples/{m}/int8/generated/graph.json"))
out=[]                                   # yolo's gemmini kernel wants OH
for o in g["ops"]:
    d=o.get("dispatch_id"); s=o.get("shape") or {}
    if d is None or not o["op"].startswith("conv2d"): continue
    t=int(s.get("OH") or 0) or ((int(s["IH"])+2*int(s["PH"])-int(s["KH"]))//int(s["SH"])+1)
    if t>=4 and t%2==0: out.append(f"{d}:OH:{t//2},{t//2}")
print(" ".join(out))
PY
)
    $PY $S/mk_split.py $M int8 ${M}_wls $CONV $EC >/dev/null 2>&1 \
      && echo "  ${M}_wls  conv=$(wc -w <<<"$CONV") ops, E/C=$(wc -w <<<"$EC") ops" \
      || { echo "  ${M}_wls MK_SPLIT FAILED"; continue; }
  fi
  $PY $W/rename_model_id.py $MB/examples/${M}_wls/int8/generated ${M}_wls $M
  for HW in gemmini_q31 V256D128_rvv; do
    (cd $XR && $PY -m modelblaster.pipeline.emit_dispatch_graph \
       --ir $MB/examples/${M}_wls/int8/generated/graph.json --out-root $ZCS/gen/vmfb \
       --target firesim_f2_rocket_saturn --hw $HW >/dev/null 2>&1)
  done
  profile_one ${M}_wls int8
done

echo "########## W2 STEP 3: regenerate workloads (now 8 families) ##########"
$PY $W/mk_workloads.py --emit 2>&1 | tail -2
echo "########## W2 READY -- rerun run_wl_sweep.sh greedy for the new families ##########"
