#!/usr/bin/env bash
# THE FULL SHARDED-vs-UNSHARDED WORKLOAD SWEEP.
#
#   run_wl_sweep.sh [solver]        default greedy
#
# For every generated workload JSON (family x machine pair) this runs TWO arms
# on the same schedule slots:
#   base   the unsplit graphs -- whole ops placed on harts
#   shard  every model replaced by a SPLIT tree (conv axis from the measured
#          best-axis search, plus the E/C pointwise and pool axes), so the
#          scheduler places TILES
# Both arms share a build recipe, which is what makes the pair comparable.
#
# Split policy per network, from the measured best-axis work:
#   dronet      gemmini prefers OC (9/10 convs); rvv is mixed, OH on 3x3
#   yolov8_nano gemmini prefers OH (12/12 of the split convs, 1.21-2.24x)
# plus E (pointwise) and C (pool) tiles above the ~5 us launch floor, chosen by
# mk_ec_args.py from each model's own measured single-hart profile.
set -uo pipefail
SOLVER=${1:-greedy}
R=/scratch/dima/rose-infra/RoSE; W=$R/experiments/workload_gen
S=$R/experiments/shard_dim/scripts; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
XR=$R/soc/sw/xpu-rt; OUT=$R/experiments/wl_sweep; mkdir -p $OUT/logs $OUT/elf
set +u; source $ZCS/scripts/activate_conda.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
PY=/scratch2/dima/miniforge3/envs/xpurt/bin/python

# The model list is DERIVED from the workloads rather than read from a
# hand-made /tmp file. It used to be the latter, and nothing in the tree ever
# wrote it: the sweep depended on a temp file that vanishes on reboot, and it
# went stale silently whenever a family gained a network -- the new model got
# no split tree and no profile, then scheduled against nothing.
MODELS_TXT=$OUT/wl_models.txt
$PY - "$XR/data/toplevel/wl_sweep" > $MODELS_TXT <<'MODLIST'
import json, glob, os, sys
names = set()
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    names |= set(json.load(open(f)).get("networks", {}))
print("\n".join(sorted(names)))
MODLIST
echo "  models named by the workloads: $(wc -l < $MODELS_TXT)"

echo "########## STEP 1: split trees for every model the workloads name ##########"
cd $MB
for M in $(cat $MODELS_TXT); do
  [ "$M" = vint ] && continue
  case $M in mlp_control*) Q=fp32;; *) Q=int8;; esac
  [ -f $MB/examples/${M}_wls/$Q/generated/graph.json ] && { echo "  ${M}_wls exists"; continue; }
  EC=$($PY $S/mk_ec_args.py $M $M $Q --pair hetero 2>/dev/null)
  CONV=$($PY - "$M" "$Q" <<'PY'
import json,sys
m,q=sys.argv[1],sys.argv[2]
MB="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
g=json.load(open(f"{MB}/examples/{m}/{q}/generated/graph.json"))
# measured policy: yolo's gemmini kernel (software im2col) wants OH; dronet's
# (hardware im2col + zero-copy OH window) wants OC. Everything else: OC.
axis = "OH" if m.startswith("yolov8") else "OC"
out=[]
for o in g["ops"]:
    d=o.get("dispatch_id"); s=o.get("shape") or {}
    if d is None or not o["op"].startswith("conv2d"): continue
    if axis=="OC":
        t=int(s.get("OC",0))
    else:
        t=int(s.get("OH") or 0) or ((int(s["IH"])+2*int(s["PH"])-int(s["KH"]))//int(s["SH"])+1)
    if t>=4 and t%2==0: out.append(f"{d}:{axis}:{t//2},{t//2}")
print(" ".join(out))
PY
)
  if [ -z "$CONV$EC" ]; then echo "  $M: nothing splittable, reusing unsplit"; continue; fi
  $PY $S/mk_split.py $M $Q ${M}_wls $CONV $EC >/dev/null 2>&1 \
    && echo "  ${M}_wls  conv=$(wc -w <<<"$CONV") ops, E/C=$(wc -w <<<"$EC") ops" \
    || echo "  ${M}_wls  MK_SPLIT FAILED"
  # mk_split copies the tree verbatim, so the split tree still calls itself
  # "$M" -- in graph.json AND in every generated C symbol. Give it its own id:
  # emit_dispatch_graph keys gen/vmfb on the graph name (so without this the
  # _wls emit lands on top of the UNSPLIT model's dispatch graph and both arms
  # schedule the split graph), and generate_xpurt_main derives the C symbol
  # prefix from the SCHEDULE's network name (so the two have to agree).
  $PY $W/rename_model_id.py $MB/examples/${M}_wls/$Q/generated ${M}_wls $M
done

echo "########## STEP 2: dispatch graphs + profiles for the split trees ##########"
cd $XR
for M in $(cat $MODELS_TXT); do
  [ "$M" = vint ] && continue
  case $M in mlp_control*) Q=fp32;; *) Q=int8;; esac
  G=$MB/examples/${M}_wls/$Q/generated/graph.json
  [ -f "$G" ] || continue
  for HW in gemmini_q31 V256D128_rvv; do
    $PY -m modelblaster.pipeline.emit_dispatch_graph --ir $G --out-root $ZCS/gen/vmfb \
       --target firesim_f2_rocket_saturn --hw $HW >/dev/null 2>&1
  done
  # tile costs: single-hart run of the SPLIT tree, same method as the per-network sweep
  for AP in serialE serialP; do
    T=${M}_wls_${AP}_base
    [ -d $R/experiments/sweep3net/res_$T ] && continue
    SWEEP_FORCE_SERIAL_COSTS=1 bash $S/sweep_pair.sh ${M}_wls ${M}_wls $Q $AP base >/dev/null 2>&1
    [ "$(grep -oE 'BUILDDONE' $R/experiments/sweep3net/logs/$T.log 2>/dev/null|tail -1)" = BUILDDONE ] && {
      rm -rf $R/experiments/sweep3net/res_$T; bash $S/sweep_submit.sh $T >/dev/null 2>&1; }
    UL=$(find $R/experiments/sweep3net/res_$T -name uartlog 2>/dev/null|head -1)
    [ -n "$UL" ] && { BE=$([ $AP = serialP ] && echo gemmini_q31 || echo V256D128_rvv)
      PYTHONPATH=$ZCS $PY scripts/uartlog_to_profile.py --uartlog "$UL" --model ${M}_wls \
        --quant $Q --backend $BE --cpu firesim_f2_rocket_saturn --tag ${M}_wls \
        --clock-mhz 1000 --out-root gen/profile >/dev/null 2>&1; }
  done
  echo "  ${M}_wls profiled"
done

echo "########## STEP 3: sharded workload JSONs ##########"
$PY - <<'PY'
import json,glob,os
XR="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"
D=f"{XR}/data/toplevel"
os.makedirs(f"{D}/wl_sweep_shard",exist_ok=True)
n=0
for f in glob.glob(f"{D}/wl_sweep/*.json"):
    d=json.load(open(f)); nets={}; ren={}
    for name,e in d["networks"].items():
        if name=="vint": nets[name]=e; continue      # vint stays unsplit for now
        e=dict(e)
        dp=e["dispatch_deps_path"].replace(f"/{name}/",f"/{name}_wls/").replace(f"/{name}.",f"/{name}_wls.")
        # dispatch_deps_path is stored RELATIVE to xpu-rt. Resolving it against
        # the cwd only worked because STEP 2 happens to leave us in $XR; from
        # anywhere else EVERY existence check misses, every network reads as
        # "nothing splittable", and all 44 shard JSONs silently revert to the
        # unsplit models -- deleting the shard arm with no error at all.
        if not os.path.exists(os.path.join(XR, dp)):
            # nothing splittable above the launch floor -- this model runs
            # unsplit in BOTH arms rather than silently pointing at nothing.
            nets[name]=e; continue
        e["identifier"]=f"{name}_wls"; e["dispatch_deps_path"]=dp
        nets[f"{name}_wls"]=e; ren[name]=f"{name}_wls"
    d["networks"]=nets
    # Edge endpoints name networks, so they have to follow the rename above.
    # workload_factory drops an edge whose endpoint is not a network key --
    # silently, with a bare `continue` -- so leaving these unmapped would give
    # the sharded arm NO dependency at all while the base arm kept it, and the
    # two arms would no longer be the same scheduling problem.
    if d.get("edges"):
        d["edges"]=[{"from":ren.get(x["from"],x["from"]),
                     "to":ren.get(x["to"],x["to"])} for x in d["edges"]]
    # idempotent: STEP 3 is re-run whenever a model gains a split tree, and
    # an unguarded prepend stacks "SHARDED arm. " once per run.
    if not d["_comment"].startswith("SHARDED arm."):
        d["_comment"]="SHARDED arm. "+d["_comment"]
    json.dump(d,open(f"{D}/wl_sweep_shard/{os.path.basename(f)}","w"),indent=1); n+=1
print(f"  wrote {n} sharded workload JSONs")
PY

echo "########## STEP 4: schedule + build + run BOTH arms ##########"
cd $XR
for ARM in base shard; do
  SUB=$([ $ARM = base ] && echo wl_sweep || echo wl_sweep_shard)
  for WL in $XR/data/toplevel/$SUB/*.json; do
    B=$(basename $WL .json)
    T=wl_${B#networks_}_${SOLVER}_${ARM}
    # A sweep this long will lose cells to a build bug or a queue hiccup, and
    # re-running the ones that already landed costs FPGA hours. WL_RESUME=1
    # keeps every completed cell and retries only the rest.
    #
    # "Completed" is a VALID uartlog, not merely a directory. A crashed or
    # timed-out run still leaves res_$T behind holding a uartlog with no trace
    # rows, and fq copies from the run host's sim_slot_*/ which survives
    # between jobs, so a cell can also collect the PREVIOUS job's log. Resuming
    # on directory existence alone therefore silently locks in exactly the
    # cells that failed -- five quad cells sat "done" that way with zero rows.
    # Same gate the results are read with: PASSED + a non-empty 14-field
    # dispatch trace + an embedded schedule= tag matching this cell.
    if [ "${WL_RESUME:-0}" = 1 ] && [ -d $OUT/res_$T ]; then
      if $PY - "$OUT/res_$T" "$T" <<'VALID'
import sys, glob, re
d, tag = sys.argv[1], sys.argv[2]
u = glob.glob(d + "/**/uartlog", recursive=True)
if not u: sys.exit(1)
t = open(u[0], errors="replace").read()
m = re.search(r"xpurt-runner: schedule=(\S+)", t)
rows, seen = 0, False
for ln in t.splitlines():
    if ln.startswith("entry_id,network,"): seen = True; continue
    if seen:
        f = ln.split(",")
        if len(f) == 14 and f[0].strip().isdigit(): rows += 1
sys.exit(0 if ("*** PASSED ***" in t and rows > 0 and m and m.group(1) == tag) else 1)
VALID
      then printf "  %-52s %s\n" "$T" "kept"; continue
      else printf "  %-52s %s\n" "$T" "stale/failed result -- rerunning"; fi
    fi
    $PY scripts/run_xpurt_schedule.py --networks-json "$WL" --solver $SOLVER \
        > $OUT/logs/sched_${B}_${ARM}.log 2>&1
    SJ=$XR/schedules/scheduled_${B}_${SOLVER}_profiled.json
    [ -f "$SJ" ] || { echo "  ${B}/${ARM}: SCHEDULE FAILED"; continue; }
    bash $W/build_workload.sh "$WL" "$SOLVER" "$ARM"
    ST=$(grep -oE 'BUILDDONE|### ABORT.*' $OUT/logs/build_$T.log 2>/dev/null|tail -1)
    if [ "$ST" = BUILDDONE ]; then
      while [ "$(jobs -rp|wc -l)" -ge 4 ]; do sleep 20; done
      rm -rf $OUT/res_$T; bash $W/submit_workload.sh $T & 
    fi
    printf "  %-52s %s\n" "$T" "${ST:-nobuild}"
  done
done
wait
echo "########## WL SWEEP COMPLETE ##########"
