#!/usr/bin/env bash
# One cell of the 2-backend sharding sweep: build + submit ONE (network, machine
# pair, arm) and collect it.
#
#   sweep_pair.sh <net> <src_exdir> <quant> <pair> <arm> [split args...]
#
#     pair = rvvpair | gempair | hetero
#     arm  = base   -> the UNSPLIT tree, same slots (the honest baseline)
#            shard  -> apply the split args, then schedule over the same slots
#            <other> -> same as shard, into its own exdir. Added so a second
#                       partition of the SAME network and pair can be measured
#                       against the first without either overwriting the
#                       other's tree: `shardec` is `shard` plus the pointwise
#                       (E) and pool-channel (C) splits.
#
# Both arms of a pair run the SAME schedule slots, so the only difference is the
# partition. That matters: a sharded number is only meaningful against a baseline
# on the same machine, and this campaign has already been bitten once by
# differencing against a build that carried a tax the baseline did not.
#
# Example (reproduces the dronet rvv-pair best-axis cell):
#   sweep_pair.sh dronet dronet_armB int8 rvvpair shard \
#       0:OH:28,28 3:OH:7,7 4:OH:7,7 8:OC:32,32 9:OH:3,4 10:OC:32,32 \
#       13:OH:2,2 14:OH:2,2 15:OC:64,64
set -uo pipefail
NET=$1; SRC=$2; QUANT=$3; PAIR=$4; ARM=$5; shift 5
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
S=$R/experiments/shard_dim/scripts; OUT=$R/experiments/sweep3net; mkdir -p $OUT/logs
TAG="${NET}_${PAIR}_${ARM}"
exec > $OUT/logs/$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB

# serialE / serialP pin EVERY dispatch to one hart. They are not a machine pair
# under test -- they are how a network that has no measured cost table gets one:
# between the two runs, every dispatch has a cost on both backends, taken from
# this hardware rather than from a model.
SERIAL=""
case $PAIR in
  rvvpair) SLOTS="CPU_E#0=rvv,CPU_E#1=rvv" ;;
  gempair) SLOTS="CPU_P#0=gemmini_q31,CPU_P#1=gemmini_q31" ;;
  hetero)  SLOTS="CPU_P#0=gemmini_q31,CPU_E#0=rvv" ;;
  serialE) SLOTS="CPU_E#0=rvv";          SERIAL="CPU_E#0" ;;
  serialP) SLOTS="CPU_P#0=gemmini_q31";  SERIAL="CPU_P#0" ;;
  *) echo "### ABORT unknown pair $PAIR"; exit 1 ;;
esac
echo "### tag=$TAG net=$NET src=$SRC quant=$QUANT pair=$PAIR arm=$ARM"
echo "### slots=$SLOTS  split_args=$*  $(date -u +%FT%TZ)"

if [ "$ARM" != "base" ] && [ $# -gt 0 ]; then
    EX="${NET}_sw_${PAIR}"
    [ "$ARM" != "shard" ] && EX="${EX}_${ARM}"
    python3 $S/mk_split.py "$SRC" "$QUANT" "$EX" "$@" || { echo "### ABORT mk_split"; exit 1; }
else
    EX="$SRC"          # baseline runs the source tree untouched
fi
G=$MB/examples/$EX/$QUANT/generated/graph.json
[ -f "$G" ] || { echo "### ABORT no graph at $G"; exit 1; }

SJ=$R/experiments/shard_dim/ohsched/${TAG}.json
if [ -n "$SERIAL" ]; then
    # Cost-discovery run: no cost table needed, and none is used.
    python3 $S/mk_serial_sched.py $SJ --graph $G --model $NET --slot "$SERIAL" \
      || { echo "### ABORT mk_serial_sched"; exit 1; }
else
    # Costs: the measured-cell table where one exists (dronet), else the two
    # serial runs above.
    # SWEEP_FORCE_SERIAL_COSTS=1 skips the measured-cell table. Needed whenever
    # the kernels have changed under it: dronet's table was captured before the
    # curated-coverage close, and batchnorm2d_s8 alone got 3.28x faster there.
    if [ "${SWEEP_FORCE_SERIAL_COSTS:-0}" = "1" ] \
       || ! python3 $S/mk_costs_synth.py /tmp/costs_$TAG.json --graph $G 2>/dev/null; then
        echo "### note: no measured-cell table for $NET; using the serial runs"
        python3 $S/mk_costs.py /tmp/unsplit_costs_$NET.json \
          $OUT/res_${NET}_serialE_base $OUT/res_${NET}_serialP_base \
          || { echo "### ABORT no costs -- run the serialE/serialP arms first"; exit 1; }
        # Those costs are keyed by the UNSPLIT graph's dispatch ids, and
        # apply_split_hint renumbers ids. Project them by NAME onto this graph.
        python3 $S/mk_costs_split.py /tmp/costs_$TAG.json --graph $G \
          --costs /tmp/unsplit_costs_$NET.json \
          --unsplit-graph $MB/examples/$SRC/$QUANT/generated/graph.json \
          || { echo "### ABORT mk_costs_split"; exit 1; }
    fi
    python3 $S/mk_par_sched.py $SJ --graph $G --model $NET --costs /tmp/costs_$TAG.json --slots "$SLOTS" \
      || { echo "### ABORT mk_par_sched"; exit 1; }
fi
NDISP=$(python3 -c "import json;print(len(json.load(open('$SJ'))['dispatches']))")
PRED=$(python3 -c "
import json;d=json.load(open('$SJ'))['dispatches']
print(round(max(v['start_time']+v['duration'] for v in d.values())*1e6,1))")
echo "#### dispatches=$NDISP predicted_us=$PRED"

find $MB/examples -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
SCHEDULE_JSON=$SJ MODELS=$NET MODEL_EXDIRS=$EX QUANTS=$QUANT QUANT=$QUANT \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=build RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf \
FORCE_REGEN=0 XPURT_TRACE=1 bash examples/xpurt_demo_armB/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|undefined reference" $OUT/logs/$TAG.log|tail -6; echo "### ABORT build"; exit 1; }
ELF=$(find $MB/examples -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "### ABORT no fresh elf"; exit 1; }
grep -q "Merged configuration '.*firesim_chipyard_quad_hetero_q31.conf'" $OUT/logs/$TAG.log \
  || { echo "### ABORT wrong Zephyr overlay"; exit 1; }
grep -q "pool_sizes: \[0, 0\]" $OUT/logs/$TAG.log && { echo "### ABORT pool_sizes [0,0]"; exit 1; }
mkdir -p $OUT/elf && cp "$ELF" $OUT/elf/$TAG.elf
echo "#### stashed $OUT/elf/$TAG.elf"
echo "############ BUILDDONE $TAG ############"
