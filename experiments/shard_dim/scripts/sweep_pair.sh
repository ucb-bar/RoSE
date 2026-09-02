#!/usr/bin/env bash
# One cell of the 2-backend sharding sweep: build + submit ONE (network, machine
# pair, arm) and collect it.
#
#   sweep_pair.sh <net> <src_exdir> <quant> <pair> <arm> [split args...]
#
#     pair = rvvpair | gempair | hetero
#     arm  = base   -> the UNSPLIT tree, same slots (the honest baseline)
#            shard  -> apply the split args, then schedule over the same slots
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

case $PAIR in
  rvvpair) SLOTS="CPU_E#0=rvv,CPU_E#1=rvv" ;;
  gempair) SLOTS="CPU_P#0=gemmini_q31,CPU_P#1=gemmini_q31" ;;
  hetero)  SLOTS="CPU_P#0=gemmini_q31,CPU_E#0=rvv" ;;
  *) echo "### ABORT unknown pair $PAIR"; exit 1 ;;
esac
echo "### tag=$TAG net=$NET src=$SRC quant=$QUANT pair=$PAIR arm=$ARM"
echo "### slots=$SLOTS  split_args=$*  $(date -u +%FT%TZ)"

if [ "$ARM" = "shard" ] && [ $# -gt 0 ]; then
    EX="${NET}_sw_${PAIR}"
    python3 $S/mk_split.py "$SRC" "$QUANT" "$EX" "$@" || { echo "### ABORT mk_split"; exit 1; }
else
    EX="$SRC"          # baseline runs the source tree untouched
fi
G=$MB/examples/$EX/$QUANT/generated/graph.json
[ -f "$G" ] || { echo "### ABORT no graph at $G"; exit 1; }

# Costs: the measured-cell table where it exists (dronet), else the serial runs.
python3 $S/mk_costs_synth.py /tmp/costs_$TAG.json --graph $G 2>/dev/null \
  || { echo "### note: no measured-cell table for $NET, falling back to mk_costs"; \
       python3 $S/mk_costs.py /tmp/costs_$TAG.json \
         $R/experiments/shard_dim/results/serial/${NET}_E $R/experiments/shard_dim/results/serial/${NET}_P \
         || { echo "### ABORT no costs"; exit 1; }; }

SJ=$R/experiments/shard_dim/ohsched/${TAG}.json
python3 $S/mk_par_sched.py $SJ --graph $G --model $NET --costs /tmp/costs_$TAG.json --slots "$SLOTS" \
  || { echo "### ABORT mk_par_sched"; exit 1; }
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
