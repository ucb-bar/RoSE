#!/usr/bin/env bash
# Build ONE end-to-end binary: a greedy list schedule over 2 hetero harts,
# durations taken from the measured serial runs of that same tree.
#   build_par.sh <tag> <exdir> <costs.json>
set -uo pipefail
TAG=$1; EX=$2; COSTS=$3
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/shard_dim/results/oh; ELFD=$OUT/elf; mkdir -p $OUT $ELFD
exec > $OUT/build_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB
G=$MB/examples/$EX/int8/generated/graph.json
SJ=$R/experiments/shard_dim/ohsched/${TAG}.json
# SLOTS lets one measured cost table be replanned for a different machine
# (default = the 1+1 hetero pair; "CPU_E#0=rvv,CPU_E#1=rvv" = an rvv PAIR).
python3 $R/experiments/shard_dim/scripts/mk_par_sched.py $SJ --graph $G --model dronet --costs $COSTS \
  ${SLOTS:+--slots "$SLOTS"}
NDISP=$(python3 -c "import json;print(len(json.load(open('$SJ'))['dispatches']))")
echo "### tag=$TAG exdir=$EX PARALLEL dispatches=$NDISP $(date -u +%FT%TZ)"
find $MB/examples -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
SCHEDULE_JSON=$SJ MODELS=dronet MODEL_EXDIRS=$EX QUANTS=int8 QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=build RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf FORCE_REGEN=0 XPURT_TRACE=1 \
  bash examples/xpurt_demo_armB/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|undefined reference" $OUT/build_$TAG.log|tail -8; echo "#### ABORT"; exit 1; }
ELF=$(find $MB/examples -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "#### ABORT no fresh elf"; exit 1; }
# Same two gates the serial path carries, for the same reason: the quad
# bitstream needs MP_MAX_NUM_CPUS=4 (a 2-hart overlay spinwaits forever in
# Zephyr's SMP boot with NO output), and a registry that does not resolve the
# schedule's slot names yields a walker with no workers.
grep -q "Merged configuration '.*firesim_chipyard_quad_hetero_q31.conf'" $OUT/build_$TAG.log || {
    echo "#### ABORT wrong Zephyr overlay"; exit 1; }
grep -q "pool_sizes: \[0, 0\]" $OUT/build_$TAG.log && {
    echo "#### ABORT pool_sizes [0,0] -- registry does not resolve the schedule slots"; exit 1; }
python3 - <<PY
import json
a=json.load(open("$MB/examples/$EX/int8/generated/gemmini_q31/kernel_picks.json"))["picks"]["conv2d_s8"]["algorithm"]
print(f"#### gemmini conv2d_s8 pick={a}")
assert a=="gemmini_tiled_conv", f"WRONG PICK {a}"
PY
[ $? -ne 0 ] && { echo "#### ABORT bad kernel pick"; exit 1; }
echo "#### gates ok: $(grep -o "pool_sizes: \[[0-9, ]*\]" $OUT/build_$TAG.log | tail -1)"
cp "$ELF" $ELFD/$TAG.elf
echo "#### stashed $ELFD/$TAG.elf ndisp=$NDISP"
echo "############ BUILDDONE $TAG ############"
