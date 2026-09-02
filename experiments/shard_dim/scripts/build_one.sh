#!/usr/bin/env bash
# Build ONE xpurt binary (model tree x arm) and stash the ELF. Builds share
# examples/xpurt_demo_armB, so callers must serialize these; submission is what
# parallelises over the four fq lanes.
#   build_one.sh <tag> <exdir> <E|P>
set -uo pipefail
TAG=$1; EX=$2; ARM=$3
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/shard_dim/results/oh; ELFD=$OUT/elf; mkdir -p $OUT $ELFD
exec > $OUT/build_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB
G=$MB/examples/$EX/int8/generated/graph.json
SJ=$R/experiments/shard_dim/ohsched/${TAG}.json
SLOT=$([ "$ARM" = "E" ] && echo "CPU_E#0" || echo "CPU_P#0")
python3 $R/experiments/shard_dim/scripts/mk_serial_sched.py $SJ --graph $G --model dronet --slot $SLOT
NDISP=$(python3 -c "import json;print(len(json.load(open('$SJ'))['dispatches']))")
echo "### tag=$TAG exdir=$EX arm=$ARM slot=$SLOT dispatches=$NDISP $(date -u +%FT%TZ)"
# Kernel-pick gate: the conv algorithm must be the HARDWARE im2col on gemmini.
python3 - <<PY
import json
p="$MB/examples/$EX/int8/generated/gemmini_q31/kernel_picks.json"
a=json.load(open(p))["picks"]["conv2d_s8"]["algorithm"]
print(f"#### gemmini conv2d_s8 pick={a}")
assert a=="gemmini_tiled_conv", f"WRONG PICK {a} -- MB_DRIFT_ATOL fallback"
PY
[ $? -ne 0 ] && { echo "#### ABORT bad kernel pick"; exit 1; }
find $MB/examples -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
SCHEDULE_JSON=$SJ MODELS=dronet MODEL_EXDIRS=$EX QUANTS=int8 QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=build RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf FORCE_REGEN=0 XPURT_TRACE=1 \
  bash examples/xpurt_demo_armB/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|undefined reference" $OUT/build_$TAG.log | tail -8; echo "#### ABORT build"; exit 1; }
ELF=$(find $MB/examples -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "#### ABORT no fresh elf"; exit 1; }
# Two gates for the two ways this binary can be built for the WRONG machine and
# still link. Both were hit once, together, and both are silent: the hart-count
# mismatch hangs in Zephyr's SMP spinwait before the boot banner (no output at
# all past "Commencing simulation"), and a registry whose slot names the
# schedule does not use yields a walker with no workers.
grep -q "Merged configuration '.*firesim_chipyard_quad_hetero_q31.conf'" $OUT/build_$TAG.log || {
    echo "#### ABORT wrong Zephyr overlay -- the quad bitstream needs MP_MAX_NUM_CPUS=4"; exit 1; }
grep -q "pool_sizes: \[0, 0\]" $OUT/build_$TAG.log && {
    echo "#### ABORT pool_sizes [0,0] -- registry does not resolve the schedule's slots"; exit 1; }
echo "#### gates ok: $(grep -o "pool_sizes: \[[0-9, ]*\]" $OUT/build_$TAG.log | tail -1)"
cp "$ELF" $ELFD/$TAG.elf
echo "#### stashed $ELFD/$TAG.elf ndisp=$NDISP mtime=$(date -r $ELFD/$TAG.elf -u +%FT%TZ)"
echo "############ BUILDDONE $TAG ############"
