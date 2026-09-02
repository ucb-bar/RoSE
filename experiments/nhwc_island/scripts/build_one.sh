#!/usr/bin/env bash
# Build ONE xpurt binary (model tree, all-gemmini serial schedule) and stash the
# ELF. Modelled on experiments/shard_dim/scripts/build_one.sh, with three
# deliberate differences:
#   * its own demo tree (examples/xpurt_demo_nhwc) so it cannot collide with a
#     concurrent shard sweep that is building through xpurt_demo_armB;
#   * MB_DRIFT_ATOL is NOT set -- kernels.c is staged by mk_tree.py, so nothing
#     here re-picks a kernel, and setting a drift tolerance would quietly relax
#     the bit-exactness requirement the NHWC island is being judged on;
#   * the freshness gate looks only inside this experiment's own example trees.
#   build_one.sh <tag> <exdir> [run|spike] [P|E]
# The slot arg exists to exercise the SHIM: placing the island on CPU_E (rvv,
# no native nhwc kernel) is the "placement can only make it slower, never
# wrong" claim, and it is only a claim until a run says max_abs_err is the
# same number the gemmini placement produced.
set -uo pipefail
TAG=$1; EX=$2; MODE=${3:-run}; ARM=${4:-P}
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/nhwc_island/results; ELFD=$OUT/elf; mkdir -p $OUT $ELFD
exec > $OUT/build_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS
cd $MB
G=$MB/examples/$EX/int8/generated/graph.json
SJ=$R/experiments/nhwc_island/sched/${TAG}.json
if [ -n "${RELAYOUT_SLOT:-}" ]; then
  python3 $R/experiments/nhwc_island/scripts/mk_mixed_sched.py $SJ --graph $G --model dronet \
    --slot CPU_${ARM}#0 --on ${RELAYOUT_SLOT}:nchw_to_nhwc_s8,nhwc_to_nchw_s8
else
  python3 $R/experiments/shard_dim/scripts/mk_serial_sched.py $SJ --graph $G --model dronet --slot CPU_${ARM}#0
fi
NDISP=$(python3 -c "import json;print(len(json.load(open('$SJ'))['dispatches']))")
echo "### tag=$TAG exdir=$EX slot=CPU_${ARM}#0 dispatches=$NDISP $(date -u +%FT%TZ)"
python3 - <<PY
import json
p="$MB/examples/$EX/int8/generated/gemmini_q31/kernel_picks.json"
a=json.load(open(p))["picks"]
print("#### gemmini picks:", {k: v.get("algorithm") or v["source"] for k, v in sorted(a.items())})
conv=a["conv2d_s8"]["algorithm"]
assert "$ARM" != "P" or conv in ("gemmini_tiled_conv","gemmini_tiled_conv_nhwc"), f"WRONG CONV PICK {conv}"
PY
[ $? -ne 0 ] && { echo "#### ABORT bad kernel pick"; exit 1; }
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
find $MB/examples/xpurt_demo_nhwc -name zephyr.elf -delete 2>/dev/null
if [ "$MODE" = "spike" ]; then
  RUNNER=spike; RUNARGS="SPIKE_TIMEOUT=7200 SPIKE_BIN=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike"
  STOP=none
else
  RUNNER=firesim; RUNARGS=""; STOP=build
fi
env SCHEDULE_JSON=$SJ MODELS=dronet MODEL_EXDIRS=$EX QUANTS=int8 QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
STOP_AFTER=$STOP RUNNER=$RUNNER FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf \
FORCE_REGEN=0 XPURT_TRACE=1 $RUNARGS \
  bash examples/xpurt_demo_nhwc/run.sh
RC=$?; echo "#### build rc=$RC"
# In spike mode a non-zero rc is EXPECTED for this model: the in-binary compare
# runs at atol=0 and gemmini_tiled_conv's single-round mvout requantize drifts
# by up to 2 LSB (see the kernel header). What matters is the NUMBER, which both
# arms must report identically -- an NHWC island is a permutation, so one LSB of
# change would mean a real bug, not a tolerance question. So keep going if the
# run produced a max_abs_err line, and abort only on a genuine build failure.
if [ $RC -ne 0 ] && ! grep -q "max_abs_err" $OUT/build_$TAG.log; then
    grep -nE "error:|undefined reference" $OUT/build_$TAG.log | tail -12
    echo "#### ABORT build"; exit 1
fi
grep -o "max_abs_err=[0-9.]*[^,]*" $OUT/build_$TAG.log | tail -1 | sed "s/^/#### NUMERICS /"
ELF=$(find $MB/examples/xpurt_demo_nhwc -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "#### ABORT no fresh elf"; exit 1; }
grep -q "Merged configuration '.*firesim_chipyard_quad_hetero_q31.conf'" $OUT/build_$TAG.log || {
    [ "$MODE" = "spike" ] || { echo "#### ABORT wrong Zephyr overlay"; exit 1; }; }
grep -q "pool_sizes: \[0, 0\]" $OUT/build_$TAG.log && {
    echo "#### ABORT pool_sizes [0,0] -- registry does not resolve the schedule's slots"; exit 1; }
cp "$ELF" $ELFD/$TAG.elf
echo "#### stashed $ELFD/$TAG.elf ndisp=$NDISP"
echo "############ BUILDDONE $TAG ############"
