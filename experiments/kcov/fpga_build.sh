#!/usr/bin/env bash
# Build ONE single-model FireSim ELF for the curated-kernel A/B and stash it.
#   fpga_build.sh <tag> <exdir> <quant> <curated_dir|"">
# Serialise calls: both arms share examples/<exdir>/<quant>/build/gemmini_q31_firesim.
set -uo pipefail
TAG=$1; EX=$2; Q=$3; CUR=$4
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/kcov; ELFD=$OUT/elf; mkdir -p $ELFD
exec > $OUT/logs/fpgabuild_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS
cd $MB
echo "### build $TAG exdir=$EX curated=${CUR:-<none/reference>} $(date -u +%FT%TZ)"
find $MB/examples/$EX -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1
GLOBAL_CURATED_DIR="$CUR" MODELBLASTER_CURATED_VERIFY=0 \
TARGET=gemmini_q31 QUANT=$Q RUNNER=firesim \
FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf STOP_AFTER=build \
  bash examples/$EX/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|undefined reference" $OUT/logs/fpgabuild_$TAG.log | tail -8; exit 1; }
grep -q "firesim_chipyard_quad_hetero_q31.conf" $OUT/logs/fpgabuild_$TAG.log || {
    echo "#### ABORT wrong Zephyr overlay"; exit 1; }
ELF=$(find $MB/examples/$EX -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "#### ABORT no fresh elf"; exit 1; }
cp "$ELF" $ELFD/$TAG.elf
python3 -c "
import json,sys
p='$MB/examples/$EX/$Q/generated/gemmini_q31/kernel_picks.json'
d=json.load(open(p))['picks']
print('#### picks:', {k:(v['algorithm'] or 'REFERENCE') for k,v in d.items()})
json.dump(d, open('$OUT/elf/$TAG.picks.json','w'), indent=1)
"
echo "#### stashed $ELFD/$TAG.elf $(ls -l $ELFD/$TAG.elf | awk '{print $5}') bytes"
echo "############ BUILDDONE $TAG ############"
