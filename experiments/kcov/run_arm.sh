#!/usr/bin/env bash
# Curated-kernel coverage: build+verify+profile ONE (model, target) arm on spike.
#   run_arm.sh <tag> <exdir> <target> <quant>
# Writes experiments/kcov/logs/<tag>.log and experiments/kcov/prof/<tag>/.
set -uo pipefail
TAG=$1; EX=$2; TGT=$3; Q=$4
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/kcov
mkdir -p $OUT/logs $OUT/prof
exec > $OUT/logs/$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS
cd $MB
echo "### tag=$TAG exdir=$EX target=$TGT quant=$Q $(date -u +%FT%TZ)"
echo "### env: MB_DRIFT_ATOL=${MB_DRIFT_ATOL:-<unset>} GLOBAL_CURATED_DIR=${GLOBAL_CURATED_DIR:-<default kernels/>}"
TARGET=$TGT QUANT=$Q RUNNER=${RUNNER:-spike} \
PROFILE_OUT_ROOT=$OUT/prof/$TAG PROFILE_BACKEND=$TGT \
  bash examples/$EX/run.sh
RC=$?
echo "### rc=$RC $(date -u +%FT%TZ)"
cp $MB/examples/$EX/$Q/generated/$TGT/kernel_picks.json $OUT/prof/$TAG.picks.json 2>/dev/null
echo "############ DONE $TAG rc=$RC ############"
