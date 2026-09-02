#!/usr/bin/env bash
# Numerical gate for a split tree: build the xpurt binary for ONE arm and run it
# under spike, which verifies every model output against the PyTorch golden.
#   spike_verify.sh <tag> <exdir> <E|P>
set -uo pipefail
TAG=$1; EX=$2; ARM=$3
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/shard_dim/results/oh; mkdir -p $OUT
exec > $OUT/spike_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB
G=$MB/examples/$EX/int8/generated/graph.json
SJ=$R/experiments/shard_dim/ohsched/${TAG}.json
SLOT=$([ "$ARM" = "E" ] && echo "CPU_E#0" || echo "CPU_P#0")
python3 $R/experiments/shard_dim/scripts/mk_serial_sched.py $SJ --graph $G --model dronet --slot $SLOT
echo "### tag=$TAG exdir=$EX arm=$ARM slot=$SLOT dispatches=$(python3 -c "import json;print(len(json.load(open('$SJ'))['dispatches']))")"
SCHEDULE_JSON=$SJ MODELS=dronet MODEL_EXDIRS=$EX QUANTS=int8 QUANT=int8 \
BACKENDS=gemmini_q31,rvv REGISTRY=$MB/cores/chipyard_quad_hetero_gemmini_q31.json \
CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv SCHED_NAME=$TAG \
RUNNER=spike FORCE_REGEN=0 XPURT_TRACE=1 SPIKE_TIMEOUT=3600 \
SPIKE_BIN=/scratch2/dima/chipyard-fsim/.conda-env/riscv-tools/bin/spike \
  bash examples/xpurt_demo_armB/run.sh
echo "#### rc=$?"
echo "############ SPIKEDONE $TAG ############"
