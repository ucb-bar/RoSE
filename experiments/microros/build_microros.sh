#!/usr/bin/env bash
# Build ONE micro-ROS baseline ELF for the F2 quad-hetero bitstream and stash it.
#   build_microros.sh <tag>
#
# The micro-ROS baseline is the fixed-pinning reference the XPU-RT scheduler is
# measured against: each network is one ROS node with its own rclc executor and
# rmw session, pinned to one hart, running its whole dispatch graph there.
# See modelblaster/examples/microros_demo/ROS_FLOW.md for the original flow
# (which targeted the 2-hart U250 Q31 bitstream and ran firesim locally).
#
# Port notes for f2_quad_hetero_norose_tacit_q31_60mhz:
#   harts 0,1 = Rocket + Q0.31 Gemmini, NO vector
#   harts 2,3 = Rocket + Saturn RVV,    NO Gemmini
# so Config B ("yolov8 alone; dronet+mlp share one hart") becomes
#   yolov8 -> hart 0 (gemmini_q31), dronet+mlp -> hart 2 (rvv), broker -> hart 3.
# On the old dual bitstream every tile had both units, so this is the one
# placement change the port requires.
set -uo pipefail
TAG=${1:-cfgB}
R=/scratch/dima/rose-infra/RoSE; ZCS=$R/soc/sw/xpu-rt/zephyr-chipyard-sw; MB=$ZCS/modelblaster
OUT=$R/experiments/microros; mkdir -p $OUT/logs $OUT/elf
exec > $OUT/logs/build_$TAG.log 2>&1
set +u; source $ZCS/scripts/activate_conda.sh; source $ZCS/scripts/set_envvars_sdk.sh; set -u
export PYTHONPATH=$ZCS MB_DRIFT_ATOL=2
cd $MB
echo "### tag=$TAG $(date -u +%FT%TZ)"

find $MB/examples -name zephyr.elf -delete 2>/dev/null
STAMP=$OUT/.stamp_$TAG; : > $STAMP; sleep 1

MODELS=yolov8_nano,dronet,mlp_control \
BACKENDS=gemmini_q31,rvv \
PIN_BACKENDS=gemmini_q31,rvv,rvv \
PIN_HARTS=${PIN_HARTS:-0,2,2} \
PERIODS_MS=${PERIODS_MS:-0,40,20} \
QUANTS=int8,int8,fp32 \
RUNNER=firesim FIRESIM_CONF=firesim_chipyard_quad_hetero_q31.conf \
MICROROS_BROKER_HART=${MICROROS_BROKER_HART:-3} \
FORCE_REGEN=0 STOP_AFTER=build \
MICROROS_NO_PUBLISH=${MICROROS_NO_PUBLISH:-1} \
MICROROS_2EXEC_BC=${MICROROS_2EXEC_BC:-1} \
MICROROS_2EXEC_FUSE_BC=${MICROROS_2EXEC_FUSE_BC:-1} \
  bash examples/microros_demo/run.sh
RC=$?; echo "#### build rc=$RC"
[ $RC -ne 0 ] && { grep -nE "error:|Error|undefined reference|No such" $OUT/logs/build_$TAG.log | tail -20; echo "### ABORT build"; exit 1; }

# Same two silent-wrong-machine gates build_one.sh uses: a hart-count mismatch
# hangs in Zephyr's SMP spinwait with no output at all past the boot banner.
grep -q "Merged configuration '.*firesim_chipyard_quad_hetero_q31.conf'" $OUT/logs/build_$TAG.log || {
    echo "#### ABORT wrong Zephyr overlay -- the quad bitstream needs MP_MAX_NUM_CPUS=4"; exit 1; }

ELF=$(find $MB/examples -name zephyr.elf -newer $STAMP 2>/dev/null | head -1)
[ -z "$ELF" ] && { echo "### ABORT no fresh elf"; exit 1; }
cp "$ELF" $OUT/elf/$TAG.elf
echo "#### stashed $OUT/elf/$TAG.elf size=$(stat -c%s $OUT/elf/$TAG.elf)"
echo "############ BUILDDONE $TAG ############"
