#!/bin/bash

PROJECT_ROOT="/scratch/$(whoami)"/rose-parent-dir
ROSE_DIR=${PROJECT_ROOT}/RoSE

BOOM_CONFIG='firesim-dual-large-boom-fp32gemmini-singlecore-with-airsim-no-nic-l2-llc4mb-ddr3'
ROCKET_CONFIG='firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3'

START_Y='0.0'
END_X='-80.0'
END_CYCLE=40_000_000_000
AIRSIM_STEPS=1
FIRESIM_CYCLES=10_000
ANGLE=200.0
DNN='trail_resnet14_complex.onnx'
DNN_SMALL='trail_resnet6_complex.onnx'
VEL=9

cd ${ROSE_DIR}/deploy/hephaestus

BW=( 102400000 204800000 409600000 819200000 1638400000 )
for (( i=0; i<${BW}; i++ ));
do
    echo "Running RoSE simulation"
    echo ${BW[i]} > bw.txt
    python3 runner.py -r FireSim -a ${AIRSIM_STEPS} -f ${FIRESIM_CYCLES} -y ${START_Y} -c ${END_CYCLE} -x ${END_X} -l ${ROSE_DIR}/deploy/hephaestus/logs/lux-debug # | tee ${ROSE_DIR}/deploy/hephaestus/logs/tunnel-exp-rocket-gemmini-${ANGLE_NAMES[$i]}.log
    cp /scratch/$(whoami)/rose-parent-dir/sim_slot_0/uartlog /scratch/$(whoami)/rose-parent-dir/bw-sweep/bw-sweep-poll/bw-sweep-poll-uartlog-${BW[i]}
    pid=$!
    sleep 10
    kill $pid
    sleep 60
done
# firesim kill &
