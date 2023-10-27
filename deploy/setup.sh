#!/bin/bash
ROSE_DIR=$(pwd)/..
FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
pip3 install msgpack-rpc-python
pip3 install airsim
pip3 install gymnasium[mujoco]

cd ${ROSE_DIR}

if [ -z "$1" ]; then
    export AIRSIM_IP="localhost"
else
    export AIRSIM_IP=$1
fi
