#!/bin/bash
DIR=$(dirname "$(realpath "$0")")
echo $DIR
PROJECT_ROOT="${DIR}/../.."
echo "Project root: $PROJECT_ROOT"
ROSE_DIR=${PROJECT_ROOT}/RoSE
FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim
cd ${FIRESIM_DIR}
source ${FIRESIM_DIR}/sourceme-manager.sh --skip-ssh-setup
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/logs ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/logs
fi
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/img ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/img
fi
pip3 install msgpack-rpc-python
pip3 install airsim
cd ${ROSE_DIR}

if [ -z "$1" ]; then
    export AIRSIM_IP="localhost"
else
    export AIRSIM_IP=$1
fi