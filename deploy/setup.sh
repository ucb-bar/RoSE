#!/bin/bash
DIR=$(dirname "$(realpath "$0")")
echo $DIR
PROJECT_ROOT="${DIR}/../.."
echo "Project root: $PROJECT_ROOT"
FIRESIM_DIR=${PROJECT_ROOT}/firesim
ROSE_DIR=${PROJECT_ROOT}/RoSE
cd ${FIRESIM_DIR}
source ${FIRESIM_DIR}/sourceme-f1-manager.sh
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/logs ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/logs
fi
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/img ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/img
fi
pip3 install msgpack-rpc-python
pip3 install airsim
cd ${ROSE_DIR}