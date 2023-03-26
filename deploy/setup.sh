#!/bin/bash
PROJECT_ROOT="/scratch/$(whoami)"
FIRESIM_DIR=${PROJECT_ROOT}/firesim
ROSE_DIR=${PROJECT_ROOT}/RoSE
source ${FIRESIM_DIR}/env.sh
source ${FIRESIM_DIR}/sourceme-f1-manager.sh
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/logs ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/logs
fi
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/img ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/img
fi
pip3 instal msgpack-rpc-python
pip3 install airsim