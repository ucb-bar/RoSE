#!/bin/bash
PROJECT_ROOT='/home/centos'
FIRESIM_DIR=${PROJECT_ROOT}/firesim
ROSE_DIR=${PROJECT_ROOT}/RoSE

cp ${ROSE_DIR}/soc/sim/config_runtime.yaml ${FIRESIM_DIR}/deploy
