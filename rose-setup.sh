#!/bin/bash
PROJECT_ROOT="/scratch/$(whoami)"
FIRESIM_DIR=${PROJECT_ROOT}/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
ROSE_DIR=${PROJECT_ROOT}/RoSE
SCALA_DIR=${ROSE_DIR}/soc/src/main/scala
FSIM_CC_DIR=${ROSE_DIR}/soc/src/main/cc

cd ${ROSE_DIR}
if [ ! -d ./build/ ]; then
    mkdir -p ./build/
fi
cd ./build/
if [ ! -d ./cmake-3.26.0-linux-x86_64/ ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.26.0/cmake-3.26.0-linux-x86_64.tar.gz
    tar -xvf cmake-3.26.0-linux-x86_64.tar.gz
fi
if [ ! -d ./yq_latest/ ]; then
    mkdir -p ./yq_latest
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O ./yq_latest/yq
    chmod +x ./yq_latest/yq
fi
echo Sourcing Deps
cd ${FIRESIM_DIR}
source ${FIRESIM_DIR}/env.sh
source ${FIRESIM_DIR}/sourceme-f1-manager.sh
cd ${ROSE_DIR}/build/cmake-3.26.0-linux-x86_64/
export PATH=$(pwd)/bin/:$PATH
export PATH=${ROSE_DIR}/build/yq_latest/:$PATH
cd ${ROSE_DIR}



