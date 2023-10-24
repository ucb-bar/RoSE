#!/bin/bash
ROSE_DIR=$(pwd)
FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
SCALA_DIR=${ROSE_DIR}/soc/src/main/scala
FSIM_CC_DIR=${ROSE_DIR}/soc/src/main/cc

FIRESIM_RUNDIR=${ROSE_DIR}/soc/sim/firesim_run_temp/
FIRESIM_BUILDDIR=${ROSE_DIR}/soc/sim/bitstreams/



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
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/logs ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/logs
fi
if [ ! -d ${ROSE_DIR}/deploy/hephaestus/img ]; then
    mkdir -p ${ROSE_DIR}/deploy/hephaestus/img
fi
if [ ! -d ${ROSE_DIR}/deploy/figures ]; then
    mkdir -p ${ROSE_DIR}/deploy/figures
fi


echo Sourcing Deps
cd ${FIRESIM_DIR}
source ${FIRESIM_DIR}/env.sh
source ${FIRESIM_DIR}/sourceme-manager.sh --skip-ssh-setup
cd ${ROSE_DIR}/build/cmake-3.26.0-linux-x86_64/
export PATH=$(pwd)/bin/:$PATH
export PATH=${ROSE_DIR}/build/yq_latest/:$PATH
cd ${ROSE_DIR}

yq -i ".run_farm.recipe_arg_overrides.default_simulation_dir = \"${FIRESIM_RUNDIR}\"" ${ROSE_DIR}/soc/sim/config/config_runtime_local.yaml

yq -i ".build_farm.recipe_arg_overrides.default_build_dir = \"${FIRESIM_BUILDDIR}\"" ${ROSE_DIR}/soc/sim/config/config_build_local.yaml

sed -i "s|/bitstream_dir|${ROSE_DIR}/soc/sim/bitstreams|g" ${ROSE_DIR}/soc/sim/config/config_hwdb_local.yaml

cd ${ROSE_DIR}/
git submodule update --init ${ROSE_DIR}/soc/sim/firesim

yq -i ".build_farm.recipe_arg_overrides.default_build_dir = \"${FIRESIM_BUILDDIR}\"" ${ROSE_DIR}/soc/sim/config/config_build_local.yaml

sed -i "s|/bitstream_dir|${ROSE_DIR}/soc/sim/bitstreams|g" ${ROSE_DIR}/soc/sim/config/config_hwdb_local.yaml

cd ${ROSE_DIR}/
# git submodule update --init ${ROSE_DIR}/soc/sim/firesim

export PATH=$PATH:${ROSE_DIR}/


