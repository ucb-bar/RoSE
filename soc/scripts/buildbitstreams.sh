#!/bin/bash
ROSE_DIR=$(pwd)
FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard

cd ${ROSE_DIR}
yq -i ".builds_to_run = [\"firesim-rocket-singlecore-fp32gemmini-with-airsim-fast-no-nic-l2-llc4mb-ddr3\"]" ${ROSE_DIR}/soc/sim/config/config_build_local.yaml
firesim buildbitstream 

cd ${ROSE_DIR}
yq -i ".builds_to_run = [\"firesim-large-boom-fp32gemmini-dualcore-with-airsim-no-nic-l2-llc4mb-ddr3\"]" ${ROSE_DIR}/soc/sim/config/config_build_local.yaml
firesim buildbitstream 