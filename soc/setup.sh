#!/bin/bash
PROJECT_ROOT="/scratch/$(whoami)"
FIRESIM_DIR=${PROJECT_ROOT}/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
ROSE_DIR=${PROJECT_ROOT}/RoSE
SCALA_DIR=${ROSE_DIR}/soc/src/main/scala
FSIM_CC_DIR=${ROSE_DIR}/soc/src/main/cc

# Copy scala files
cp ${SCALA_DIR}/AirSimIO.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/example/
cp ${SCALA_DIR}/IOBinders.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/
cp ${SCALA_DIR}/BridgeBinders.scala ${CHIPYARD_DIR}/generators/firechip/src/main/scala/
cp ${SCALA_DIR}/DigitalTop.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/
cp ${SCALA_DIR}/AbstractConfig.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/
cp ${SCALA_DIR}/AirSimBridge.scala  ${FIRESIM_DIR}/sim/firesim-lib/src/main/scala/bridges/
cp ${SCALA_DIR}/RoSEConfigs.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/

# Copy C++ files
cp ${FSIM_CC_DIR}/airsim.cc ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
# cp ${FSIM_CC_DIR}/heartbeat.cc ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
cp ${FSIM_CC_DIR}/airsim.h ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
# cp ${FSIM_CC_DIR}/heartbeat.h ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
cp ${FSIM_CC_DIR}/firesim_top.cc ${FIRESIM_DIR}/sim/src/main/cc/firesim/


# Copy simulation configs
cp ${ROSE_DIR}/soc/sim/config_runtime_local.yaml ${FIRESIM_DIR}/deploy/config_runtime.yaml
cp ${ROSE_DIR}/soc/sim/config_build_recipes_local.yaml ${FIRESIM_DIR}/deploy/config_build_recipes.yaml
cp ${ROSE_DIR}/soc/sim/config_build_local.yaml ${FIRESIM_DIR}/deploy/config_build.yaml

# Patch build script
sed -i 's/midas, icenet, testchipip, sifive_blocks)/midas, icenet, testchipip, sifive_blocks, chipyard)/g' ${FIRESIM_DIR}/sim/build.sbt
cd ${ROSE_DIR}/soc/sw/onnxruntime-riscv
git submodule update --init --recursive



