#!/bin/bash
DIR=$(dirname "$(realpath "$0")")
echo $DIR
PROJECT_ROOT="${DIR}/../.."
echo "Project root: $PROJECT_ROOT"
FIRESIM_DIR=${PROJECT_ROOT}/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
ROSE_DIR=${PROJECT_ROOT}/RoSE
SCALA_DIR=${ROSE_DIR}/soc/src/main/scala
FSIM_CC_DIR=${ROSE_DIR}/soc/src/main/cc

# Create an array of source files
sources=(
    #airsim scala files
    "${SCALA_DIR}/AirSimIO.scala"
    "${SCALA_DIR}/IOBinders.scala" 
    "${SCALA_DIR}/BridgeBinders.scala" 
    "${SCALA_DIR}/DigitalTop.scala" 
    "${SCALA_DIR}/AbstractConfig.scala" 
    "${SCALA_DIR}/AirSimBridge.scala" 
    "${SCALA_DIR}/RoSEConfigs.scala"
    #rose scala files
    "${SCALA_DIR}/RoSEAdapter.scala"
    "${SCALA_DIR}/RoSEBrdige.scala"
    #C++ files
    "${FSIM_CC_DIR}/airsim.cc"
    "${FSIM_CC_DIR}/airsim.h"
    "${FSIM_CC_DIR}/firesim_top.cc"
    #simulation configs
    "${ROSE_DIR}/soc/sim/config_runtime_local.yaml"
    "${ROSE_DIR}/soc/sim/config_build_recipes_local.yaml"
    "${ROSE_DIR}/soc/sim/config_build_local.yaml"
    "${ROSE_DIR}/soc/sim/config_hwdb_local.yaml"
    #workload configs
    "${ROSE_DIR}/soc/sim/airsim-driver-fed.json"
    "${ROSE_DIR}/soc/sim/airsim-control-fed.json"
)

# Create an array of destination files
destinations=(
    #scala destinations
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/example/AirSimIO.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/IOBinders.scala"
    "${CHIPYARD_DIR}/generators/firechip/src/main/scala/BridgeBinders.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/DigitalTop.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/AbstractConfig.scala"
    "${FIRESIM_DIR}/sim/firesim-lib/src/main/scala/bridges/AirSimBridge.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/RoSEConfigs.scala"
    #rose scala destinations
    "${CHIPYARD_DIR}/generators/rose/src/main/scala/RoSEAdapter.scala"
    "${FIRESIM_DIR}/sim/firesim-lib/src/main/scala/bridges/RoSEBridge.scala"
    #C++ destinations
    "${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/airsim.cc"
    "${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/airsim.h"
    "${FIRESIM_DIR}/sim/src/main/cc/firesim/firesim_top.cc"
    #simulation configs destinations
    "${FIRESIM_DIR}/deploy/config_runtime.yaml"
    "${FIRESIM_DIR}/deploy/config_build_recipes.yaml"
    "${FIRESIM_DIR}/deploy/config_build.yaml"
    "${FIRESIM_DIR}/deploy/config_hwdb.yaml"
    #workload configs destinations
    "${FIRESIM_DIR}/deploy/workloads/airsim-driver-fed.json"
    "${FIRESIM_DIR}/deploy/workloads/airsim-control-fed.json"
) 

# Iterate over the arrays and create symbolic links
for ((i=0;i<${#sources[@]};++i)); do
    # Check if the destination file or symlink already exists
    if [[ -e "${destinations[$i]}" || -L "${destinations[$i]}" ]]; then
        echo "Removing existing file or symbolic link at ${destinations[$i]}..."
        rm -rf "${destinations[$i]}"
    fi

    # Create the symbolic link
    echo "Creating symbolic link from ${sources[$i]} to ${destinations[$i]}..."
    ln -s "${sources[$i]}" "${destinations[$i]}"
done

# # Copy scala files
# cp ${SCALA_DIR}/AirSimIO.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/example/
# cp ${SCALA_DIR}/IOBinders.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/
# cp ${SCALA_DIR}/BridgeBinders.scala ${CHIPYARD_DIR}/generators/firechip/src/main/scala/
# cp ${SCALA_DIR}/DigitalTop.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/
# cp ${SCALA_DIR}/AbstractConfig.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/
# cp ${SCALA_DIR}/AirSimBridge.scala  ${FIRESIM_DIR}/sim/firesim-lib/src/main/scala/bridges/
# cp ${SCALA_DIR}/RoSEConfigs.scala ${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/

# # Copy C++ files
# cp ${FSIM_CC_DIR}/airsim.cc ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
# # cp ${FSIM_CC_DIR}/heartbeat.cc ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
# cp ${FSIM_CC_DIR}/airsim.h ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
# # cp ${FSIM_CC_DIR}/heartbeat.h ${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/
# cp ${FSIM_CC_DIR}/firesim_top.cc ${FIRESIM_DIR}/sim/src/main/cc/firesim/


# # Copy simulation configs
# cp ${ROSE_DIR}/soc/sim/config_runtime_local.yaml        ${FIRESIM_DIR}/deploy/config_runtime.yaml
# cp ${ROSE_DIR}/soc/sim/config_build_recipes_local.yaml  ${FIRESIM_DIR}/deploy/config_build_recipes.yaml
# cp ${ROSE_DIR}/soc/sim/config_build_local.yaml          ${FIRESIM_DIR}/deploy/config_build.yaml
# cp ${ROSE_DIR}/soc/sim/config_hwdb_local.yaml           ${FIRESIM_DIR}/deploy/config_hwdb.yaml

# Copy workload configs
# cp ${ROSE_DIR}/soc/sim/airsim-driver-fed.json ${FIRESIM_DIR}/deploy/workloads/
# cp ${ROSE_DIR}/soc/sim/airsim-control-fed.json ${FIRESIM_DIR}/deploy/workloads/

# Init firemarshal submodules
cd ${FIRESIM_DIR}/sw/firesim-software/
./init-submodules.sh

# Patch build script
sed -i 's/midas, icenet, testchipip, sifive_blocks)/midas, icenet, testchipip, sifive_blocks, chipyard)/g' ${FIRESIM_DIR}/sim/build.sbt

echo 'lazy val rose = (project in file("generators/rose"))
  .dependsOn(rocketchip, testchipip)
  .settings(libraryDependencies ++= rocketLibDeps.value)
  .settings(commonSettings)' >> ${CHIPYARD_DIR}/build.sbt

sed -i 's/gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator)/gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator, rose)/g' ${CHIPYARD_DIR}/build.sbt

cd ${ROSE_DIR}/soc/sw/onnxruntime-riscv
git submodule update --init --recursive



