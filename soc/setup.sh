#!/bin/bash
ROSE_DIR=$(pwd)/
# FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim
# CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
CHIPYARD_DIR=${ROSE_DIR}/soc/sim/chipyard
FIRESIM_DIR=${CHIPYARD_DIR}/sims/firesim
SCALA_DIR=${ROSE_DIR}/soc/src/main/scala
FSIM_CC_DIR=${ROSE_DIR}/soc/src/main/cc

# echo "Updating onnxruntime-riscv submodules"
# cd ${ROSE_DIR}
# git submodule update --init --recursive ${ROSE_DIR}/soc/sw/onnxruntime-riscv
cd ${ROSE_DIR}
mkdir -p ${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/scala/rose

# Create an array of source files
sources=(
    "${SCALA_DIR}/IOBinders.scala" 
    "${SCALA_DIR}/Ports.scala" 
    "${SCALA_DIR}/BridgeBinders.scala" 
    "${SCALA_DIR}/DigitalTop.scala" 
    "${SCALA_DIR}/RoSEConfigs.scala"
    "${SCALA_DIR}/RoSEFireSimConfigs.scala"
    #rose scala files
    "${SCALA_DIR}/RoSEAdapter.scala"
    "${SCALA_DIR}/RoSEBridgeModule.scala"
    "${SCALA_DIR}/RoSEBridgeStub.scala"
    "${SCALA_DIR}/RoSEBridgePort.scala"
    "${SCALA_DIR}/RoSEIO.scala"
    "${SCALA_DIR}/RoSEGeneratorConfig.scala"
    "${SCALA_DIR}/RoSEDMA.scala" 
    "${SCALA_DIR}/Dataflow.scala"
    #C++ files
    "${FSIM_CC_DIR}/airsim.cc"
    "${FSIM_CC_DIR}/airsim.h"
    #simulation configs
    "${ROSE_DIR}/soc/sim/config/config_runtime_local.yaml"
    "${ROSE_DIR}/soc/sim/config/config_build_recipes_local.yaml"
    "${ROSE_DIR}/soc/sim/config/config_build_local.yaml"
    "${ROSE_DIR}/soc/sim/config/config_hwdb_local.yaml"
    #workload configs
    "${ROSE_DIR}/soc/sim/config/airsim-driver-fed.json"
    "${ROSE_DIR}/soc/sim/config/airsim-control-fed.json"
    #ONNX sources
    "${ROSE_DIR}/soc/sw/dnn/cmd_args.h"
    "${ROSE_DIR}/soc/sw/dnn/runner.cpp"
    "${ROSE_DIR}/soc/sw/dnn/drone.cpp"
    "${ROSE_DIR}/soc/sw/dnn/drone_dynamic.cpp"
    "${ROSE_DIR}/soc/sw/dnn/mmio.h"
    "${ROSE_DIR}/soc/sw/dnn/Makefile"
)

# Create an array of destination files
destinations=(
    #scala destinations
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/iobinders/IOBinders.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/iobinders/Ports.scala"
    "${CHIPYARD_DIR}/generators/firechip/chip/src/main/scala/BridgeBinders.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/DigitalTop.scala"
    "${CHIPYARD_DIR}/generators/chipyard/src/main/scala/config/RoSEConfigs.scala" #****
    #"${CHIPYARD_DIR}/generators/firechip/src/main/scala/RoSEFireSimConfigs.scala"
    "${CHIPYARD_DIR}/generators/firechip/chip/src/main/scala/RoSEFireSimConfigs.scala"
    #rose scala destinations
    "${CHIPYARD_DIR}/generators/rose/src/main/scala/RoSEAdapter.scala" #****
    # "${FIRESIM_DIR}/sim/firesim-lib/src/main/scala/bridges/RoSEBridge.scala"
    "${CHIPYARD_DIR}/generators/firechip/goldengateimplementations/src/main/scala/RoSEBridge.scala"
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/scala/rose/RoSEBridge.scala"
    "${CHIPYARD_DIR}/generators/firechip/bridgeinterfaces/src/main/scala/RoSEBridgePort.scala"
    "${CHIPYARD_DIR}/generators/rose/src/main/scala/RoSEIO.scala" #****
    "${CHIPYARD_DIR}/generators/rose/src/main/scala/RoSEGeneratorConfig.scala" #****
    "${CHIPYARD_DIR}/generators/rose/src/main/scala/RoSEDMA.scala" #****
    "${CHIPYARD_DIR}/generators/rose/src/main/scala/Dataflow.scala" #****
    #C++ destinations
    "${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/airsim.cc"
    "${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/airsim.h"
    #simulation configs destinations
    "${FIRESIM_DIR}/deploy/config_runtime.yaml"
    "${FIRESIM_DIR}/deploy/config_build_recipes.yaml"
    "${FIRESIM_DIR}/deploy/config_build.yaml"
    "${FIRESIM_DIR}/deploy/config_hwdb.yaml"
    #workload configs destinations
    "${FIRESIM_DIR}/deploy/workloads/airsim-driver-fed.json"
    "${FIRESIM_DIR}/deploy/workloads/airsim-control-fed.json"
    #ONNX sources
    "${ROSE_DIR}/soc/sw/onnxruntime-riscv/systolic_runner/imagenet_runner/src/cmd_args.h"
    "${ROSE_DIR}/soc/sw/onnxruntime-riscv/systolic_runner/imagenet_runner/src/runner.cpp"
    "${ROSE_DIR}/soc/sw/onnxruntime-riscv/systolic_runner/imagenet_runner/src/drone.cpp"
    "${ROSE_DIR}/soc/sw/onnxruntime-riscv/systolic_runner/imagenet_runner/src/drone_dynamic.cpp"
    "${ROSE_DIR}/soc/sw/onnxruntime-riscv/systolic_runner/imagenet_runner/src/mmio.h"
    "${ROSE_DIR}/soc/sw/onnxruntime-riscv/systolic_runner/imagenet_runner/Makefile"
) 

# create rose if there is not one already
if [ ! -d "${CHIPYARD_DIR}/generators/rose/src/main/scala/" ]; then
  mkdir -p "${CHIPYARD_DIR}/generators/rose/src/main/scala/"
fi

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

# Init firemarshal submodules
cd ${FIRESIM_DIR}/sw/firesim-software/
git checkout ubuntu-add
./init-submodules.sh

# Patch build script
sed -i 's/midas, icenet, testchipip, sifive_blocks)/midas, icenet, testchipip, sifive_blocks, chipyard)/g' ${FIRESIM_DIR}/sim/build.sbt

if grep -q "lazy val rose" ${CHIPYARD_DIR}/build.sbt; then
  echo "rose found in chipyard sbt, not appending."
else
  echo "rose not found in chipyard sbt, appending."
  echo '
lazy val rose = (project in file("generators/rose"))
  .dependsOn(rocketchip, testchipip)
  .settings(libraryDependencies ++= rocketLibDeps.value)
  .settings(commonSettings)' >> ${CHIPYARD_DIR}/build.sbt
fi

# Check if 'rose' is already a dependency under lazy val chipyard
echo "Updating cy build.sbt"
sed -i 's/gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator,/gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator, rose,/g' ${CHIPYARD_DIR}/build.sbt

if grep -q "lazy val rose" ${FIRESIM_DIR}/sim/build.sbt; then
  echo "rose found in firesim/sim sbt, not appending."
else
  echo "rose not found in firesim/sim sbt, appending."
  echo '
lazy val rose          = ProjectRef(chipyardDir, "rose")
' >> ${FIRESIM_DIR}/sim/build.sbt
fi
echo "Updating fsim build.sbt"
sed -i 's/.dependsOn(midas, icenet, testchipip, rocketchip_blocks)/.dependsOn(midas, icenet, testchipip, rocketchip_blocks, rose)/g' ${FIRESIM_DIR}/sim/build.sbt

# echo "Updating onnxruntime-riscv submodules"
# cd ${ROSE_DIR}
# git submodule update --init --recursive ${ROSE_DIR}/soc/sw/onnxruntime-riscv
# cd ${ROSE_DIR}



