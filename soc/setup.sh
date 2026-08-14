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
mkdir -p ${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/scala/tacit

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
    #tacit trace bridge scala files (ported from riscv-tacit/chipyard@fix-issue-unit)
    "${SCALA_DIR}/TraceRawByte.scala"
    "${SCALA_DIR}/TacitBridge.scala"
    "${SCALA_DIR}/TacitModule.scala"
    "${SCALA_DIR}/TacitBridgeGG.scala"
    #C++ files (RoSE bridge driver; legacy filename was airsim.{cc,h})
    "${FSIM_CC_DIR}/rosebridge.cc"
    "${FSIM_CC_DIR}/rosebridge.h"
    #tacit trace bridge host driver
    "${FSIM_CC_DIR}/tacit.cc"
    "${FSIM_CC_DIR}/tacit.h"
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
    #tacit trace bridge scala destinations
    "${CHIPYARD_DIR}/generators/firechip/bridgeinterfaces/src/main/scala/TraceRawByte.scala"
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/scala/tacit/TacitBridge.scala"
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/scala/tacit/TacitModule.scala"
    "${CHIPYARD_DIR}/generators/firechip/goldengateimplementations/src/main/scala/TacitBridge.scala"
    #C++ destinations
    # chipyard-as-top: bridge C++ drivers live under firechip/bridgestubs, not firesim-lib
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/cc/bridges/rosebridge.cc"
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/cc/bridges/rosebridge.h"
    #tacit trace bridge host driver destinations
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/cc/bridges/tacit.cc"
    "${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/cc/bridges/tacit.h"
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
    # Skip gracefully if the destination's parent dir doesn't exist (e.g. the
    # onnxruntime-riscv submodule is not initialized — only needed for the DNN
    # build, not for Verilator metasim). Avoids hard-failing mid-setup.
    dest_parent="$(dirname "${destinations[$i]}")"
    if [ ! -d "${dest_parent}" ]; then
        echo "WARNING: destination dir ${dest_parent} missing; skipping ${destinations[$i]} (relevant submodule not initialized?)"
        continue
    fi

    # Check if the destination file or symlink already exists
    if [[ -e "${destinations[$i]}" || -L "${destinations[$i]}" ]]; then
        echo "Removing existing file or symbolic link at ${destinations[$i]}..."
        rm -rf "${destinations[$i]}"
    fi

    # Create the symbolic link
    echo "Creating symbolic link from ${sources[$i]} to ${destinations[$i]}..."
    ln -s "${sources[$i]}" "${destinations[$i]}"
done

# Init firemarshal submodules.
# chipyard-as-top: FireMarshal lives at ${CHIPYARD_DIR}/software/firemarshal and is
# already initialized by chipyard/build-setup.sh. The legacy ${FIRESIM_DIR}/sw/firesim-software
# path and the 'ubuntu-add' branch no longer exist. Only needed for building Linux
# target images, NOT for Verilator metasim, so this is best-effort.
if [ -d ${CHIPYARD_DIR}/software/firemarshal ]; then
  ( cd ${CHIPYARD_DIR}/software/firemarshal && ./init-submodules.sh )
fi
cd ${ROSE_DIR}

# ---------------------------------------------------------------------------
# Patch chipyard build.sbt to add the `rose` generator and wire it in.
#
# chipyard-as-top: ALL sbt wiring lives in ${CHIPYARD_DIR}/build.sbt. The firechip
# projects (bridgestubs/bridgeinterfaces/chip) already depend on `chipyard`, and the
# goldengateimplementations + bridgeinterfaces dirs are copied into MIDAS/GoldenGate
# at build time (TARGET_COPY_TO_MIDAS_SCALA_DIRS). So we do NOT patch the nested
# firesim sim/build.sbt (that is the firesim-standalone build, unused here).
#
# Dependency facts (from the RoSE Scala sources):
#   - rose.* sources import firechip.bridgeinterfaces.*  -> rose dependsOn firechip_bridgeinterfaces
#   - chipyard sources (IOBinders/Ports/RoSEConfigs) import rose.* and
#     firechip.bridgeinterfaces.*  -> chipyard dependsOn rose + firechip_bridgeinterfaces
# ---------------------------------------------------------------------------

# 1. Define the rose generator project (idempotent). Mirrors the standard
#    rocketchip-based generator pattern (chisel6 plugin inherited via rocketLibDeps),
#    plus a dependency on firechip_bridgeinterfaces for the shared port/param types.
if grep -q "lazy val rose " ${CHIPYARD_DIR}/build.sbt; then
  echo "rose project found in chipyard sbt, not appending."
else
  echo "Appending rose project to chipyard build.sbt."
  echo '
lazy val rose = (project in file("generators/rose"))
  .dependsOn(rocketchip, testchipip, firechip_bridgeinterfaces)
  .settings(libraryDependencies ++= rocketLibDeps.value)
  .settings(commonSettings)' >> ${CHIPYARD_DIR}/build.sbt
fi

# 2. Add rose + firechip_bridgeinterfaces to chipyard's always-on deps (idempotent).
#    Chipyard 1.14.0 restructured build.sbt: the `chipyard` project is now built in a
#    block from a `baseProjects: Seq[ProjectReference]` list (no longer a flat
#    `.dependsOn(...)` chain). We splice rose + firechip_bridgeinterfaces into that Seq,
#    right after the stable `constellation, barf, shuttle, rerocc,` line.
#    (Pre-1.14.0 fallback: the old flat-list sed, kept for older checkouts.)
echo "Wiring rose + firechip_bridgeinterfaces into chipyard baseProjects."
if ! grep -q "rose, firechip_bridgeinterfaces," ${CHIPYARD_DIR}/build.sbt; then
  if grep -q "constellation, barf, shuttle, rerocc," ${CHIPYARD_DIR}/build.sbt; then
    # 1.14.0+ block-style baseProjects Seq
    sed -i 's/^\([[:space:]]*\)constellation, barf, shuttle, rerocc,/\1constellation, barf, shuttle, rerocc,\n\1rose, firechip_bridgeinterfaces,/' ${CHIPYARD_DIR}/build.sbt
  else
    # pre-1.14.0 flat dependsOn list
    sed -i 's/gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator,/gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator, rose, firechip_bridgeinterfaces,/g' ${CHIPYARD_DIR}/build.sbt
  fi
fi

# ---------------------------------------------------------------------------
# Patch spike (riscv-isa-sim) to expose sim_t::step() for the RoSE lockstep
# harness (rose_spike_sim). This is the ONLY spike source modification RoSE makes;
# it moves `void step(size_t)` from private to protected so rose_sim_t can drive
# the step loop. See ROSE_SPIKE_LOCKSTEP_PLAN.md (on the planning branch). Idempotent (skips if applied);
# best-effort (the harness is optional — don't hard-fail metasim/FPGA setup).
# ---------------------------------------------------------------------------
ISA_SIM_DIR=${CHIPYARD_DIR}/toolchains/riscv-tools/riscv-isa-sim
STEP_PATCH=${FSIM_CC_DIR}/rose_spike/rose_spike_sim_stepaccess.patch
if [ -f "${ISA_SIM_DIR}/riscv/sim.h" ] && [ -f "${STEP_PATCH}" ]; then
  if grep -q "RoSE lockstep: expose step()" "${ISA_SIM_DIR}/riscv/sim.h"; then
    echo "spike sim.h step() patch already applied; skipping."
  else
    if ( cd "${ISA_SIM_DIR}" && git apply --check "${STEP_PATCH}" ) 2>/dev/null; then
      ( cd "${ISA_SIM_DIR}" && git apply "${STEP_PATCH}" )
      echo "Applied spike sim.h step() patch for the RoSE lockstep harness."
    else
      echo "WARNING: could not apply ${STEP_PATCH} to spike sim.h (already patched or spike moved?); the rose_spike_sim lockstep harness may not build until this is resolved. The --extlib plugin (librose_spike.so) is unaffected."
    fi
  fi
else
  echo "WARNING: spike sim.h or step patch missing; skipping RoSE lockstep patch (harness build only)."
fi
cd ${ROSE_DIR}

# echo "Updating onnxruntime-riscv submodules"
# cd ${ROSE_DIR}
# git submodule update --init --recursive ${ROSE_DIR}/soc/sw/onnxruntime-riscv
# cd ${ROSE_DIR}



