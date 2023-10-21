#!/bin/bash
ROSE_DIR=$(pwd)
FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard

# Make onnx patches
sed -i '/assert(ran_on_idx >= 0 && ran_on_idx < num_threads_);/s/ran_on_idx >= 0 && //' ${ROSE_DIR}/soc/sw/onnxruntime-riscv/include/onnxruntime/core/platform/EigenNonBlockingThreadPool.h
sed -i 's/  int dummy;/  int dummy = 0;/' ${ROSE_DIR}/soc/sw/onnxruntime-riscv/cmake/external/googletest/googletest/src/gtest-death-test.cc
sed -i "s/'\\\\x00' <= c and //g" ${ROSE_DIR}/soc/sw/onnxruntime-riscv/cmake/external/json/single_include/nlohmann/json.hpp


cd ${ROSE_DIR}/soc/
cd ./sw/onnxruntime-riscv/
./build.sh --enable_training --parallel --config=Debug --cmake_extra_defines onnxruntime_USE_SYSTOLIC=ON onnxruntime_SYSTOLIC_INT8=OFF onnxruntime_SYSTOLIC_FP32=ON
cd ./systolic_runner/imagenet_runner/ && bash ./build.sh  --enable_training && cd ../..
cd ../

# Patch filenames:
sed -i "s|PROJECT_ROOT=\"/scratch/\$(whoami)\"|PROJECT_ROOT=\"${ROSE_DIR}\"|g" "${ROSE_DIR}/soc/sw/rose-images/airsim-driver-fed/host-init.sh"
sed -i "s|PROJECT_ROOT=\"/scratch/\$(whoami)\"|PROJECT_ROOT=\"${ROSE_DIR}\"|g" "${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed/host-init.sh"

# Build firemarshal iamges
marshal build ${ROSE_DIR}/soc/sw/rose-images/airsim-driver-fed.json
marshal build ${ROSE_DIR}/soc/sw/rose-images/airsim-control-fed.json

if [ ! -d ${FIRESIM_DIR}/deploy/workloads/airsim-driver-fed ]; then
    mkdir -p ${FIRESIM_DIR}/deploy/workloads/airsim-driver-fed
fi

if [ ! -d ${FIRESIM_DIR}/deploy/workloads/airsim-control-fed ]; then
    mkdir -p ${FIRESIM_DIR}/deploy/workloads/airsim-control-fed
fi

ln -s ${FIRESIM_DIR}/sw/firesim-software/images/airsim-driver-fed-bin  ${FIRESIM_DIR}/deploy/workloads/airsim-driver-fed/airsim-driver-fed-bin
ln -s ${FIRESIM_DIR}/sw/firesim-software/images/airsim-driver-fed.img  ${FIRESIM_DIR}/deploy/workloads/airsim-driver-fed/airsim-driver-fed.img

ln -s ${FIRESIM_DIR}/sw/firesim-software/images/airsim-control-fed-bin  ${FIRESIM_DIR}/deploy/workloads/airsim-control-fed/airsim-control-fed-bin
ln -s ${FIRESIM_DIR}/sw/firesim-software/images/airsim-control-fed.img  ${FIRESIM_DIR}/deploy/workloads/airsim-control-fed/airsim-control-fed.img

# Build baremetal
cd ${ROSE_DIR}/soc/sw/baremetal/
make
make install

cd ${ROSE_DIR}