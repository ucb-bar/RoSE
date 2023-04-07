#!/bin/bash
PROJECT_ROOT="/scratch/$(whoami)"
FIRESIM_DIR=${PROJECT_ROOT}/firesim
CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard
ROSE_DIR=${PROJECT_ROOT}/RoSE

cd ${ROSE_DIR}/soc/
cd ./sw/onnxruntime-riscv/
./build.sh --enable_training --config=Debug --cmake_extra_defines onnxruntime_USE_SYSTOLIC=ON onnxruntime_SYSTOLIC_INT8=OFF onnxruntime_SYSTOLIC_FP32=ON
cd ./systolic_runner/imagenet_runner/ && bash ./build.sh  --enable_training && cd ../..
cd ../
marshal build rose-images/airsim-driver-fed.json
marshal build rose-images/airsim-control-fed.json

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
