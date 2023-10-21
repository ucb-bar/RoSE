#!/bin/bash

# This script will run on the host from the workload directory
# (e.g. workloads/example-fed) every time the workload is built.
# It is recommended to call into something like a makefile because
# this script may be called multiple times.

PROJECT_ROOT="/scratch/$(whoami)"
ROSE_DIR=${PROJECT_ROOT}/

cd ${ROSE_DIR}/soc/
# bash ./build.sh
cp ./sw/onnxruntime-riscv/systolic_runner/imagenet_runner/drone_test ./sw/rose-images/airsim-control-fed/overlay/root/
cp ./sw/onnxruntime-riscv/systolic_runner/imagenet_runner/drone_dynamic_test ./sw/rose-images/airsim-control-fed/overlay/root/