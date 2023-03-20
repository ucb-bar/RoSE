#!/bin/bash

cd ./sw/onnxruntime-riscv/
./build.sh --enable_training --config=Debug --cmake_extra_defines onnxruntime_USE_SYSTOLIC=ON onnxruntime_SYSTOLIC_INT8=OFF onnxruntime_SYSTOLIC_FP32=ON
cd ./systolic_runner/imagenet_runner/ && bash ./build.sh  --enable_training && cd ../..
