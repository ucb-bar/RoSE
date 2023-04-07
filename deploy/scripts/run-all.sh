#!/bin/bash

PROJECT_ROOT="/scratch/$(whoami)"
ROSE_DIR=${PROJECT_ROOT}/RoSE

cd ${ROSE_DIR}
bash ${ROSE_DIR}/deploy/scripts/tunnel-exp.sh
bash ${ROSE_DIR}/deploy/scripts/rose-hw-sw-sweep.sh
bash ${ROSE_DIR}/deploy/scripts/rose-velocity-sweep.sh
bash ${ROSE_DIR}/deploy/scripts/rose-dynamic-exp.sh
bash ${ROSE_DIR}/deploy/scripts/rose-perf-sync-only.sh
bash ${ROSE_DIR}/deploy/scripts/rose-perf-tunnel-exp.sh