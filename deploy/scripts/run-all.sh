#!/bin/bash

cd ${ROSE_DIR}
bash ${ROSE_DIR}/deploy/scripts/tunnel-exp.sh
bash ${ROSE_DIR}/deploy/scripts/rose-hw-sw-sweep.sh
bash ${ROSE_DIR}/deploy/scripts/rose-velocity-sweep.sh
bash ${ROSE_DIR}/deploy/scripts/rose-dynamic_exp.sh
bash ${ROSE_DIR}/deploy/scripts/rose-perf-sync-only.sh