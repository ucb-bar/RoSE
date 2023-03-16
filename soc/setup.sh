#!/bin/bash
PROJECT_ROOT='/scratch/vnikiforov'
FIRESIM_DIR=${PROJECT_ROOT}/firesim
ROSE_DIR=${PROJECT_ROOT}/RoSE

cp ${ROSE_DIR}/soc/sim/config_runtime_local.yaml ${FIRESIM_DIR}/deploy/config_runtime.yaml
cp ${ROSE_DIR}/soc/sim/config_build_recipes_local.yaml ${FIRESIM_DIR}/deploy/config_build_recipes.yaml
cp ${ROSE_DIR}/soc/sim/config_build_local.yaml ${FIRESIM_DIR}/deploy/config_build.yaml
