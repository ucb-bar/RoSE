#!/usr/bin/env bash
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
cd sims/firesim && source sourceme-manager.sh --skip-ssh-setup && cd sim
timeout 400 make run-verilator TARGET_PROJECT=firesim DESIGN=FireSim \
  PLATFORM=xilinx_alveo_u250 TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  SIM_BINARY=/scratch/dima/rose-infra/RoSE/soc/sim/zephyr_hello_build/zephyr/zephyr.elf 2>&1
echo "ZEPHYR_RUN_EXIT=$?"
