#!/usr/bin/env bash
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
cd sim
# Compile the full RoSE Scala surface: firechip pulls chipyard -> rose -> bridgeinterfaces/bridgestubs
make PLATFORM=xilinx_alveo_u250 TARGET_PROJECT=firesim DESIGN=FireSim \
  TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  verilog 2>&1
