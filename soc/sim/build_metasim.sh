#!/usr/bin/env bash
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
cd sim
# Verilate the generated FireSim RTL into the VFireSim metasim binary.
# SYNCASYNCNET waiver: bridge rxfifo CDC reset trips Verilator -Wall (cosmetic for metasim).
make verilator PLATFORM=xilinx_alveo_u250 TARGET_PROJECT=firesim DESIGN=FireSim \
  TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  EXTRA_VERILATOR_FLAGS='--assert -Wno-SYNCASYNCNET' 2>&1
echo "METASIM_BUILD_EXIT=$?"