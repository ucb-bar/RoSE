#!/usr/bin/env bash
# Rebuild the FireSim-xilinx_alveo_u250 driver for the Saturn+RoSE MMIO config so
# the two rosebridge.cc fixes (net_read_full framing 9787e6e + ROSE_SYNC_HOST/PORT
# env 02f3284) land in the driver binary. C++ recompile+relink only (RTL/generated-src
# already present -> no re-elaboration). Mirrors runtime_config.build_sim_driver().
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
cd sim
echo "=== BUILD START $(date) ==="
make PLATFORM=xilinx_alveo_u250 TARGET_PROJECT=firesim \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  DESIGN=FireSim TARGET_CONFIG=RoseTLRocketSaturnMMIOOnlyConfig \
  PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  xilinx_alveo_u250 2>&1
echo "=== DRIVER_BUILD_EXIT=$? $(date) ==="
