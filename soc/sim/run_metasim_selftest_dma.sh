#!/usr/bin/env bash
# Run the RoSE selftest through the DMA-enabled metasim (built by
# build_metasim_dma.sh). Pair with the PatternEnv sync harness on :10001:
#   (deploy/.venv-rose) python deploy/hephaestus/run_sync_only.py \
#       --yaml_path deploy/config/config_gym_PatternEnv-v0.yaml   # (ROSE_GYM_ENV=PatternEnv-v0)
# SUCCESS = uartlog prints:  ROSE selftest: dma=PASS reqrsp=PASS => PASS
# ROSE_DMA_RX / CXXFLAGS are kept identical to the build so make treats the
# driver as up-to-date and does not silently rebuild it without -DROSE_DMA_RX.
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
export ROSE_DMA_RX=1
cd sims/firesim && source sourceme-manager.sh --skip-ssh-setup && cd sim
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-DROSE_DMA_RX"
timeout 900 make run-verilator TARGET_PROJECT=firesim DESIGN=FireSim \
  PLATFORM=xilinx_alveo_u250 TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  SIM_BINARY=/scratch/dima/rose-infra/RoSE/soc/sim/zephyr_rose_builds/selftest/zephyr/zephyr.elf 2>&1
echo "METASIM_RUN_EXIT=$?"
