#!/usr/bin/env bash
# Run the reqrsp_stress guest (looped DMA+reqrsp) through the DMA metasim, paired
# with the already-running PatternEnv sync on :10001. Reproduces the bridge stall
# in a controlled, fully-visible RTL sim (no FPGA). SUCCESS = uartlog "ROSE stress:
# 300 iters 0 fails => PASS"; a HANG at "STRESS iter K: reqrsp..." == bug reproduced.
set -eo pipefail
# Idempotently apply the Verilator lint fix the DMA metasim build needs (else it aborts on
# SYNCASYNCNET, a benign reset-net CDC warning). Lives in the chipyard submodule, so we
# self-apply here rather than carry a submodule-pointer commit.
MK=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/sims/firesim/sim/midas/src/main/cc/rtlsim/Makefrag-verilator
if [ -f "$MK" ] && ! grep -q 'Wno-SYNCASYNCNET' "$MK"; then
  sed -i 's/\(\s*\)-Wno-UNDRIVEN \\/\1-Wno-UNDRIVEN \\\n\1-Wno-SYNCASYNCNET \\/' "$MK"
  echo "[run_metasim_stress] applied -Wno-SYNCASYNCNET to Makefrag-verilator"
fi
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
export ROSE_DMA_RX=1
cd sims/firesim && source sourceme-manager.sh --skip-ssh-setup && cd sim
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-DROSE_DMA_RX"
timeout 2700 make run-verilator TARGET_PROJECT=firesim DESIGN=FireSim \
  PLATFORM=xilinx_alveo_u250 TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  SIM_BINARY=/scratch/dima/rose-infra/RoSE/soc/sim/zephyr_rose_builds/reqrsp_stress/zephyr/zephyr.elf 2>&1
echo "METASIM_STRESS_EXIT=$?"
