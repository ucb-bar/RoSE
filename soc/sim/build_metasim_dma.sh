#!/usr/bin/env bash
# Build the DMA-enabled RoSE metasim (VFireSim).
#
# This is build_metasim.sh with the ROSE_DMA_RX host->FPGA bulk datapath turned
# on in BOTH places that must agree:
#   * ROSE_DMA_RX=1 in the environment  -> read at FireSim/Chisel elaboration
#     time by RoSEBridgeModule.scala, which routes rxfifo from the 512b
#     StreamFromHostCPU adapter (instead of per-word MMIO in_bits).
#   * CXXFLAGS += -DROSE_DMA_RX          -> compiles the rosebridge_t driver's
#     DMA push() datapath (instead of the per-word MMIO send() loop).
# NOTE: the macro is injected via the CXXFLAGS *environment* variable, NOT via
# a TARGET_CXX_FLAGS make argument -- the firechip makefrag does
# `TARGET_CXX_FLAGS += -I.../bridgestubs/...`, so a command-line
# TARGET_CXX_FLAGS=... would clobber every bridge include dir and break the C++
# compile. The inner makefile seeds its CXXFLAGS from the environment, so this
# rides in cleanly alongside the makefrag's -I flags.
# Everything else (grant token, cycle_step/budget, rx_budget/bigstep, bww/route
# config, FPGA->host txfifo, reqrsp RX) stays on MMIO exactly as the default.
#
# Metasim (verilator, CPU) only -- NO FPGA, NO bitstream. Verilate is slow.
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
export ROSE_DMA_RX=1
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
cd sim
export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-DROSE_DMA_RX"
make verilator PLATFORM=xilinx_alveo_u250 TARGET_PROJECT=firesim DESIGN=FireSim \
  TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig PLATFORM_CONFIG=BaseXilinxAlveoU250Config \
  TARGET_PROJECT_MAKEFRAG=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/generators/firechip/chip/src/main/makefrag/firesim \
  EXTRA_VERILATOR_FLAGS='--assert -Wno-SYNCASYNCNET' 2>&1
echo "METASIM_BUILD_EXIT=$?"
