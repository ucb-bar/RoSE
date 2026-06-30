#!/usr/bin/env bash
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
export PATH=/ecad/tools/xilinx/Vivado/2023.1/bin:$PATH
source /ecad/tools/xilinx/Vivado/2023.1/settings64.sh 2>/dev/null || true
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
which vivado
firesim buildbitstream
