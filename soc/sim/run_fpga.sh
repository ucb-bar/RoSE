#!/usr/bin/env bash
# FireSim FPGA infrasetup/runworkload launcher for the U250 RoSE bitstream.
# Usage: run_fpga.sh {infrasetup|runworkload|kill}
set -eo pipefail
STEP="${1:-infrasetup}"
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
export PATH=/ecad/tools/xilinx/Vivado/2023.1/bin:$PATH
source /ecad/tools/xilinx/Vivado/2023.1/settings64.sh 2>/dev/null || true
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
# fabric authenticates to the localhost run farm with ~/firesim.pem
if [ ! -e ~/firesim.pem ]; then ln -sf ~/.ssh/id_rsa ~/firesim.pem; fi
case "$STEP" in
  infrasetup)   firesim infrasetup ;;
  runworkload)  firesim runworkload ;;
  kill)         firesim kill ;;
  *) echo "unknown step: $STEP"; exit 2 ;;
esac
echo "FIRESIM_${STEP}_EXIT=$?"
