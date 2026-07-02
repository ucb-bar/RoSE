#!/usr/bin/env bash
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
export PATH=/ecad/tools/xilinx/Vivado/2023.1/bin:$PATH
source /ecad/tools/xilinx/Vivado/2023.1/settings64.sh 2>/dev/null || true
cd sims/firesim
source sourceme-manager.sh --skip-ssh-setup
which vivado
# FireSim's buildbitstream runs replace_rtl on the build-farm host (localhost) over
# fabric/paramiko, which authenticates with the key at ~/firesim.pem (env.key_filename).
# On a local build farm that key must exist and be accepted by localhost; point it at the
# passphrase-less ~/.ssh/id_rsa (which is in localhost's authorized_keys).
if [ ! -e ~/firesim.pem ]; then
  ln -sf ~/.ssh/id_rsa ~/firesim.pem
fi
firesim buildbitstream
