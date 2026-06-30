#!/usr/bin/env bash
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
ROSE=/scratch/dima/rose-infra/RoSE
riscv64-unknown-elf-gcc -march=rv64imafd -mabi=lp64d -mcmodel=medany \
  -specs=htif_nano.specs -static -T tests/htif.ld -I tests \
  -I $ROSE/soc/sw/generated-src/rose_c_header \
  $ROSE/soc/sw/rose-images/airsim-packettest/airsim-packettest-dmavalidate.c \
  -o tests/airsim-packettest-dmavalidate.riscv
echo "BUILT $(ls -la tests/airsim-packettest-dmavalidate.riscv)"
