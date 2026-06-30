#!/usr/bin/env bash
# Rebuild + install just spike (libriscv.so + libfesvr.a) into $RISCV so testchipip
# 1.14.0's cospike_impl.cc compiles. Replicates the spike steps of build-toolchain-extra.sh.
set -eo pipefail
cd /scratch/dima/rose-infra/RoSE/soc/sim/chipyard
source env.sh
source scripts/utils.sh
common_setup
RDIR="$(git rev-parse --show-toplevel)"
cd "$RDIR"
echo "RISCV=$RISCV"
SRCDIR="$(pwd)/toolchains/riscv-tools"
. ./scripts/build-util.sh
echo '==> Installing Spike (libriscv.so)'
module_all riscv-isa-sim --prefix="${RISCV}" --with-boost=no --with-boost-asio=no --with-boost-regex=no
echo '==> Installing libfesvr static library'
CLEANAFTERINSTALL=""
module_make riscv-isa-sim libfesvr.a
cp -p "${SRCDIR}/riscv-isa-sim/build/libfesvr.a" "${RISCV}/lib/"
echo "SPIKE_BUILD_EXIT=$?"