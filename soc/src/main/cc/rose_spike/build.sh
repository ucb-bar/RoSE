#!/usr/bin/env bash
#
# Build the RoSE Spike bridge in both flavors:
#   1. librose_spike.so    - passive --extlib plugin (fast functional tier, single-core)
#   2. rose_spike_sim      - control-inverted lockstep harness (multicore, two-sided barrier)
#
# Requires the chipyard-built spike libraries under
# toolchains/riscv-tools/riscv-isa-sim/build (libriscv.a etc.), and the sim.h
# `step()` protected patch (see rose_spike_sim_stepaccess.patch, applied by setup.sh).
#
# Usage: build.sh [plugin|harness|all]   (default: all)
set -eo pipefail

ROSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CY="$ROSE_DIR/soc/sim/chipyard"
S="$CY/toolchains/riscv-tools/riscv-isa-sim"
CC="$ROSE_DIR/soc/src/main/cc"
OUT="$ROSE_DIR/soc/sim"

cd "$CY"; source env.sh

CXXFLAGS=(-std=c++2a -O2 -D_GNU_SOURCE -include sys/syscall.h
          -I"$S" -I"$S/riscv" -I"$S/fesvr" -I"$S/softfloat" -I"$S/build")

what="${1:-all}"

if [[ "$what" == "plugin" || "$what" == "all" ]]; then
  g++ "${CXXFLAGS[@]}" -fPIC -shared \
    "$CC/rose_sync_client.cc" "$CC/rose_spike/rose_spike_device.cc" \
    -o "$OUT/librose_spike.so"
  echo "built $OUT/librose_spike.so"
fi

if [[ "$what" == "harness" || "$what" == "all" ]]; then
  g++ "${CXXFLAGS[@]}" \
    "$CC/rose_sync_client.cc" "$CC/rose_spike/rose_spike_sim.cc" \
    -Wl,--start-group \
      "$S/build/libriscv.a" "$S/build/libfesvr.a" "$S/build/libsoftfloat.a" \
      "$S/build/libdisasm.a" "$S/build/libfdt.a" \
    -Wl,--end-group -lpthread -ldl \
    -o "$OUT/rose_spike_sim"
  echo "built $OUT/rose_spike_sim"
fi
