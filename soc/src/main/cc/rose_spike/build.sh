#!/usr/bin/env bash
#
# Build the RoSE Spike bridge in these flavors:
#   1. librose_spike.so    - passive --extlib plugin (fast functional tier, single-core)
#   2. rose_spike_sim      - control-inverted lockstep harness (multicore, two-sided barrier)
#   3. rose_spike_trace    - like (2) but linked against the TACIT l_trace spike, so a
#                            CONFIG_STARTUP_TACIT guest emits tacit.out during the lockstep
#
# (1)/(2) require the chipyard-built spike libraries under
# toolchains/riscv-tools/riscv-isa-sim/build (libriscv.a etc.), and the sim.h
# `step()` protected patch (see rose_spike_sim_stepaccess.patch, applied by setup.sh).
# (3) requires the l_trace spike source+libs (ROSE_LTRACE_SPIKE, default the
# chipyard-fsim riscv-isa-sim @ l_trace); build.sh applies the same step() patch to it.
#
# Usage: build.sh [plugin|harness|trace|all]   (default: all; 'all' excludes trace)
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

if [[ "$what" == "trace" ]]; then
  # TACIT l_trace variant: same harness source, linked against the l_trace spike so a
  # CONFIG_STARTUP_TACIT guest's 0x3000000 encoder writes produce tacit.out in the CWD.
  LT="${ROSE_LTRACE_SPIKE:-/scratch2/dima/chipyard-fsim/toolchains/riscv-tools/riscv-isa-sim}"
  [[ -f "$LT/build/libriscv.a" ]] || { echo "ERROR: l_trace libs not found under $LT/build (set ROSE_LTRACE_SPIKE)"; exit 1; }
  # Idempotent step() access patch on the l_trace sim.h (header-only, no spike rebuild).
  SIMH="$LT/riscv/sim.h"
  if ! grep -q 'RoSE lockstep: expose step' "$SIMH"; then
    perl -0pi -e 's{(\n\s*)(void step\(size_t n\); // step through simulation)}{$1protected: /* RoSE lockstep: expose step() to the rose_sim_t harness subclass */$1$2$1private:}' "$SIMH"
    echo "patched step() access in $SIMH"
  else
    echo "step() access already patched in $SIMH"
  fi
  LTFLAGS=(-std=c++2a -O2 -D_GNU_SOURCE -DROSE_TRACE_BUILD -include sys/syscall.h
           -I"$LT" -I"$LT/riscv" -I"$LT/fesvr" -I"$LT/softfloat" -I"$LT/build")
  # The l_trace libs are built with the SYSTEM gcc/glibc (they need __isoc23_* /
  # __libc_single_threaded / statx), so link with the system toolchain, not the chipyard
  # conda one that env.sh activated — otherwise the link fails on those glibc symbols.
  TRACE_CXX="${ROSE_TRACE_CXX:-/usr/bin/g++}"
  env -u CONDA_PREFIX -u LIBRARY_PATH -u LD_LIBRARY_PATH -u CPATH \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH PATH=/usr/bin:/bin \
    "$TRACE_CXX" "${LTFLAGS[@]}" \
    "$CC/rose_sync_client.cc" "$CC/rose_spike/rose_spike_sim.cc" \
    -Wl,--start-group \
      "$LT/build/libriscv.a" "$LT/build/libfesvr.a" "$LT/build/libsoftfloat.a" \
      "$LT/build/libdisasm.a" "$LT/build/libfdt.a" \
    -Wl,--end-group -lpthread -ldl \
    -o "$OUT/rose_spike_trace"
  echo "built $OUT/rose_spike_trace"
fi
