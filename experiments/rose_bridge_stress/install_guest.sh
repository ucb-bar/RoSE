#!/usr/bin/env bash
# Install the durable reqrsp_stress guest source into the zephyr-chipyard-sw submodule
# and build it (board spike_riscv64, runs on the Rocket metasim + the FPGA). The guest
# source is kept HERE (main repo) so the stress test survives a submodule reset.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSE="$(cd "$HERE/../.." && pwd)"
DST="$ROSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose/reqrsp_stress"
mkdir -p "$DST/src" "$DST/boards"
cp "$HERE/guest/main.c"        "$DST/src/main.c"
cp "$HERE/guest/prj.conf"      "$DST/prj.conf"
cp "$HERE/guest/CMakeLists.txt" "$DST/CMakeLists.txt"
cp -r "$HERE/guest/boards/." "$DST/boards/" 2>/dev/null || true
echo "[install_guest] source installed to $DST"
bash "$ROSE/soc/sim/build_zephyr_rose.sh" reqrsp_stress
echo "[install_guest] built: $ROSE/soc/sim/zephyr_rose_builds/reqrsp_stress/zephyr/zephyr.elf"
