#!/usr/bin/env bash
# Build the SHARED flight controller for the real ESP32C6 "riskybird" target
# (board esp32c6_devkitc/esp32c6/hpcore). Same application/estimator/TinyMPC as the RoSE
# co-sim build (soc/sim/build_zephyr_rose.sh); only the board overlay + .conf differ, so the
# sensors bind to the real bosch,bmi08x-* driver over I2C instead of the RoSE virtual driver.
#
# ONE-TIME setup: the ESP32C6 SoC support needs the Espressif HAL (hal_espressif), which the
# lean RISC-V RoSE zephyr workspace (west-riscv.yml manifest) does not fetch by default. Add
# it to the workspace once, either by copying it from an espressif-enabled workspace:
#     cp -a <other_ws>/modules/hal/espressif "$ZEPHYR_BASE/../modules/hal/espressif"
# or by fetching via west against the full manifest. Then `pip install "esptool>=5.0.2"`.
# Point ROSE_HAL_ESPRESSIF at it (default below). The real ToF (st,vl53l1x) additionally
# needs hal_st, and optical flow needs a PMW3901 driver -- see the sample overlay notes;
# without them the app builds IMU-only (flow/ToF guarded off).
#
# Configure via env (defaults target the RoSE dev host):
#   ZEPHYR_BASE, ZEPHYR_SDK_INSTALL_DIR, ROSE_HAL_ESPRESSIF, and `west` on PATH.
set -eo pipefail
ROSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROSE_DIR/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_flight_controller"
OUT="$ROSE_DIR/soc/sim/zephyr_rose_builds/rose_flight_controller_esp32c6"

: "${ZEPHYR_BASE:=/scratch2/dima/zephyr-chipyard-sw-fresh/zephyr_ws/zephyr}"
: "${ZEPHYR_SDK_INSTALL_DIR:=/scratch2/dima/zephyr-chipyard-sw-fresh/tools-manual/zephyr-sdk-1.0.0-beta1}"
: "${ROSE_HAL_ESPRESSIF:=$ZEPHYR_BASE/../modules/hal/espressif}"
export ZEPHYR_BASE ZEPHYR_SDK_INSTALL_DIR
export ZEPHYR_TOOLCHAIN_VARIANT="${ZEPHYR_TOOLCHAIN_VARIANT:-zephyr}"

if ! command -v west >/dev/null 2>&1; then
  echo "ERROR: 'west' not on PATH. Activate the zephyr conda env first." >&2; exit 1
fi
if [ ! -d "$ROSE_HAL_ESPRESSIF" ]; then
  echo "ERROR: hal_espressif not found at $ROSE_HAL_ESPRESSIF (see one-time setup above)." >&2; exit 1
fi

echo "ZEPHYR_BASE=$ZEPHYR_BASE"
echo "hal_espressif=$ROSE_HAL_ESPRESSIF"
west build -p always -b esp32c6_devkitc/esp32c6/hpcore --build-dir "$OUT" "$APP" \
  -- -DZEPHYR_EXTRA_MODULES="$ROSE_HAL_ESPRESSIF"
echo "BUILT $OUT/zephyr/zephyr.elf"
