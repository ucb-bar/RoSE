#!/usr/bin/env bash
# Build the on-SoC flight stack guest ELF: fused vision + EKF + TinyMPC -> 4 motor thrusts.
# Target board spike_riscv64, RVV enabled. Consumes the committed fused RVV model dir.
#
# Reproducible defaults match the "controlled gate-nav" configuration:
#   gate-centerline altitude 2.0 m, 20 Hz vision (FUSED_VISION_DIV=10), yaw-rate cmd.
#
# Env overrides:
#   MODEL_DIR   - fused RVV model kernels dir (default: committed model/rvv_f16)
#   BUILD_DIR   - west build dir            (default: experiments/rose_nav_cosim/build_guest)
#   FUSED_VISION_DIV, START_Z, TARGET_Z, CTRL_ITERS, SETTLE_ITERS, YAW_CMD_GAIN, START_YAW
#
# Usage:
#   source experiments/rose_nav_cosim/env.sh
#   experiments/rose_nav_cosim/build_guest.sh
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ROSE_DIR:?source experiments/rose_nav_cosim/env.sh first}"
ZSW="$ROSE_DIR/soc/sw/xpu-rt/zephyr-chipyard-sw"

MODEL_DIR="${MODEL_DIR:-$HERE/model/rvv_f16}"
BUILD_DIR="${BUILD_DIR:-$HERE/build_guest}"
MB_POOL_INC="${MB_POOL_INC:-$ROSE_DIR/soc/sw/xpu-rt/ModelBlaster/runtime/modelblaster_pool}"

FUSED_VISION_DIV="${FUSED_VISION_DIV:-10}"
START_Z="${START_Z:-2.0f}"
TARGET_Z="${TARGET_Z:-2.0f}"
START_YAW="${START_YAW:-0.0f}"
YAW_CMD_GAIN="${YAW_CMD_GAIN:-0.5f}"
CTRL_ITERS="${CTRL_ITERS:-5000}"
SETTLE_ITERS="${SETTLE_ITERS:-200}"

[ -f "$MODEL_DIR/weights.c" ] || { echo "ERROR: no model at $MODEL_DIR (weights.c missing)"; exit 1; }

echo "[build_guest] MODEL_DIR=$MODEL_DIR"
echo "[build_guest] BUILD_DIR=$BUILD_DIR"

west build -p always -b spike_riscv64 --build-dir "$BUILD_DIR" \
  "$ZSW/samples/rose_fused_mpc" -- \
  -DZEPHYR_EXTRA_MODULES="$ROSE_MODULE" -DRISCV_VECTOR=1 -DNAV_MODE=1 -DROSE_FUSED_NAV=1 \
  -DMODEL_DIR="$MODEL_DIR" -DMB_POOL_INC="$MB_POOL_INC" \
  -DEXTRA_CONF_FILE="$HERE/rvv_nav.conf" \
  -DCTRL_ITERS="$CTRL_ITERS" -DSETTLE_ITERS="$SETTLE_ITERS" \
  "-DSTART_Z=$START_Z" "-DTARGET_Z=$TARGET_Z" "-DFUSED_VISION_DIV=$FUSED_VISION_DIV" \
  "-DSTART_YAW=$START_YAW" "-DYAW_CMD_GAIN=$YAW_CMD_GAIN"

echo "[build_guest] march flags:"
grep -hoE "\-march=[a-z0-9_]+" "$BUILD_DIR/build.ninja" | sort -u
ls -la "$BUILD_DIR/zephyr/zephyr.elf" | awk '{print "[build_guest] ELF:",$5,"bytes",$NF}'
