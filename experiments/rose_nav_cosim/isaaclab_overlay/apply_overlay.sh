#!/usr/bin/env bash
# Apply the RoSE custom overlay onto a stock IsaacLab checkout.
#
# The RoSE nav co-sim's custom tasks (warehouse gate course, crazyflie thrust/vision
# envs, gym registration) live IN THIS REPO at soc/sw/xpu-rt/sims/isaaclab_tasks/ and
# are imported as the `sims.isaaclab_tasks` package via PYTHONPATH -- they are NOT copied
# into the IsaacLab tree. So the only things that must be applied INTO the IsaacLab
# checkout are the two install-tree deltas captured here:
#   1. patches/isaaclab_contrib-multirotor-thruster-mapping.patch  (isaaclab_contrib core fix)
#   2. scripts/demos/quadcopter_fpv.py                             (untracked demo script)
#
# Usage:  ISAACLAB_DIR=/path/to/IsaacLab bash apply_overlay.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-/scratch2/dima/IsaacLab}"

[ -d "$ISAACLAB_DIR/.git" ] || { echo "ERROR: $ISAACLAB_DIR is not a git checkout of IsaacLab"; exit 1; }

PIN="$(grep -oE '[0-9a-f]{40}' "$HERE/PINNED_VERSION.txt" | head -1)"
HAVE="$(git -C "$ISAACLAB_DIR" rev-parse HEAD)"
if [ "$HAVE" != "$PIN" ]; then
  echo "WARNING: IsaacLab HEAD=$HAVE != pinned $PIN (overlay was validated on the pinned commit)"
fi

echo "[overlay] applying isaaclab_contrib multirotor patch..."
if git -C "$ISAACLAB_DIR" apply --check "$HERE/patches/isaaclab_contrib-multirotor-thruster-mapping.patch" 2>/dev/null; then
  git -C "$ISAACLAB_DIR" apply "$HERE/patches/isaaclab_contrib-multirotor-thruster-mapping.patch"
  echo "[overlay]   applied."
else
  echo "[overlay]   SKIP (already applied or does not apply cleanly -- inspect manually)."
fi

echo "[overlay] installing scripts/demos/quadcopter_fpv.py..."
mkdir -p "$ISAACLAB_DIR/scripts/demos"
cp "$HERE/scripts/demos/quadcopter_fpv.py" "$ISAACLAB_DIR/scripts/demos/quadcopter_fpv.py"

echo "[overlay] done. (The custom TASKS are used from this repo via PYTHONPATH, not copied in.)"
