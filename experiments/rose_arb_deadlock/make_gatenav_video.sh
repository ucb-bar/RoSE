#!/usr/bin/env bash
# Assemble the chase-cam JPEGs (f_NNNNN.jpg) from the gate-nav flight into an mp4.
set -uo pipefail
SP=/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/d4827fc8-b516-4227-b71d-79192ba241cd/scratchpad
CHASE_DIR=${1:-$SP/gn_chase}
OUT=${2:-$SP/fpga_gatenav_3gate.mp4}
N=$(ls "$CHASE_DIR"/f_*.jpg 2>/dev/null | wc -l)
echo "chase frames: $N in $CHASE_DIR"
[ "$N" = 0 ] && { echo "no frames"; exit 1; }
# 10 Hz render, decim 2 -> ~5 fps effective; play at 20 fps for a watchable clip
ffmpeg -y -framerate 20 -pattern_type glob -i "$CHASE_DIR/f_*.jpg" \
  -c:v libx264 -pix_fmt yuv420p -vf "scale=960:-2" "$OUT" 2>&1 | tail -3
echo "-> $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
