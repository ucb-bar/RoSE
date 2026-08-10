#!/usr/bin/env bash
# OPTIONAL / PROVENANCE: regenerate the fused RVV model dir (model/rvv_f16).
#
# You do NOT need this to build the guest or reproduce the co-sim -- the generated
# kernels are committed at experiments/rose_nav_cosim/model/rvv_f16 and build with
# zero Python/PyTorch dependency. This script documents HOW that dir was produced.
#
# EXTERNAL INPUTS (NOT in this repo -- collaborator's tree under /scratch/agustin):
#   - trained checkpoint: fused_bc_warehouse_v12_mixed_cnn/.../best.pt
#   - collaborator model source: DIMA/vitfly/models/fused_model.py
# On a machine without that tree, regen is impossible; use the committed model dir.
#
# WINNING CONFIG = int8 (per-channel) encoders + fp16 tail + fp16 low-dim inputs.
# Pure-int8 FAILS on real gate data (low-dim flow outlier + head saturation).
#
# Two stages (run inside the ModelBlaster python env):
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ROSE_DIR:?source experiments/rose_nav_cosim/env.sh (or export ROSE_DIR) first}"
MB="$ROSE_DIR/soc/sw/xpu-rt/ModelBlaster"

# External collaborator inputs (override if your paths differ):
export MODELBLASTER_FUSED_CKPT="${MODELBLASTER_FUSED_CKPT:-/scratch/agustin/projects/DIMA/train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt}"
CALIB_PKL="${CALIB_PKL:-$HERE/calib/calib_real.pkl}"     # REAL gate frames (committed)

[ -f "$MODELBLASTER_FUSED_CKPT" ] || { echo "MISSING checkpoint: $MODELBLASTER_FUSED_CKPT (external, not in git)"; exit 2; }

WORK="${WORK:-$HERE/regen_work}"; mkdir -p "$WORK"

# --- Stage 1: extract the quantized graph (int8 per-channel encoders, f16 low-dim, hybrid tail)
MB_FUSED_HYBRID=1 \
MB_FUSED_LOWDIM_FLOAT=f16 \
MB_FUSED_CALIB_PKL="$CALIB_PKL" \
python "$MB/pipeline/extract_graph.py" \
  --model fused_full --quant int8 --per-channel --num-calibration 60 \
  --out-dir "$WORK"      # -> $WORK/graph.json + weights.npz (== committed model_src/)

# --- Stage 2: lower graph.json -> RVV f16 kernels (codegen only; no checkpoint needed here)
python "$MB/pipeline/generate_kernels.py" \
  --ir "$WORK/graph.json" --target rvv --quant int8 \
  --repo-root "$ROSE_DIR/soc/sw/xpu-rt" \
  --out-dir "$WORK/rvv_f16"

echo "Regenerated -> $WORK/rvv_f16 (compare against committed model/rvv_f16)"
