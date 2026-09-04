#!/usr/bin/env bash
# Export the four campaign models to .pte. Uses a venv layered on the (separate,
# NOT the RoSE in-tree) ExecuTorch 1.0.1 / torch 2.9 env; the RoSE shared zephyr
# env is torch 2.13 with no executorch and MUST NOT be touched.
set -uo pipefail
ETENV=/scratch2/dima/misc_sw/XPU-RT/zephyr-chipyard-sw/tools/miniforge3/envs/zephyr
E=/scratch/dima/rose-infra/RoSE/experiments/executorch
export PATH="$ETENV/bin:$PATH"          # flatc lives here (ET's serializer shells out to it)
export MB_REPO=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw
PY=$E/.venv/bin/python
for spec in "$@"; do
  m=${spec%%:*}; q=${spec##*:}
  echo "=== export $m ($q) ==="
  timeout 7200 $PY $E/scripts/export_mb_models.py --model "$m" --quant "$q" --outdir "$E/pte" \
      > "$E/logs/export_${m}.log" 2>&1
  rc=$?
  echo "  rc=$rc  $(grep -m1 provenance "$E/logs/export_${m}.log")"
  [ $rc -ne 0 ] && tail -6 "$E/logs/export_${m}.log"
done
