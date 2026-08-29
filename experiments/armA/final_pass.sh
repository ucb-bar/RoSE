#!/usr/bin/env bash
# ARM A final pass: ingest post-fix gemmini profiles -> schedule -> harness build.
set -uo pipefail
D=/scratch/dima/rose-infra/RoSE/experiments/armA
X=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt
cd "$D"
python3 ingest_uartlog.py p2_dronet_gemmini_q31.uartlog dronet int8 gemmini_q31 firesim_f2_armA || exit 1
python3 ingest_uartlog.py p2_yolov8_nano_gemmini_q31.uartlog yolov8_nano int8 gemmini_q31 firesim_f2_armA || exit 1
cd "$X"
/scratch2/dima/miniforge3/envs/xpurt/bin/python scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_3net_armA.json --solver greedy --profiled \
  > "$D/schedule.log" 2>&1
echo "scheduler rc=$? -- makespan:"
grep -E "Final greedy makespan|Makespan \(non-periodic\)|needed=" "$D/schedule.log"
cd "$D" && ./build_harness.sh > harness2.build.log 2>&1
echo "harness build rc=$?"
grep -E "^ingested|^ELF:" harness2.build.log
