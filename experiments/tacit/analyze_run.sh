#!/usr/bin/env bash
# analyze_run.sh <trace.out> <zephyr.elf> [expected_cycles]
# Runs the two decisive stream checks from F2_TACIT_VERDICT.md section 2d plus
# a full decode, on one tacit<N>.out.
set -uo pipefail
T="$1"; E="$2"; EXP="${3:-}"
R=/scratch/dima/rose-infra/RoSE
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
echo "############ integrity: $(basename "$T") ############"
python3 "$R/experiments/tacit/integrity.py" "$T" $EXP 2>&1 | tail -9
cat > "$OUT/cfg.json" <<JSON
{ "encoded_trace": "$(readlink -f "$T")",
  "application_binary_asid_tuples": [ ["$(readlink -f "$E")", "0"] ],
  "sbi_binary": "$(readlink -f "$E")",
  "kernel_binary": "", "kernel_jump_label_patch_log": "", "driver_binary_entry_tuples": [],
  "header_only": false, "to_stats": true, "to_txt": false, "to_stack_txt": false,
  "to_atomics": false, "to_afdo": false, "gcno": "", "to_gcda": false,
  "to_speedscope": false, "to_perfetto": false, "to_vbb": true, "to_path_profile": false }
JSON
echo "############ decode ############"
( cd "$OUT" && timeout 1800 "$R/tools/tacit-decoder/target/release/tacit-decoder" --config "$OUT/cfg.json" 2>&1 | tail -8 )
echo "############ hottest basic blocks ############"
sort -t, -k3 -rn "$OUT/trace.vbb.csv" 2>/dev/null | head -10
cp "$OUT/trace.vbb.csv" "$(dirname "$T")/$(basename "$T" .out).vbb.csv" 2>/dev/null
