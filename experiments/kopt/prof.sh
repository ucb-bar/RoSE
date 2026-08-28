#!/usr/bin/env bash
# uartlog -> profile csv -> SoL table.  usage: prof.sh <uartlog> <backend>
set -euo pipefail
SP=/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/380445be-ecd7-4b3a-9531-bd33760b300b/scratchpad
LOG="$1"; BE="${2:-rvv}"
CSV="${LOG%.uartlog}.csv"
awk '/MODELBLASTER_PROFILE_BEGIN/{f=1;next} /MODELBLASTER_PROFILE_END/{f=0} f' "$LOG" | tr -d '\r' > "$CSV"
grep -o 'max_abs_err=[^ ]*' "$LOG" | head -1
python "$SP/sol_table.py" "$CSV" "$BE" 2>/dev/null
