#!/usr/bin/env bash
# fqcollect.sh [jobsfile] -- wait on every job listed, pull uartlogs, provenance-check
set -uo pipefail
DEST=/scratch/dima/rose-infra/RoSE/experiments/armB_int8drain
JOBS="${1:-$DEST/jobs.txt}"
MGR="ubuntu@3.88.218.39"; KEY="$HOME/.ssh/firesim.pem"
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 -i "$KEY")
for i in $(seq 1 600); do
  pending=0
  while read -r k id res; do
    [[ -z "${id:-}" || "$id" == "NONE" ]] && continue
    st=$("${SSH[@]}" "$MGR" "grep -E 'job ${id} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1")
    [[ -z "$st" ]] && pending=$((pending+1))
  done < "$JOBS"
  echo "  [$(date +%T)] pending=$pending"
  [[ $pending -eq 0 ]] && break
  sleep 30
done
while read -r k id res; do
  verdict=$("${SSH[@]}" "$MGR" "grep -E 'job ${id} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1")
  echo "--- $k (job $id): ${verdict:-NO TERMINAL STATE}"
  "${SSH[@]}" "$MGR" "find ${res} -name uartlog | head -1 | xargs -r cat" > "$DEST/$k.uartlog" 2>/dev/null
  got=$(grep -ao "armB_${k}\.elf" "$DEST/$k.uartlog" | head -1)
  mdl=$(grep -o 'harness: model=[A-Za-z0-9_]*' "$DEST/$k.uartlog" | head -1 | cut -d= -f2)
  echo "    bytes=$(stat -c %s "$DEST/$k.uartlog" 2>/dev/null) elf='${got}' model='${mdl}'"
done < "$JOBS"
