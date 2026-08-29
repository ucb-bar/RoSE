#!/usr/bin/env bash
# Submit the BN A/B ELFs to the F2 fq queue, wait, collect uartlogs.
# usage: submit.sh <arm_target> [<arm_target> ...]   e.g. bnfold_gemmini_q31
set -uo pipefail
MB=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster
DEST=/scratch/dima/rose-infra/RoSE/experiments/bnfold
MGR="ubuntu@3.88.218.39"; KEY="$HOME/.ssh/firesim.pem"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -i "$KEY")
HW="f2_dual_small_norose_tacit_q31_60mhz"; TREE="/home/ubuntu/chipyard-rose"

declare -A JOB RES
for k in "$@"; do
  arm="${k%%_*}"; tgt="${k#*_}"
  elf="$MB/examples/yolov8_nano_${arm}/int8/build/${tgt}_firesim/zephyr/zephyr.elf"
  [[ -f "$elf" ]] || { echo "!! no ELF for $k at $elf"; continue; }
  remote="/home/ubuntu/bnab_${k}.elf"; res="/home/ubuntu/bnab_${k}_res"
  scp -q -o BatchMode=yes -i "$KEY" "$elf" "$MGR:$remote" || { echo "!! scp failed $k"; continue; }
  "${SSH[@]}" -n "$MGR" "rm -rf $res && mkdir -p $res" >/dev/null 2>&1
  out=$("${SSH[@]}" -n "$MGR" "cd ~/fpga_queue && export FQ_SOCKET=/var/lib/fq/fq.sock && \
        timeout 60 ./bin/fq submit --tree $TREE --hw-config $HW --elf $remote \
        --timeout 5400 --results $res 2>&1 | head -2")
  id=$(grep -oE 'job [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)
  JOB[$k]="$id"; RES[$k]="$res"
  echo "submitted $k -> job ${id:-?}  ($out)"
  sleep 12
done

echo "=== waiting for ${#JOB[@]} jobs ..."
for i in $(seq 1 400); do
  pending=0
  for k in "${!JOB[@]}"; do
    st=$("${SSH[@]}" -n "$MGR" "grep -E 'job ${JOB[$k]} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1")
    [[ -z "$st" ]] && pending=$((pending+1))
  done
  echo "  [$(date +%T)] pending=$pending"
  [[ $pending -eq 0 ]] && break
  sleep 30
done

for k in "${!JOB[@]}"; do
  verdict=$("${SSH[@]}" -n "$MGR" "grep -E 'job ${JOB[$k]} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1")
  echo "--- $k: ${verdict:-NO TERMINAL STATE}"
  "${SSH[@]}" -n "$MGR" "find ${RES[$k]} -name uartlog | head -1 | xargs -r cat" > "$DEST/$k.uartlog" 2>/dev/null
  got=$(grep -ao "bnab_${k}\.elf" "$DEST/$k.uartlog" | head -1)
  echo "    bytes=$(stat -c %s "$DEST/$k.uartlog" 2>/dev/null) elfname='${got}'"
done
