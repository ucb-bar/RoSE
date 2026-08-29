#!/usr/bin/env bash
# fqsub.sh <local_elf> <key>   -- scp + submit one ELF, append "<key> <jobid> <resdir>" to jobs.txt
set -uo pipefail
DEST=/scratch/dima/rose-infra/RoSE/experiments/armA
MGR="ubuntu@3.88.218.39"; KEY="$HOME/.ssh/firesim.pem"
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 -i "$KEY")
HW="f2_dual_small_norose_tacit_q31_60mhz"; TREE="/home/ubuntu/chipyard-rose"
elf="$1"; k="$2"; TMO="${3:-5400}"
[[ -f "$elf" ]] || { echo "!! no ELF for $k at $elf"; exit 1; }
remote="/home/ubuntu/armA_${k}.elf"; res="/home/ubuntu/armA_${k}_res"
scp -q -o BatchMode=yes -i "$KEY" "$elf" "$MGR:$remote" || { echo "!! scp failed $k"; exit 1; }
"${SSH[@]}" "$MGR" "rm -rf $res && mkdir -p $res" >/dev/null 2>&1
out=$("${SSH[@]}" "$MGR" "cd ~/fpga_queue && export FQ_SOCKET=/var/lib/fq/fq.sock && \
      timeout 60 ./bin/fq submit --tree $TREE --hw-config $HW --elf $remote \
      --timeout $TMO --results $res 2>&1 | head -2")
id=$(grep -oE 'job [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)
echo "$k ${id:-NONE} $res" >> "$DEST/jobs.txt"
echo "submitted $k -> job ${id:-?} | $out"
