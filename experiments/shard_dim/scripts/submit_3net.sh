#!/usr/bin/env bash
# Run one built 3net benchmark ELF on an fq lane and pull the uartlog back.
#   submit_3net.sh <tag>        e.g. submit_3net.sh net3_hetero_greedy
set -uo pipefail
TAG=$1
R=/scratch/dima/rose-infra/RoSE; OUT=$R/experiments/3net
exec > $OUT/logs/fq_$TAG.log 2>&1
MGR=ubuntu@3.88.218.39; KEY=~/.ssh/firesim.pem
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY)
HW=f2_quad_hetero_norose_tacit_q31_60mhz
echo "### submit $TAG $(date -u +%FT%TZ)"
scp -q -o BatchMode=yes -i $KEY "$OUT/elf/$TAG.elf" $MGR:/home/ubuntu/n3_$TAG.elf || { echo "ABORT scp"; exit 1; }
"${SSH[@]}" $MGR "rm -rf /home/ubuntu/n3res_$TAG && mkdir -p /home/ubuntu/n3res_$TAG" </dev/null
O=$("${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; timeout 60 ./bin/fq submit \
  --tree /home/ubuntu/chipyard-rose --hw-config $HW --elf /home/ubuntu/n3_$TAG.elf \
  --timeout 5400 --results /home/ubuntu/n3res_$TAG 2>&1 | head -3" </dev/null)
echo "#### submit: $O"
JID=$(grep -oE 'job [0-9]+' <<<"$O" | grep -oE '[0-9]+' | head -1)
echo "#### jid=$JID"; [ -z "$JID" ] && { echo "ABORT no jid"; exit 1; }
for i in $(seq 1 320); do
  sleep 20
  "${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq status 2>/dev/null" </dev/null \
    | grep -qE "^ *$JID " || break
done
mkdir -p $OUT/res_$TAG
scp -q -o BatchMode=yes -i $KEY -r $MGR:/home/ubuntu/n3res_$TAG/. $OUT/res_$TAG/ 2>/dev/null
UL=$(find $OUT/res_$TAG -name uartlog | head -1); echo "#### uartlog=${UL:-NONE}"
# identity check: fq copies from the run host's sim_slot_*/ and those survive
# between jobs, so a cell can silently collect the previous job's uartlog.
if [ -n "$UL" ]; then
  S=$(grep -oE 'xpurt-runner: schedule=[^ ]+' "$UL" | head -1 | cut -d= -f2)
  [ "$S" = "$TAG" ] && echo "#### identity OK ($S)" || echo "#### IDENTITY MISMATCH: uartlog says '$S', expected '$TAG'"
  grep -E "entries=|MODELBLASTER_VERIFY|PASSED|FAILED" "$UL" | head -4
fi
echo "############ FQDONE $TAG ############"
