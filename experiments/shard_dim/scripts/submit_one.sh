#!/usr/bin/env bash
# Ship one stashed ELF to the AWS manager, submit ONE fq job, wait, pull back.
# Safe to run four of these concurrently -- the fq pool has four lanes.
#   submit_one.sh <tag>
set -uo pipefail
TAG=$1
R=/scratch/dima/rose-infra/RoSE
OUT=$R/experiments/shard_dim/results/oh; ELFD=$OUT/elf
exec > $OUT/fq_$TAG.log 2>&1
MGR=ubuntu@3.88.218.39; KEY=~/.ssh/firesim.pem
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY)
HW=f2_quad_hetero_norose_tacit_q31_60mhz
echo "### submit $TAG $(date -u +%FT%TZ)"
scp -q -o BatchMode=yes -i $KEY "$ELFD/$TAG.elf" $MGR:/home/ubuntu/oh_$TAG.elf || { echo "ABORT scp"; exit 1; }
"${SSH[@]}" $MGR "rm -rf /home/ubuntu/ohres_$TAG && mkdir -p /home/ubuntu/ohres_$TAG" </dev/null
O=$("${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; timeout 60 ./bin/fq submit \
  --tree /home/ubuntu/chipyard-rose --hw-config $HW \
  --elf /home/ubuntu/oh_$TAG.elf --timeout 3000 --results /home/ubuntu/ohres_$TAG 2>&1 | head -3" </dev/null)
echo "#### submit: $O"
JID=$(grep -oE 'job [0-9]+' <<<"$O" | grep -oE '[0-9]+' | head -1)
echo "#### jid=$JID"
[ -z "$JID" ] && { echo "#### ABORT no jid"; exit 1; }
for i in $(seq 1 220); do
  sleep 20
  "${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq status 2>/dev/null" </dev/null \
    | grep -qE "^ *$JID " || break
done
mkdir -p $OUT/res_$TAG
scp -q -o BatchMode=yes -i $KEY -r $MGR:/home/ubuntu/ohres_$TAG/. $OUT/res_$TAG/ 2>/dev/null
UL=$(find $OUT/res_$TAG -name uartlog | head -1)
echo "#### uartlog=${UL:-NONE}"
[ -n "$UL" ] && grep -E "entries=|MODELBLASTER_VERIFY|PASSED|FAILED" $UL | head -5
echo "############ FQDONE $TAG ############"
