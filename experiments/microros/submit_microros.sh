#!/usr/bin/env bash
# Ship one stashed micro-ROS baseline ELF to the AWS manager, run it on an fq
# lane, pull the uartlog back.  submit_microros.sh <tag>
# Same shape as experiments/shard_dim/scripts/sweep_submit.sh -- fq schedules
# across free lanes, so several of these can run concurrently.
set -uo pipefail
TAG=$1
R=/scratch/dima/rose-infra/RoSE; OUT=$R/experiments/microros
exec > $OUT/logs/fq_$TAG.log 2>&1
MGR=ubuntu@3.88.218.39; KEY=~/.ssh/firesim.pem
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i $KEY)
HW=f2_quad_hetero_norose_tacit_q31_60mhz
TIMEOUT=${FQ_TIMEOUT:-4000}
# Lane f2-07 (i-0a63a647be5a8949e) is on degraded hardware -- pin away from
# it rather than letting fq pick. Unset LANE to use whatever is free.
LANE=${LANE:-f2-05}
LANEARG=""; [ -n "$LANE" ] && LANEARG="--lane $LANE"
echo "### submit $TAG $(date -u +%FT%TZ) elf_mtime=$(date -r $OUT/elf/$TAG.elf -u +%FT%TZ 2>/dev/null)"
scp -q -o BatchMode=yes -i $KEY "$OUT/elf/$TAG.elf" $MGR:/home/ubuntu/uros_$TAG.elf || { echo "ABORT scp"; exit 1; }
"${SSH[@]}" $MGR "rm -rf /home/ubuntu/urosres_$TAG && mkdir -p /home/ubuntu/urosres_$TAG" </dev/null
O=$("${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; timeout 60 ./bin/fq submit \
  --tree /home/ubuntu/chipyard-rose --hw-config $HW --elf /home/ubuntu/uros_$TAG.elf \
  --timeout $TIMEOUT $LANEARG --results /home/ubuntu/urosres_$TAG 2>&1 | head -3" </dev/null)
echo "#### submit: $O"
JID=$(grep -oE 'job [0-9]+' <<<"$O" | grep -oE '[0-9]+' | head -1)
echo "#### jid=$JID"; [ -z "$JID" ] && { echo "ABORT no jid"; exit 1; }
for i in $(seq 1 300); do
  sleep 20
  "${SSH[@]}" $MGR "cd /home/ubuntu/fpga_queue 2>/dev/null || cd /home/ubuntu; ./bin/fq status 2>/dev/null" </dev/null \
    | grep -qE "^ *$JID " || break
done
mkdir -p $OUT/res_$TAG
scp -q -o BatchMode=yes -i $KEY -r $MGR:/home/ubuntu/urosres_$TAG/. $OUT/res_$TAG/ 2>/dev/null
UL=$(find $OUT/res_$TAG -name uartlog | head -1); echo "#### uartlog=${UL:-NONE} bytes=$(stat -c%s "${UL:-/dev/null}" 2>/dev/null)"
[ -n "$UL" ] && grep -E "MODELBLASTER_WALL_CYCLES|ROS_TRACE_END|MODELBLASTER_VERIFY|PASSED|FAILED" $UL | head -10
echo "############ FQDONE $TAG ############"
