#!/usr/bin/env bash
# Submit a guest ELF to the AWS F2 fq pool and bring back the uartlog.
# usage: fq_run.sh <local-elf> <tag> [timeout]
set -euo pipefail
ELF="$1"; TAG="$2"; TO="${3:-3000}"
MGR="ubuntu@3.88.218.39"
KEY="$HOME/.ssh/firesim.pem"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $KEY"
HW="f2_dual_small_norose_tacit_q31_60mhz"
TREE="/home/ubuntu/chipyard-rose"
R="/home/ubuntu/r/$TAG"          # SHORT path: AF_UNIX ~108B cap

[ -f "$ELF" ] || { echo "no elf: $ELF" >&2; exit 2; }
$SSH $MGR "mkdir -p $R"
scp -q -o StrictHostKeyChecking=no -i "$KEY" "$ELF" "$MGR:$R/z.elf"
$SSH $MGR "cd ~/fpga_queue && FQ_SOCKET=/var/lib/fq/fq.sock ./bin/fq submit \
  --tree $TREE --hw-config $HW --elf $R/z.elf \
  --timeout $TO --results $R --wait --quiet" >/dev/null 2>&1 || true
OUT="$(dirname "$ELF")/../../../fq_$TAG.uartlog"
mkdir -p "$(dirname "$OUT")"
$SSH $MGR "cat $R/uartlog 2>/dev/null || find $R -name 'uartlog*' -exec cat {} \; 2>/dev/null" > "$OUT"
echo "$OUT"
wc -l < "$OUT"
