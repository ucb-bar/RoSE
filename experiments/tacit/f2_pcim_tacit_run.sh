#!/usr/bin/env bash
# Runs on the AWS F2 MANAGER. Same shape as f2_tacit_run.sh, plus the two
# host-side prerequisites the PCIM host-DMA datapath needs and the BAR4 drain
# did not.
#
#   1. HUGEPAGES. simif_f2::allocate_to_cpu_buffer() maps one 2 MiB hugepage
#      per to-host stream (fpga_dma_mem_map_huge). The FPGA masters PCIM writes
#      with *physical* addresses, so a stream buffer has to be physically
#      contiguous for its whole length, which only a hugepage guarantees.
#      With HugePages_Total=0 every allocation fails and the driver exits.
#
#   2. BUS MASTER ENABLE. The F2 app PF comes up with COMMAND=0x0002 -- memory
#      space enabled, bus master CLEAR -- and programming the AFI rescans the
#      PFs and clears it again. Without BME the shell silently drops every PCIM
#      write, so the trace file would be created and stay empty. There is no
#      seam between fq's infrasetup (which programs) and its runworkload (which
#      starts the driver), so a watcher holds the bit set for the run.
#
# The LANE IS PINNED and the prerequisites are applied to that lane's host
# ONLY. The pool is shared: reserving hugepages on, or touching the PCI COMMAND
# register of, a host that is running someone else's job is not ours to do.
#
# Usage: f2_pcim_tacit_run.sh <tag> <lane> <host> [hw_config] [timeout_s]
set -uo pipefail
TAG="$1"; LANE="$2"; HOST="$3"
HW="${4:-f2_dual_small_norose_tacit_q31_60mhz_pcim}"
TO="${5:-2400}"
TREE="/home/ubuntu/chipyard-rose"
R="/home/ubuntu/r/$TAG"
export FQ_SOCKET=/var/lib/fq/fq.sock
cd ~/fpga_queue || exit 1

RSSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i /home/ubuntu/firesim.pem"

echo "== lane=$LANE host=$HOST hw=$HW tag=$TAG"

# Refuse to touch a host that is already busy -- the pool is shared.
busy=$($RSSH ubuntu@"$HOST" 'pgrep -c FireSim-f2 || true' </dev/null 2>/dev/null | tr -d '[:space:]')
if [ "${busy:-0}" != "0" ]; then
  echo "ABORT: $HOST already has $busy FireSim-f2 process(es) running"
  exit 1
fi

# sim_slot_*/ survives between jobs, so a run whose sim never started would
# otherwise hand back the PREVIOUS occupant's trace.
$RSSH ubuntu@"$HOST" 'rm -f /home/ubuntu/sim_slot_*/tacit*.out' </dev/null >/dev/null 2>&1
echo "== cleared stale tacit*.out on $HOST"

$RSSH ubuntu@"$HOST" \
  'sudo sysctl -w vm.nr_hugepages=64 >/dev/null; grep -E "HugePages_Total|HugePages_Free" /proc/meminfo | tr "\n" " "; echo' </dev/null 2>&1

# Idempotent watcher; harmless for non-PCIM designs (a design that never
# masters the bus is unaffected by being allowed to).
$RSSH ubuntu@"$HOST" \
  'nohup setsid bash -c "for i in \$(seq 1 3000); do for d in \$(lspci -D -n | grep 1d0f:f0 | cut -d\  -f1); do sudo setpci -s \$d COMMAND=0x0006 >/dev/null 2>&1; done; sleep 1; done" >/dev/null 2>&1 & echo bme-watcher-started' </dev/null 2>&1

./bin/fq submit --tree "$TREE" --hw-config "$HW" --elf "$R/z.elf" --lane "$LANE" \
   --timeout "$TO" --results "$R" --wait 2>&1 | tail -25
echo "== fq submit returned $?"

# fq's firesim backend hardcodes common_simulation_outputs=["uartlog"], so
# tacit<N>.out is never copied back by runworkload.
echo "== collecting tacit*.out from $HOST =="
mkdir -p "$R/tacit"
$RSSH ubuntu@"$HOST" 'ls -l /home/ubuntu/sim_slot_*/tacit*.out 2>/dev/null' </dev/null 2>/dev/null
# SIZE CAP. A multi-GB trace is the SIGNATURE OF A HUNG GUEST, not a result:
# a wedged hart emits the same tight loop at ~10 MB/s per tile forever. Copying
# it back is worse than useless -- on 2026-09-03 two such runs returned 5.8 GB
# and 12 GB and took this manager's 193 GB root filesystem to ZERO bytes free,
# killing an unrelated healthy job and orphaning a lane. Both deadlock
# diagnoses that day came from a 12 MB head slice, so cap the copy and note it.
CAP=${TACIT_CAP_BYTES:-$((64*1024*1024))}
for f in $($RSSH ubuntu@"$HOST" 'ls /home/ubuntu/sim_slot_*/tacit*.out 2>/dev/null' </dev/null 2>/dev/null); do
  b=$(basename "$f")
  sz=$($RSSH ubuntu@"$HOST" "stat -c %s '$f' 2>/dev/null" </dev/null 2>/dev/null)
  if [ "${sz:-0}" -gt "$CAP" ]; then
    echo "   $b is ${sz}B > cap ${CAP}B -- copying the first ${CAP}B only (guest likely hung)"
    $RSSH ubuntu@"$HOST" "head -c $CAP '$f'" </dev/null > "$R/tacit/${b%.out}.head.bin" 2>/dev/null
  else
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i /home/ubuntu/firesim.pem "ubuntu@$HOST:$f" "$R/tacit/" 2>/dev/null
  fi
done
ls -l "$R/tacit/" 2>/dev/null || echo "NOTHING COLLECTED"
echo "== uartlog tail =="
tail -30 "$R/uartlog" 2>/dev/null || echo "no uartlog"
