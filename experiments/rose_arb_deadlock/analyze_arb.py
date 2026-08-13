#!/usr/bin/env python3
"""Classify the [ARB] arbiter-counter trace from a hung/instrumented gate-nav flight.
Reads the driver uartlog, extracts the [ARB] heartbeats, and reports which delivery
stage flatlined at the hang -> where the 0x42 reqrsp response dies.

[ARB] sh=<state_sheader> tx=<tx_fired> rx0=<ch0> rx1=<ch1> ch2~=<tx-rx0-rx1>
      bf=<budget_fired> cyc=<cycle_count> budg=<cycle_budget>
      last=0x<last_sched_cmd> nb=<last_nb> brxq=<budget_rx depth> txd=<fsim_txdata depth>
"""
import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "arb_driver_uartlog.txt"
pat = re.compile(
    r"\[ARB\] sh=(\d+) tx=(\d+) rx0=(\d+) rx1=(\d+) ch2~=(-?\d+) bf=(\d+) "
    r"cyc=(\d+) budg=(\d+) last=0x([0-9a-fA-F]+) nb=(\d+) brxq=(\d+) txd=(\d+)")
rows = []
with open(path, errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if m:
            g = m.groups()
            rows.append(dict(sh=int(g[0]), tx=int(g[1]), rx0=int(g[2]), rx1=int(g[3]),
                             ch2=int(g[4]), bf=int(g[5]), cyc=int(g[6]), budg=int(g[7]),
                             last=int(g[8], 16), nb=int(g[9]), brxq=int(g[10]), txd=int(g[11])))

if not rows:
    print("NO [ARB] lines found in", path)
    sys.exit(1)

print(f"parsed {len(rows)} [ARB] heartbeats")
print("first:", rows[0])
print("last :", rows[-1])

# Find the flatline point: last index where sh (or tx) still changed.
def last_change(key):
    for i in range(len(rows) - 1, 0, -1):
        if rows[i][key] != rows[i-1][key]:
            return i
    return 0

tail = rows[-8:]
print("\n-- last 8 heartbeats --")
for r in tail:
    print(f"  sh={r['sh']} tx={r['tx']} rx0={r['rx0']} rx1={r['rx1']} ch2~={r['ch2']} "
          f"bf={r['bf']} cyc={r['cyc']} budg={r['budg']} last=0x{r['last']:x} nb={r['nb']} "
          f"brxq={r['brxq']} txd={r['txd']}")

a, b = rows[0], rows[-1]
sh_moved  = b['sh']  != a['sh']
tx_moved  = b['tx']  != a['tx']
cyc_moved = b['cyc'] != a['cyc']
# flatline over the last quarter of the trace:
q = rows[max(0, len(rows)*3//4)]
sh_flat_tail  = (b['sh']  == q['sh'])
tx_flat_tail  = (b['tx']  == q['tx'])
ch2_flat_tail = (b['ch2'] == q['ch2'])
cyc_flat_tail = (b['cyc'] == q['cyc'])

print("\n== CLASSIFICATION ==")
print(f"cycle_count moving at tail (grants flowing): {not cyc_flat_tail}")
print(f"state_sheader flat at tail: {sh_flat_tail} | tx_fired flat at tail: {tx_flat_tail} | ch2~ flat at tail: {ch2_flat_tail}")
print(f"last scheduled cmd at hang: 0x{b['last']:x} (nb={b['nb']}) | host queues brxq={b['brxq']} txd={b['txd']}")

if cyc_flat_tail:
    print(">> Grants NOT flowing at tail — co-sim ended/terminated here, not a live hang sample. "
          "Re-check window (look earlier for the transition).")
elif sh_flat_tail and not cyc_flat_tail:
    print(">> OUTCOME A: arbiter STUCK mid-packet (state_sheader flat while grants flow). "
          f"Arbiter never returns to sIdle -> framing/sLoad stall on last=0x{b['last']:x}. "
          f"txd={b['txd']} (>0 => words stuck at MMIO send backpressure; ==0 => FPGA has them, RTL sLoad wedged).")
elif (not sh_flat_tail) and ch2_flat_tail:
    print(">> OUTCOME B: arbiter processes headers but ch2 delivery DEAD (ch2~ flat, sh advancing). "
          "rx(2).ready stuck low -> guest ch2 FIFO not drained OR routing_table[0x42] wrong.")
elif not sh_flat_tail and not ch2_flat_tail:
    print(">> OUTCOME C: all counters advance past the hang -> response IS delivered; "
          "guest-side hang (rose_recv_reqrsp ch2 / guest vision logic). Bug is in the guest firmware.")
else:
    print(">> Ambiguous — inspect the last-8 table by hand.")
