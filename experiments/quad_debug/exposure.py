#!/usr/bin/env python3
"""Exposure metric for the quad-only trap-path vector corruption.

The fault (see NOTES.md) needs an interrupt to land on a Saturn hart while
that hart still has a vector memop in flight.  Two factors set the per-run
probability, and both can be read straight out of the emitted dispatch table:

  ipi_src  number of harts that run a worker, minus one.  Every k_sem_give
           that readies a waiter ends in arch_sched_broadcast_ipi(), which
           pokes MSIP on every OTHER online hart
           (zephyr/arch/riscv/core/ipi_clint.c:arch_sched_directed_ipi), so a
           Saturn hart is interrupted by every other BUSY hart.  Idle harts
           emit nothing, which is why this counts busy harts and not
           CONFIG_MP_MAX_NUM_CPUS (=4 in every arm of this sweep).
  rvv_ms   predicted busy time on the Saturn harts (2 and 3): how long the
           vulnerable window is per run.

exposure = ipi_src * rvv_ms multiplies "how often" by "how long".  It is a
rate proxy for the hazard, NOT a prediction of which individual cell dies --
whether a given binary faults depends on where the interrupts actually land,
which is fixed per bitstream but arbitrary.

Usage: exposure.py [table_dir]
"""
import re
import sys
import glob
import os
import collections

GEN = (sys.argv[1] if len(sys.argv) > 1 else
       "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/"
       "modelblaster/examples/xpurt_demo_armB/int8/generated")

ROW = re.compile(r"\{\s*\.entry_id\s*=.*?\}", re.S)


def fld(row, name):
    return re.search(r"\.%s\s*=\s*(-?[0-9.eE+]+)f?" % name, row)


def scan(path):
    src = open(path).read()
    start = src.find("_TABLE[")
    if start < 0:
        return collections.Counter(), collections.Counter(), 0
    body = src[start:]
    harts = collections.Counter()
    ms = collections.Counter()
    fan = 0
    for row in ROW.findall(body):
        h = fld(row, "hart")
        if h is None:
            continue
        d = fld(row, "duration_ms")
        f = fld(row, "n_fanout")
        harts[int(h.group(1))] += 1
        ms[int(h.group(1))] += float(d.group(1)) if d else 0.0
        fan += int(f.group(1)) if f else 0
    return harts, ms, fan


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(GEN, "wl_*_greedy_*.c"))):
        if p.endswith("_main.c"):
            continue
        tag = os.path.basename(p)[:-2]
        harts, ms, fan = scan(p)
        if not harts:
            continue
        busy = sorted(harts)
        rvv_ms = ms[2] + ms[3]
        srcs = max(len(busy) - 1, 0)
        rows.append((tag, busy, sum(harts.values()), rvv_ms, fan,
                     srcs, srcs * rvv_ms))
    if not rows:
        print("no tables found under", GEN)
        return
    w = max(len(r[0]) for r in rows)
    print(f"{'tag'.ljust(w)}  harts         entries   rvv_ms  fanout  ipi_src   exposure")
    for r in sorted(rows, key=lambda r: -r[6]):
        print(f"{r[0].ljust(w)}  {str(r[1]).ljust(12)} {r[2]:7d} {r[3]:9.1f} {r[4]:7d} "
              f"{r[5]:7d} {r[6]:10.1f}")


main()
