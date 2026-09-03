#!/usr/bin/env python3
"""Reduce a micro-ROS baseline uartlog to the numbers the U250 summary reported.

`data/microros_vs_xpurt_3net_summary.md` (in the pre-RoSE FreshScheduler tree)
characterised a run by: per-net dispatch count, iteration count, first/last
dispatch in ms, and -- the finding the whole 2026-05-12 fix turned on -- how
busy the SHARED hart is *while the one-shot network runs*. Reproduce exactly
those so an F2 run is comparable to the U250 numbers line for line.

  analyze_microros.py <trace.csv> [--anchor yolov8_nano]

<trace.csv> is what `python -m modelblaster.scripts.plot_ros_trace <uartlog>
--clock-mhz 1 --csv <trace.csv>` writes.
"""
import argparse
import collections
import csv
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--anchor", default="yolov8_nano",
                    help="one-shot network whose window the others are scored in")
    a = ap.parse_args()

    rows = []
    with open(a.csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((r["network"], int(r["instance"]), int(r["hart"]),
                         float(r["actual_start_ms"]), float(r["actual_end_ms"])))
    if not rows:
        print("no trace rows", file=sys.stderr)
        return 1

    by_net = collections.defaultdict(list)
    for net, inst, hart, s, e in rows:
        by_net[net].append((inst, hart, s, e))

    print(f"{'net':<14} {'disp':>5} {'iters':>6} {'harts':>7} "
          f"{'first_ms':>10} {'last_ms':>10} {'span_ms':>9} {'busy_ms':>9}")
    for net, v in sorted(by_net.items()):
        first = min(s for _, _, s, _ in v)
        last = max(e for _, _, _, e in v)
        busy = sum(e - s for _, _, s, e in v)
        harts = ",".join(str(h) for h in sorted({h for _, h, _, _ in v}))
        print(f"{net:<14} {len(v):>5} {len({i for i, _, _, _ in v}):>6} "
              f"{harts:>7} {first:>10.3f} {last:>10.3f} "
              f"{last - first:>9.3f} {busy:>9.3f}")

    if a.anchor not in by_net:
        print(f"\n(anchor {a.anchor} absent -- no window analysis)")
        return 0
    av = by_net[a.anchor]
    w0, w1 = min(s for _, _, s, _ in av), max(e for _, _, _, e in av)
    anchor_harts = {h for _, h, _, _ in av}
    print(f"\nwindow = {a.anchor} [{w0:.3f}, {w1:.3f}] ms  ({w1 - w0:.3f} ms)")
    # Overlap of every OTHER hart's dispatches with the anchor's window: the
    # "hart 1 utilization during yolov8" number the fix was measured by.
    per_hart = collections.defaultdict(float)
    per_hart_n = collections.Counter()
    per_net_in_win = collections.Counter()
    per_net_iters = collections.defaultdict(set)
    for net, inst, hart, s, e in rows:
        ov = max(0.0, min(e, w1) - max(s, w0))
        if ov <= 0:
            continue
        per_hart[hart] += ov
        per_hart_n[hart] += 1
        if net != a.anchor:
            per_net_in_win[net] += 1
            per_net_iters[net].add(inst)
    span = w1 - w0
    for hart in sorted(per_hart):
        tag = " (anchor)" if hart in anchor_harts else ""
        print(f"  hart {hart}: {per_hart[hart]:9.3f} ms busy = "
              f"{100.0 * per_hart[hart] / span:6.2f}%  "
              f"({per_hart_n[hart]} dispatches){tag}")
    for net in sorted(per_net_in_win):
        print(f"  {net}: {per_net_in_win[net]} dispatches / "
              f"{len(per_net_iters[net])} iters inside the {a.anchor} window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
