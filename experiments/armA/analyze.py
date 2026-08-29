#!/usr/bin/env python3
"""Predicted-vs-actual summary for an armA xpurt 3-net F2 uartlog."""
import sys, csv, io, re, json, collections

log = sys.argv[1]
text = open(log, errors="replace").read()
m = re.search(r"=== MODELBLASTER_XPURT_TRACE_BEGIN ===\n(.*?)=== MODELBLASTER_XPURT_TRACE_END ===",
              text, re.S)
if not m:
    raise SystemExit("no XPURT trace block")
rows = list(csv.DictReader(io.StringIO(m.group(1))))
# actual_*_cycles are k_cycle_get_64() mtime ticks = 1 us at the modeled 1 GHz
US = 1000.0
pred_end = 0.0; act_end = 0.0; act_start0 = min(int(r["actual_start_cycles"]) for r in rows)
busy_p = collections.defaultdict(float); busy_a = collections.defaultdict(float)
net_pred = collections.defaultdict(float); net_act = collections.defaultdict(float)
late = 0
for r in rows:
    ps = float(r["predicted_start_ms"]); pd = float(r["predicted_duration_ms"])
    a0 = (int(r["actual_start_cycles"]) - act_start0) / US
    a1 = (int(r["actual_end_cycles"])   - act_start0) / US
    k = r["core_kind"]
    pred_end = max(pred_end, ps+pd); act_end = max(act_end, a1)
    busy_p[k] += pd; busy_a[k] += (a1-a0)
    net_pred[r["network"]] = max(net_pred[r["network"]], ps+pd)
    net_act[r["network"]]  = max(net_act[r["network"]],  a1)
    if a1 > ps+pd: late += 1

wall = dict(re.findall(r"=== MODELBLASTER_WALL_CYCLES \[([^\]]+)\] === (\d+)", text))
walli = re.findall(r"=== MODELBLASTER_WALL_CYCLES_INST \[([^\]]+)\] === (\d+)", text)
ver = re.findall(r"=== MODELBLASTER_VERIFY(?: \[([^\]]+)\])? === max_abs_err=(\S+) max_rel_err=(\S+) n=(\d+)", text)

print(f"log            : {log}")
print(f"trace entries  : {len(rows)}")
print(f"predicted makespan : {pred_end:9.3f} ms")
print(f"actual   makespan  : {act_end:9.3f} ms   (pred/actual = {pred_end/act_end:.3f}, actual/pred = {act_end/pred_end:.3f})")
print(f"entries finishing later than predicted: {late}/{len(rows)}")
print("\nper-core (core_kind) busy:")
for k in sorted(busy_p):
    print(f"  {k:12s} predicted busy {busy_p[k]:8.3f} ms (util {100*busy_p[k]/pred_end:5.1f}%)"
          f"   actual busy {busy_a[k]:8.3f} ms (util {100*busy_a[k]/act_end:5.1f}%)")
print("\nper-network completion (ms):")
for n in sorted(net_pred):
    print(f"  {n:14s} predicted {net_pred[n]:8.3f}   actual {net_act[n]:8.3f}")
print("\nMODELBLASTER_WALL_CYCLES (us, = ms/1000):")
for n,v in wall.items(): print(f"  {n:14s} {int(v):9d} us = {int(v)/1000:7.3f} ms")
print("per-instance:", {k:int(v) for k,v in walli})
print("verify:", ver)
