#!/usr/bin/env python3
"""predicted-vs-actual metrics from an xpurt uartlog trace block.

usage: xpurt_metrics.py <uartlog> [--clock-mhz 1]
Emits the makespan / per-network busy / per-core busy table.
"""
import sys, csv, io, collections, json

BEGIN = "=== MODELBLASTER_XPURT_TRACE_BEGIN ==="
END   = "=== MODELBLASTER_XPURT_TRACE_END ==="

path = sys.argv[1]
clock_mhz = 1.0
if "--clock-mhz" in sys.argv:
    clock_mhz = float(sys.argv[sys.argv.index("--clock-mhz")+1])

text = open(path, errors="replace").read()
blk = text.split(BEGIN)[1].split(END)[0]
lines = [l for l in blk.splitlines() if l.strip()]
hdr = next(i for i,l in enumerate(lines) if l.startswith("entry_id"))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[hdr:]))))
rows = [r for r in rows if r.get("entry_id") and r["entry_id"].isdigit()]

def f(r,k): return float(r[k])
for r in rows:
    r["_ps"] = f(r,"predicted_start_ms")
    r["_pd"] = f(r,"predicted_duration_ms")
    r["_pe"] = r["_ps"] + r["_pd"]
    r["_as"] = int(r["actual_start_cycles"]) / (clock_mhz*1000.0)
    r["_ae"] = int(r["actual_end_cycles"])   / (clock_mhz*1000.0)
    r["_ad"] = max(r["_ae"] - r["_as"], 0.0)

pred_mk = max(r["_pe"] for r in rows)
act_mk  = max(r["_ae"] for r in rows)

def group(key):
    p = collections.defaultdict(float); a = collections.defaultdict(float)
    for r in rows:
        p[key(r)] += r["_pd"]; a[key(r)] += r["_ad"]
    return p, a

pn, an = group(lambda r: r["network"])
pc, ac = group(lambda r: r["core_kind"])
overrun = [r for r in rows if r["_ae"] > r["_pe"] + 1e-3]
worst = max((r["_ae"]-r["_pe"] for r in overrun), default=0.0)
insts = collections.defaultdict(set)
for r in rows: insts[r["network"]].add(int(r["instance"]))

out = {
  "entries": len(rows),
  "predicted_makespan_ms": round(pred_mk,3),
  "actual_makespan_ms": round(act_mk,3),
  "ratio_actual_over_predicted": round(act_mk/pred_mk,3),
  "instances": {k: len(v) for k,v in sorted(insts.items())},
  "per_network_busy_ms": {k: {"predicted": round(pn[k],3), "actual": round(an[k],3),
                              "ratio": round(an[k]/pn[k],3) if pn[k] else None}
                          for k in sorted(pn)},
  "per_core_busy_ms": {k: {"predicted": round(pc[k],3), "actual": round(ac[k],3),
                           "ratio": round(ac[k]/pc[k],3) if pc[k] else None,
                           "utilisation_actual": round(ac[k]/act_mk,3)}
                       for k in sorted(pc)},
  "entries_over_predicted_finish": len(overrun),
  "worst_overrun_ms": round(worst,3),
}
print(json.dumps(out, indent=2))
