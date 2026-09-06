"""How wide is the seed spread, and is any margin smaller than it?

Compares seed 0 (from the main sweep) with seeds 1 and 2 on the same
workload-arms. A solver-vs-solver gap narrower than the seed spread is not a
result, and this is the only thing in the study that can say which gaps those
are.
"""
import json, glob, os, sys
import numpy as np

base_dir, var_dir, dest = sys.argv[1], sys.argv[2], sys.argv[3]
base = {}
for f in glob.glob(os.path.join(base_dir, "*.json")):
    r = json.load(open(f))
    if "workload" in r:
        base[(r["arm"], r["workload"], r["solver"])] = r
var = {}
for f in glob.glob(os.path.join(var_dir, "*.json")):
    r = json.load(open(f))
    if "workload" in r:
        var.setdefault((r["arm"], r["workload"], r["solver"]), {})[r.get("seed", 0)] = r

out, lines = {}, []
lines.append(f"{'solver':14s}{'n':>4}{'mean spread %':>15}{'median spread %':>17}"
             f"{'max spread %':>14}{'mean |d| vs s0 %':>18}")
for solver in ("pso", "sa", "cpsat", "cpsat:warm"):
    spreads, devs = [], []
    for k, seeds in sorted(var.items()):
        if k[2] != solver:
            continue
        b = base.get(k)
        vals = []
        if b and b.get("objective") is not None:
            vals.append(("s0", b["objective"], b.get("misses", 0)))
        for sd, r in sorted(seeds.items()):
            if r.get("objective") is not None:
                vals.append((f"s{sd}", r["objective"], r.get("misses", 0)))
        if len(vals) < 2:
            continue
        o = np.array([v[1] for v in vals])
        spreads.append(float((o.max() - o.min()) / o.mean() * 100))
        if b and b.get("objective"):
            devs.extend(abs(v[1] - b["objective"]) / b["objective"] * 100
                        for v in vals[1:])
    if spreads:
        out[solver] = dict(n=len(spreads), mean_spread_pct=round(float(np.mean(spreads)), 4),
                           median_spread_pct=round(float(np.median(spreads)), 4),
                           max_spread_pct=round(float(np.max(spreads)), 4),
                           mean_abs_dev_vs_seed0_pct=round(float(np.mean(devs)), 4) if devs else None)
        v = out[solver]
        lines.append(f"{solver:14s}{v['n']:>4}{v['mean_spread_pct']:>15.4f}"
                     f"{v['median_spread_pct']:>17.4f}{v['max_spread_pct']:>14.4f}"
                     f"{v['mean_abs_dev_vs_seed0_pct']:>18.4f}")
json.dump(out, open(os.path.join(dest, "variance.json"), "w"), indent=1)
txt = "\n".join(lines)
open(os.path.join(dest, "variance.txt"), "w").write(
    "Seed spread over {s0,s1,s2} on 20 workload-arms (10 families x "
    "{base/gempair, shard/quad}); spread = (max-min)/mean of the objective.\n\n"
    + txt + "\n")
print(txt)
