#!/usr/bin/env python3
"""Per-arm heterogeneity metrics from a pair of results.csv (gemmini, saturn)."""
import csv, sys, os

def load(p):
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if not r.get("dispatch_id"):
                continue
            d[int(r["dispatch_id"])] = (float(r["mean_time"]), r.get("op", ""))
    return d

def main(gpath, spath, label):
    g, s = load(gpath), load(spath)
    ids = sorted(set(g) & set(s))
    miss = (set(g) ^ set(s))
    tg = sum(g[i][0] for i in ids)
    ts = sum(s[i][0] for i in ids)
    best = sum(min(g[i][0], s[i][0]) for i in ids)
    worst = sum(max(g[i][0], s[i][0]) for i in ids)
    ng = sum(1 for i in ids if g[i][0] <= s[i][0])
    print(f"--- {label}")
    print(f"  dispatches            {len(ids)}" + (f"  (asym ids: {sorted(miss)})" if miss else ""))
    print(f"  all-on-Gemmini  ms    {tg:.2f}")
    print(f"  all-on-Saturn   ms    {ts:.2f}")
    print(f"  best-per-dispatch ms  {best:.2f}   (gemmini wins {ng}, saturn wins {len(ids)-ng})")
    print(f"  worst-per-dispatch ms {worst:.2f}")
    print(f"  specialisation gain   {min(tg,ts)/best:.2f}x  (vs best single core)")
    # per-op breakdown
    agg = {}
    for i in ids:
        op = g[i][1] or s[i][1]
        a = agg.setdefault(op, [0,0.0,0.0,0.0])
        a[0]+=1; a[1]+=g[i][0]; a[2]+=s[i][0]; a[3]+=min(g[i][0], s[i][0])
    print(f"  {'op':28s}{'n':>5}{'gem ms':>12}{'sat ms':>12}{'best ms':>12}")
    for op, a in sorted(agg.items(), key=lambda kv:-kv[1][3]):
        print(f"  {op:28s}{a[0]:5d}{a[1]:12.2f}{a[2]:12.2f}{a[3]:12.2f}")
    return dict(n=len(ids), gem=tg, sat=ts, best=best)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "arm")
