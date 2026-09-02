#!/usr/bin/env python3
"""Work conservation vs split degree k, measured against the model.

  analyze_sweep.py <sweep_uartlog> <baseline_uartlog> <sweep_graph> <backend> <out_csv>

Metric: work_ratio = sum(tile times) / unsplit time.  1.0 = perfect split;
k = total duplication.  Because every tile ran serially on one hart, this is
work conservation ONLY -- no concurrency tax is folded in.
"""
import csv, json, math, sys, os
sys.path.insert(0, "/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/380445be-ecd7-4b3a-9531-bd33760b300b/scratchpad/quad")
from qcollect import parse

V = 32  # vlmax_oc for rvv conv at VLEN=256, e32m4 -- source-verified

def predict(backend, k, oc, floor_us=None, t_full=None):
    if backend == "rvv":
        return k * math.ceil((oc / k) / V) / math.ceil(oc / V)
    if floor_us and t_full:                     # gemmini: transpose re-run per tile
        return 1.0 + (k - 1) * floor_us / t_full
    return float("nan")

def main():
    sw, base, graph, backend, out = sys.argv[1:6]
    S, B = parse(sw), parse(base)
    g = json.load(open(graph))
    ops = [o for o in g["ops"] if o.get("dispatch_id") is not None]
    inv = {}
    for old, news in (g.get("id_remap") or {}).items():
        for n in news: inv[int(n)] = int(old)
    bname = {v["name"]: v["us"] for v in B["prof"].values()}
    # group measured tile times by parent op
    grp = {}
    for did, rec in S["prof"].items():
        parent = inv.get(did, did)
        base_name = rec["name"].split(".tile_")[0]
        grp.setdefault(base_name, {"tiles": [], "op": rec["op"], "shape": rec["shape"]})
        grp[base_name]["tiles"].append(rec["us"])
    rows = []
    for name, d in sorted(grp.items()):
        if name not in bname: continue
        k = len(d["tiles"]); tf = bname[name]
        if k < 2:
            rows.append(dict(name=name, OC="", k=1, tile_w="", t_unsplit_us=round(tf, 1),
                             sum_tiles_us=round(sum(d["tiles"]), 1),
                             work_ratio=round(sum(d["tiles"]) / tf, 3),
                             predicted="", err="",
                             cost_P2_us=round(tf, 1), gain_P2_pct=0.0,
                             cost_P4_us=round(tf, 1), gain_P4_pct=0.0))
            continue
        sh = dict(p.split("=") for p in d["shape"].split(";") if "=" in p)
        oc = int(sh.get("OC", 0)) * k if "OC" in sh else 0   # tile shape carries OC/k
        wr = sum(d["tiles"]) / tf
        pr = predict(backend, k, oc)
        rows.append(dict(name=name, OC=oc, k=k, tile_w=oc // k,
                         t_unsplit_us=round(tf, 1), sum_tiles_us=round(sum(d["tiles"]), 1),
                         work_ratio=round(wr, 3),
                         predicted=round(pr, 3) if pr == pr else "",
                         err=round(wr - pr, 3) if pr == pr else "",
                         cost_P2_us=round(math.ceil(k / 2) * (sum(d["tiles"]) / k), 1),
                         gain_P2_pct=round(100 * (math.ceil(k / 2) * (sum(d["tiles"]) / k) - tf) / tf, 1),
                         cost_P4_us=round(math.ceil(k / 4) * (sum(d["tiles"]) / k), 1),
                         gain_P4_pct=round(100 * (math.ceil(k / 4) * (sum(d["tiles"]) / k) - tf) / tf, 1)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'op':<18} {'OC':>4} {'k':>2} {'tile_w':>7} {'unsplit':>8} {'sum_tiles':>10} "
          f"{'ratio':>7} {'pred':>6} {'err':>7} {'P=2 cost':>9} {'gain':>7} {'P=4 cost':>9} {'gain':>7}")
    print("  cost on a P-hart pair = ceil(k/P)*tile_time. THIS SoC gives P=2 per same-kind pair;")
    print("  the P=4 column is what a future 4-unit-of-one-kind SoC would buy.")
    errs = []
    for r in rows:
        if r["err"] != "": errs.append(abs(r["err"]))
        print(f"{r['name']:<18} {str(r['OC']):>4} {r['k']:>2} {str(r['tile_w']):>7} "
              f"{r['t_unsplit_us']:>8.1f} {r['sum_tiles_us']:>10.1f} {r['work_ratio']:>7.3f} "
              f"{str(r['predicted']):>6} {str(r['err']):>7} {r['cost_P2_us']:>9.1f} "
              f"{r['gain_P2_pct']:>6.1f}% {r['cost_P4_us']:>9.1f} {r['gain_P4_pct']:>6.1f}%")
    if errs:
        errs.sort()
        print(f"\n  model error: mean {sum(errs)/len(errs):.3f}x  median {errs[len(errs)//2]:.3f}x  "
              f"max {errs[-1]:.3f}x  over {len(errs)} (op,k) cells")
    print(f"  -> {out}")
main()
