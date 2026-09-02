#!/usr/bin/env python3
"""Collect the 2-backend sharding sweep into one table.

Each cell is (network, machine pair, arm). Both arms of a pair run the SAME
schedule slots, so base->shard is a like-for-like comparison on one machine.

  analyze_sweep3net.py [net ...]
"""
import collections, glob, json, os, re, sys

OUT = "/scratch/dima/rose-infra/RoSE/experiments/sweep3net"
SCHED = "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/ohsched"
PAIRS = ["rvvpair", "gempair", "hetero"]
LABEL = {"rvvpair": "rvv + rvv", "gempair": "gemmini + gemmini",
         "hetero": "rvv + gemmini"}


def cell(tag):
    c = glob.glob(f"{OUT}/res_{tag}/**/uartlog", recursive=True) + [f"{OUT}/res_{tag}/uartlog"]
    p = next((x for x in c if os.path.exists(x)), None)
    if not p:
        return None
    t = open(p, errors="ignore").read()
    rows = [l.split(",") for l in t.split("\n") if re.match(r"^\d+,", l) and len(l.split(",")) > 8]
    iv = []
    for r in rows:
        try:
            iv.append((int(r[-2]), int(r[-1])))
        except ValueError:
            pass
    if not iv:
        return None
    span = (max(b for _, b in iv) - min(a for a, _ in iv)) / 1000.0
    err = re.search(r"max_abs_err=([0-9.eE+-]+)", t)
    harts = sorted({r[-3] for r in rows})
    pred = None
    sp = f"{SCHED}/{tag}.json"
    if os.path.exists(sp):
        d = json.load(open(sp))["dispatches"]
        pred = max(v["start_time"] + v["duration"] for v in d.values()) * 1e3
    return dict(ms=span, pred=pred, err=(err.group(1) if err else "?"),
                passed="PASSED" in t, n=len(rows), harts=harts)


def main():
    nets = sys.argv[1:] or sorted({os.path.basename(d).split("_")[0]
                                   for d in glob.glob(f"{OUT}/res_*")})
    print(f"  {'network':<13}{'machine pair':<20}{'unsplit':>10}{'sharded':>10}"
          f"{'speedup':>9}{'pred us':>10}{'err%':>7}  gates")
    print("  " + "-" * 92)
    for net in nets:
        for pair in PAIRS:
            b, s = cell(f"{net}_{pair}_base"), cell(f"{net}_{pair}_shard")
            if not b and not s:
                continue
            bm = f"{b['ms']:.3f}" if b else "-"
            sm = f"{s['ms']:.3f}" if s else "-"
            sp = f"{b['ms']/s['ms']:.2f}x" if (b and s) else "-"
            pe = (f"{100*(s['ms']-s['pred'])/s['pred']:+.1f}"
                  if (s and s.get("pred")) else "-")
            pr = f"{s['pred']*1000:.0f}" if (s and s.get("pred")) else "-"
            g = []
            for nm, c in (("base", b), ("shard", s)):
                if c:
                    g.append(f"{nm}:{'PASS' if c['passed'] else 'FAIL'}/err={c['err']}")
            print(f"  {net:<13}{LABEL[pair]:<20}{bm:>10}{sm:>10}{sp:>9}{pr:>10}{pe:>7}  {' '.join(g)}")
    print("\n  speedup = unsplit / sharded on the SAME machine pair;"
          " pred/err% are the schedule's prediction vs measurement.")


main()
