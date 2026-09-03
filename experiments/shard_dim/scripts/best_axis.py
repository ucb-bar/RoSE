#!/usr/bin/env python3
"""Pick each conv's split axis from MEASURED tile costs, not a model.

  best_axis.py <net> [--pair rvvpair|gempair|hetero] [--emit]

The previous best-axis result (dronet 1.88x on 2 rvv harts vs 1.37x all-OC) was
chosen against a cost model, and measured tile costs have since shown that model
under-prices OC and OH by 37-45% on gemmini. So the choice needs redoing on
measurement.

Method: two trees per network -- every splittable conv on OC, then every one on
OH -- each run on ONE hart per backend. A single-hart run measures each tile in
isolation, so summing a conv's tiles gives that conv's real cost under that axis
on that backend. The axis with the lower sum wins.

`--pair` decides which backend(s) the choice consults: an rvv pair can only ever
run the tile on rvv, so it must choose on rvv's numbers alone.
"""
import argparse, glob, os, re, sys, collections

OUT = "/scratch/dima/rose-infra/RoSE/experiments/sweep3net"
ARMS = {"serialE": "rvv", "serialP": "gemmini"}


def tiles(tag):
    """{parent_op_name: (n_tiles, total_us)} for a single-hart split run."""
    p = next((x for x in glob.glob(f"{OUT}/res_{tag}/**/uartlog", recursive=True)
              if os.path.exists(x)), None)
    if not p:
        return None
    t = open(p, errors="ignore").read()
    if not re.search(r"xpurt-runner: schedule=" + re.escape(tag) + r"\b", t):
        return None
    agg = collections.defaultdict(lambda: [0, 0])
    for l in t.split("\n"):
        f = l.split(",")
        if not (re.match(r"^\d+,", l) and len(f) == 14):
            continue
        if ".tile_" not in f[5]:
            continue
        par = f[5].split(".tile_")[0]
        agg[par][0] += 1
        agg[par][1] += int(f[-1]) - int(f[-2])
    return {k: tuple(v) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net"); ap.add_argument("--pair", default=None)
    ap.add_argument("--emit", action="store_true",
                    help="print mk_split args for the winning axes")
    a = ap.parse_args()
    arms = {"rvvpair": ["serialE"], "gempair": ["serialP"],
            "hetero": ["serialE", "serialP"]}.get(a.pair, list(ARMS))

    data = {(ax, arm): tiles(f"{a.net}_{arm}_ax{ax}") for ax in ("OC", "OH")
            for arm in arms}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        sys.exit(f"missing profiles: {missing} -- run the axis_search first")

    print(f"\n  {a.net}  best split axis from MEASURED tile costs"
          f"{'  (pair=' + a.pair + ')' if a.pair else ''}")
    print(f"  {'conv':<26}" + "".join(f"{ARMS[m]+' OC':>12}{ARMS[m]+' OH':>12}" for m in arms)
          + f"{'winner':>10}{'gain':>8}")
    print("  " + "-" * (26 + 24 * len(arms) + 18))
    wins = collections.Counter(); args = []
    for conv in sorted({k for v in data.values() for k in v}):
        cells, tot = [], {}
        for m in arms:
            for ax in ("OC", "OH"):
                v = (data[(ax, m)] or {}).get(conv)
                cells.append(f"{v[1]/1000.0:12.3f}" if v else f"{'-':>12}")
                if v:
                    tot[ax] = tot.get(ax, 0) + v[1]
        if len(tot) < 2:
            win, gain = (next(iter(tot)) if tot else "-"), ""
        else:
            win = min(tot, key=tot.get)
            lose = "OH" if win == "OC" else "OC"
            gain = f"{tot[lose]/tot[win]:.2f}x"
        wins[win] += 1
        print(f"  {conv[:25]:<26}" + "".join(cells) + f"{win:>10}{gain:>8}")
    print(f"\n  winners: " + ", ".join(f"{k}={v}" for k, v in sorted(wins.items())))


main()
