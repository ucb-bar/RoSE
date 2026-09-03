#!/usr/bin/env python3
"""Modelled tile cost vs MEASURED tile cost, per split axis.

  cmp_tile_costs.py <net> <exdir> <quant> <pair> [--suffix R|G|H]

Until now every sharded schedule in this campaign was placed using tile costs
DERIVED from the unsplit parent -- `mk_costs_split.py` applies the OC slab
quantum, the OH copy tax, or a flat proportional rule for N/E/C. Those are
assumptions. A single-hart run of the SPLIT tree measures each tile in
isolation, so the two can finally be differenced.

Why it matters per axis:
  * OC   ceil(w/q)/ceil(W/q) with q=32 (rvv) / 16 (gemmini). Right only if the
         inner loop really is blocked on that quantum.
  * OH   window_rows/padded_IH times an empirically fitted copy tax. Two fitted
         constants per backend, from one earlier study.
  * E/C  flat proportional, added 2026-09-02 and never checked against a
         measured tile. The only supporting evidence is indirect (maxpool
         halved 1233 -> 627 us/call).
A large signed error on one axis means that axis's model is wrong; a large
error on all of them means the parent costs are stale.
"""
import argparse, glob, json, os, re, subprocess, sys, collections, statistics

R = "/scratch/dima/rose-infra/RoSE"
S = f"{R}/experiments/shard_dim/scripts"
OUT = f"{R}/experiments/sweep3net"
MB = f"{R}/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"


def measured(tag):
    """{dispatch_id: us} from a single-hart run, or {}."""
    p = next((x for x in glob.glob(f"{OUT}/res_{tag}/**/uartlog", recursive=True)
              if os.path.exists(x)), None)
    if not p:
        return {}
    t = open(p, errors="ignore").read()
    if not re.search(r"xpurt-runner: schedule=" + re.escape(tag) + r"\b", t):
        return {}
    out = {}
    for l in t.split("\n"):
        f = l.split(",")
        if re.match(r"^\d+,", l) and len(f) == 14:
            out[int(f[3])] = int(f[-1]) - int(f[-2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net"); ap.add_argument("exdir"); ap.add_argument("quant")
    ap.add_argument("pair"); ap.add_argument("--suffix", default=None)
    # A RELATIVE error needs a denominator worth dividing by. yolov8_nano's
    # split graph has 233 dispatches, many of them silu tiles measuring 0-1 us,
    # where a 2 us modelling miss reads as +400% and the p90 ran to 16,907%.
    # That is the metric exploding, not the model failing. Tiles below the
    # floor are counted and reported, never silently dropped.
    ap.add_argument("--min-us", type=float, default=5.0)
    a = ap.parse_args()
    sfx = a.suffix or {"rvvpair": "R", "gempair": "G", "hetero": "H"}[a.pair]
    arm = f"shardec{sfx}"
    ex = f"{a.net}_sw_serialE_{arm}"
    graph = f"{MB}/examples/{ex}/{a.quant}/generated/graph.json"
    if not os.path.exists(graph):
        sys.exit(f"no split graph at {graph} -- run the phase 3.5 profile first")

    meas = {"serialE": measured(f"{a.net}_serialE_{arm}"),
            "serialP": measured(f"{a.net}_serialP_{arm}")}
    if not all(meas.values()):
        sys.exit("missing a single-hart profile of the split tree")

    # The MODELLED costs, produced exactly as the scheduler would have.
    tmp_un, tmp_sp = "/tmp/_cmp_unsplit.json", "/tmp/_cmp_split.json"
    subprocess.run([sys.executable, f"{S}/mk_costs.py", tmp_un,
                    f"{OUT}/res_{a.net}_serialE_base",
                    f"{OUT}/res_{a.net}_serialP_base"], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, f"{S}/mk_costs_split.py", tmp_sp,
                    "--graph", graph, "--costs", tmp_un,
                    "--unsplit-graph",
                    f"{MB}/examples/{a.exdir}/{a.quant}/generated/graph.json"],
                   check=True, stdout=subprocess.DEVNULL)
    model = json.load(open(tmp_sp))

    axis_of, kind_of = {}, {}
    for o in json.load(open(graph))["ops"]:
        d = o.get("dispatch_id")
        if d is None:
            continue
        axis_of[d] = (o.get("split_from") or {}).get("axis", "-")
        kind_of[d] = o["op"]

    BE = {"serialE": "rvv", "serialP": "gemmini_q31"}
    rows = collections.defaultdict(list)
    tiny = collections.Counter()
    for arm_key, be in BE.items():
        mm = meas[arm_key]
        for d, us in mm.items():
            mv = (model.get(be) or {}).get(str(d))
            if mv is None or us <= 0:
                continue
            if us < a.min_us:
                tiny[(axis_of.get(d, "-"), be)] += 1
                continue
            rows[(axis_of.get(d, "-"), be)].append((float(mv) - us) / us * 100.0)

    print(f"\n  {a.net} / {a.pair}   modelled tile cost vs MEASURED (single-hart split run)")
    print(f"  {'axis':<6}{'backend':<12}{'tiles':>6}{'median err':>12}{'p90 |err|':>11}   verdict")
    print("  " + "-" * 68)
    for (ax, be), errs in sorted(rows.items()):
        if not errs:
            continue
        med = statistics.median(errs)
        p90 = sorted(abs(e) for e in errs)[max(0, int(0.9 * len(errs)) - 1)]
        verdict = ("model ok" if abs(med) < 10 and p90 < 25 else
                   "MODEL OVER-ESTIMATES" if med > 0 else "MODEL UNDER-ESTIMATES")
        skipped = tiny.get((ax, be), 0)
        note = f"   {verdict}" + (f"  [{skipped} tiles < {a.min_us}us excluded]" if skipped else "")
        print(f"  {ax:<6}{be:<12}{len(errs):>6}{med:>+11.1f}%{p90:>10.1f}%{note}")
    print("\n  err = (modelled - measured)/measured. '-' is an unsplit op and is the"
          "\n  control: it should read ~0, since nothing about it is derived.")


main()
