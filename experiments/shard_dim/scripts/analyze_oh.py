#!/usr/bin/env python3
"""OH (spatial) vs OC (output-channel) operator splitting, measured on F2.

  analyze_oh.py <out_prefix>

Method, unchanged from the alignment study so the numbers compose with it:
every tile of every op is scheduled SERIALLY on one hart, so the metric is work
conservation with zero concurrency tax --

    work_ratio = sum(tile times) / unsplit time      (1.0 = the split is free)

Two things this adds.

DRIFT CORRECTION. Comparing an op across two binaries is comparing two BSS
layouts, worth up to ~9% on gemmini. Every split binary here leaves some convs
UNSPLIT, and those are the same code doing the same work as in the baseline
binary, so their ratio is a direct measurement of that binary's drift. Each
work_ratio is divided by the geometric mean of its binary's unsplit monitors.
Uncorrected values are kept alongside so the correction can be audited.

HALO. For an OC split the duplicated work is COMPUTE: every tile re-reads the
whole input and re-does its share of the spatial sweep at a quantized OC width.
For an OH split it is DATA MOVEMENT only: the tiles compute disjoint output
rows, so the MAC count is exactly conserved, and what the halo duplicates is
input ROWS (the KH-SH rows each pair of neighbours shares). So the two axes are
priced with two different models:

    OC:  work_ratio ~= sum_i ceil(w_i/Q) / ceil(OC/Q)        (slab model)
    OH:  work_ratio ~= 1 + (row_dup - 1) * f_in + copy/T     (halo model)

where row_dup = sum_i window_rows_i / span_rows.
"""
import csv, glob, json, math, os, statistics, sys

MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_ul import parse_uartlog, conv_rows

QUANT = {"rvv": 32, "gemmini_q31": 16}
ARM_BACKEND = {"E": "rvv", "P": "gemmini_q31"}
ARM_MAXERR = {"E": "0", "P": "2"}

#: tag -> (example dir, results root). The aln* rows are the alignment study's
#: already-measured binaries, reused rather than re-run: the OH work is inert
#: for a graph with no OH tile, so those binaries are unchanged by this study.
RUNS = {
    "B0":  ("dronet_alnB0", "aln"),
    "S0":  ("dronet_alnS0", "aln"),
    "L1":  ("dronet_alnL1", "aln"),
    "L2":  ("dronet_alnL2", "aln"),
    "L3":  ("dronet_alnL3", "aln"),
    "ohA": ("dronet_ohA", "oh"),
    "ocA": ("dronet_ocA", "oh"),
    "ohB": ("dronet_ohB", "oh"),
    "ohC": ("dronet_ohC", "oh"),
}
ROOT = {"aln": "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/results/aln",
        "oh":  "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/results/oh"}
PREFIX = {"aln": "res_aln", "oh": "res_"}


def ceil_div(a, b):
    return -(-a // b)


def graph_of(exdir):
    return json.load(open(f"{MB}/examples/{exdir}/int8/generated/graph.json"))


def parent_geom():
    """{conv name: parent shape} from the unsplit tree."""
    return {o["name"]: o["shape"]
            for o in graph_of("dronet_armB")["ops"] if o.get("op") == "conv2d_s8"}


def split_meta(exdir):
    """{base conv name: (axis, [split_from per tile], [tile shapes])}."""
    out = {}
    for o in graph_of(exdir)["ops"]:
        if o.get("op") != "conv2d_s8":
            continue
        sf = o.get("split_from")
        if not sf:
            continue
        base = o["name"].split(".tile_")[0]
        e = out.setdefault(base, (sf["axis"], [], []))
        e[1].append(sf)
        e[2].append(o["shape"])
    for k, (ax, sfs, shs) in out.items():
        order = sorted(range(len(sfs)), key=lambda i: sfs[i]["tile"])
        out[k] = (ax, [sfs[i] for i in order], [shs[i] for i in order])
    return out


def load(tag, arm):
    exdir, root = RUNS[tag]
    d = os.path.join(ROOT[root], f"{PREFIX[root]}{tag}{arm}")
    uls = glob.glob(os.path.join(d, "**", "uartlog"), recursive=True)
    if not uls:
        return None
    r = parse_uartlog(max(uls, key=os.path.getsize))
    nsched = len([o for o in graph_of(exdir)["ops"]
                  if o.get("dispatch_id") is not None])
    r.update(tag=tag, arm=arm, backend=ARM_BACKEND[arm], nsched=nsched,
             exdir=exdir, convs=conv_rows(r))
    r["acc_ok"] = (r["max_abs_err"] == ARM_MAXERR[arm])
    r["gate"] = (r["banner"] == nsched and r["passed"] and not r["illegal"]
                 and r["acc_ok"] and len(r["convs"]) == 10)
    return r


def main():
    outpre = sys.argv[1]
    runs = {}
    for tag in RUNS:
        for arm in ("E", "P"):
            r = load(tag, arm)
            if r:
                runs[(tag, arm)] = r

    print(f"{'run':<6} {'arm':<4} {'backend':<12} {'banner':>7} {'sched':>6} "
          f"{'PASS':>5} {'err':>5} {'GATE':>5}")
    for (t, a), r in sorted(runs.items()):
        print(f"{t:<6} {a:<4} {r['backend']:<12} {str(r['banner']):>7} "
              f"{r['nsched']:>6} {str(r['passed']):>5} {str(r['max_abs_err']):>5} "
              f"{'OK' if r['gate'] else 'FAIL':>5}")

    geom = parent_geom()
    base = {}
    for arm in ("E", "P"):
        r = runs.get(("B0", arm))
        if not r or not r["gate"]:
            print(f"  !! no usable B0 baseline for arm {arm}")
            continue
        for name, tiles in r["convs"].items():
            base[(name, arm)] = tiles[0][1]

    rows, tiles_rows = [], []
    for (tag, arm), r in sorted(runs.items()):
        if tag == "B0" or not r["gate"]:
            continue
        meta = split_meta(r["exdir"])
        # Drift monitor: convs this binary did NOT split.
        mon = [r["convs"][n][0][1] / base[(n, arm)]
               for n in r["convs"]
               if n not in meta and (n, arm) in base and len(r["convs"][n]) == 1]
        drift = math.exp(statistics.fmean(math.log(x) for x in mon)) if mon else 1.0
        for name, tiles in r["convs"].items():
            if name not in meta:
                continue
            axis, sfs, shs = meta[name]
            us = [t[1] for t in tiles]
            if len(us) != len(sfs):
                print(f"  !! {tag}{arm} {name}: {len(us)} profiled tiles vs "
                      f"{len(sfs)} in the graph"); continue
            g = geom[name]
            tb = base.get((name, arm))
            wr_raw = sum(us) / tb if tb else float("nan")
            wr = wr_raw / drift if tb else float("nan")
            Q = QUANT[r["backend"]]
            if axis == "OC":
                widths = [int(s["tile_oc"]) for s in sfs]
                pred = sum(ceil_div(w, Q) for w in widths) / ceil_div(g["OC"], Q)
                model = "slab"
                row_dup = 1.0
                part = str(widths)
            else:
                widths = [int(s["tile_oh"]) for s in sfs]
                span = (g["OH"] - 1) * g["SH"] + g["KH"]
                row_dup = sum(s["window_rows"] for s in sfs) / span
                # Bytes the tile machinery moves that the unsplit conv does not:
                # the gather (pre-padded window) plus the scatter (output band).
                IWP = g["IW"] + 2 * g["PW"]
                gather_B = sum(g["IC"] * s["window_rows"] * IWP for s in sfs)
                scatter_B = sum(g["OC"] * s["tile_oh"] * g["OW"] for s in sfs)
                # The share of the gather that is the HALO proper: rows read
                # more than once because neighbouring tiles overlap. The rest
                # of the gather is the one pass over the input that any
                # implementation of this axis has to make.
                halo_B = g["IC"] * (sum(s["window_rows"] for s in sfs) - span) * IWP
                pred = 1.0          # MAC-conserving; overhead is data movement
                model = "halo"
                part = str(widths)
            over_us = (sum(us) - tb * drift) if tb else float("nan")
            if axis == "OC":
                gather_B = scatter_B = halo_B = 0
            tiles_rows += [dict(run=tag, arm=arm, backend=r["backend"], op=name,
                                axis=axis, tile=i, width=w, us=round(u, 2))
                           for i, (w, u) in enumerate(zip(widths, us))]
            rows.append(dict(
                run=tag, arm=arm, backend=r["backend"], op=name, axis=axis,
                OC=g["OC"], OH=g["OH"], IC=g["IC"], KH=g["KH"], SH=g["SH"],
                k=len(widths), partition=part,
                tile_us=";".join(f"{x:.1f}" for x in us),
                sum_tiles_us=round(sum(us), 1),
                t_unsplit_us=round(tb, 1) if tb else "",
                drift=round(drift, 4), n_monitors=len(mon),
                work_ratio_raw=round(wr_raw, 4) if tb else "",
                work_ratio=round(wr, 4) if tb else "",
                model=model, pred=round(pred, 4),
                row_dup=round(row_dup, 4),
                overhead_us=round(over_us, 2) if tb else "",
                gather_B=gather_B, scatter_B=scatter_B, halo_B=halo_B,
                ns_per_byte=(round(over_us * 1000 / (gather_B + scatter_B), 3)
                             if tb and (gather_B + scatter_B) else ""),
                err_vs_pred=round(wr - pred, 4) if tb else ""))

    for rws, suf in ((rows, "cells"), (tiles_rows, "tiles")):
        p = f"{outpre}_{suf}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0])); w.writeheader(); w.writerows(rws)
        print(f"  -> {p}  ({len(rws)} rows)")

    # Head-to-head table: for each (op, backend) the best available OH and OC.
    print(f"\n{'backend':<12} {'op':<16} {'OC':>4} {'OH':>4} {'K':>2} {'S':>2} "
          f"{'unsplit_us':>11} {'OC_split':>9} {'OH_split':>9} {'winner':>8}  note")
    for be in ("rvv", "gemmini_q31"):
        arm = "E" if be == "rvv" else "P"
        for name in sorted(geom, key=lambda n: int(n.split(".")[-1])):
            g = geom[name]
            oc = [r for r in rows if r["backend"] == be and r["op"] == name
                  and r["axis"] == "OC" and r["k"] == 2 and r["work_ratio"] != ""]
            oh = [r for r in rows if r["backend"] == be and r["op"] == name
                  and r["axis"] == "OH" and r["k"] == 2 and r["work_ratio"] != ""]
            if not oc and not oh:
                continue
            f_oc = min((r["work_ratio"] for r in oc), default=None)
            f_oh = min((r["work_ratio"] for r in oh), default=None)
            win = ("OH" if (f_oh is not None and (f_oc is None or f_oh < f_oc))
                   else "OC")
            tb = base.get((name, arm))
            rd = oh[0]["row_dup"] if oh else float("nan")
            print(f"{be:<12} {name:<16} {g['OC']:>4} {g['OH']:>4} {g['KH']:>2} "
                  f"{g['SH']:>2} {tb if tb else 0:>11.1f} "
                  f"{('%.3f' % f_oc) if f_oc else '     -':>9} "
                  f"{('%.3f' % f_oh) if f_oh else '     -':>9} {win:>8}  "
                  f"row_dup={rd:.3f}")

    print("\n--- OH overhead anatomy (k=2): where the non-free part goes ---")
    print(f"{'backend':<12} {'op':<16} {'over_us':>8} {'gather_kB':>10} "
          f"{'scatter_kB':>11} {'halo_kB':>8} {'halo%gather':>12} {'ns/B':>7}")
    for r in rows:
        if r["axis"] != "OH" or int(r["k"]) != 2 or r["overhead_us"] == "":
            continue
        gB, hB = r["gather_B"], r["halo_B"]
        print(f"{r['backend']:<12} {r['op']:<16} {float(r['overhead_us']):>8.1f} "
              f"{gB/1024:>10.1f} {r['scatter_B']/1024:>11.1f} {hB/1024:>8.1f} "
              f"{100*hB/gB if gB else 0:>11.1f}% {str(r['ns_per_byte']):>7}")


main()
