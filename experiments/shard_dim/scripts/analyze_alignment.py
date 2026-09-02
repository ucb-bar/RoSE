#!/usr/bin/env python3
"""ALIGNMENT axis of OC operator splitting: measured split cost vs the slab model.

The variable is not the split DEGREE and not EVENNESS but ALIGNMENT: whether
each tile width is a whole multiple of the backend's blocking quantum.

    rvv      quantum V = 32  (TILE_OC clamped to vsetvlmax_e32m4() at VLEN=256)
    gemmini  quantum D = 16  (systolic array DIM)

    slab_ratio(partition) = sum_i ceil(w_i / Q) / ceil(OC / Q)

  analyze_alignment.py <results_root> <out_prefix>

<results_root> holds res_aln<BIN><ARM>/ trees pulled back from fq, one per job.
Baselines come from the aln B0 run of the SAME arm (all convs unsplit).
"""
import csv, glob, json, math, os, re, sys

MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
CFG = "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/configs/alignment_jobs.json"
QUANT = {"rvv": 32, "gemmini_q31": 16}
ARM_BACKEND = {"E": "rvv", "P": "gemmini_q31"}
# accuracy gate: the unsplit dronet arm-B baseline for each arm. A split must
# reproduce it EXACTLY -- an OC split is a disjoint slice, so it is bit-exact
# when the tile weight repack is right.
ARM_MAXERR = {"E": "0", "P": "2"}
# gemmini's floor is a per-tile NCHW->NHWC input transpose, not a slab count.
# a ~= 12.7 ns * IC*IH*IW  (J2's fit; R^2=0.949 vs input size)
GEM_FLOOR_NS_PER_ELEM = 12.7e-3   # us per input element


def parse_uartlog(path):
    txt = open(path, errors="replace").read()
    m = re.search(r"schedule=(\S+) entries=(\d+)", txt)
    banner = int(m.group(2)) if m else None
    prof = {}
    for blk in re.finditer(r"MODELBLASTER_PROFILE_BEGIN \[(\w+)\] ===\n(.*?)=== MODELBLASTER_PROFILE_END",
                           txt, re.S):
        for r in csv.DictReader([l for l in blk.group(2).strip().split("\n") if l.strip()]):
            prof[int(r["dispatch_id"])] = {
                "backend": r["backend"], "name": r["name"], "op": r["op"],
                "shape": dict(p.split("=") for p in r["shape"].split(";") if "=" in p),
                "us": int(r["cycles"]) / 1000.0}
    v = re.search(r"MODELBLASTER_VERIFY \[\w+\] === max_abs_err=(\S+)", txt)
    return {"banner": banner, "prof": prof,
            "passed": "*** PASSED ***" in txt,
            "max_abs_err": v.group(1) if v else None,
            "illegal": bool(re.search(r"[Ii]llegal instruction|Unhandled trap|Fatal", txt))}


def ceil_div(a, b):
    return -(-a // b)


def collect(root):
    cfg = json.load(open(CFG))
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "res_aln*"))):
        tag = os.path.basename(d)[4:]              # aln<BIN><ARM>
        b, arm = tag[3:-1], tag[-1]
        uls = glob.glob(os.path.join(d, "**", "uartlog"), recursive=True)
        if not uls:
            print(f"  !! {tag}: no uartlog"); continue
        ul = max(uls, key=os.path.getsize)
        r = parse_uartlog(ul)
        sched = f"/tmp/aln_sched/aln{b}{arm}.json"
        nsched = len(json.load(open(sched))["dispatches"]) if os.path.exists(sched) else None
        r.update(binary=b, arm=arm, backend=ARM_BACKEND[arm], nsched=nsched, uartlog=ul)
        r["acc_ok"] = (r["max_abs_err"] == ARM_MAXERR[arm])
        r["gate"] = (r["banner"] == nsched and r["passed"] and not r["illegal"]
                     and r["acc_ok"])
        runs[(b, arm)] = r
    return cfg, runs


def group_tiles(run):
    """dispatch profile -> {op_name: [(tile_idx, OC, us), ...]} in tile order."""
    grp = {}
    for did, rec in sorted(run["prof"].items()):
        if rec["op"] not in ("conv2d_s8",):
            continue
        base = rec["name"].split(".tile_")[0]
        t = int(rec["name"].split(".tile_")[1]) if ".tile_" in rec["name"] else 0
        grp.setdefault(base, []).append((t, int(rec["shape"].get("OC", 0)), rec["us"],
                                         rec["shape"]))
    for k in grp:
        grp[k].sort()
    return grp


def main():
    root, outpre = sys.argv[1], sys.argv[2]
    cfg, runs = collect(root)
    labels = cfg["partition_labels"]

    print(f"{'run':<8} {'arm':<4} {'backend':<12} {'banner':>7} {'sched':>6} "
          f"{'PASS':>5} {'err':>5} {'GATE':>5}")
    for (b, arm), r in sorted(runs.items()):
        print(f"{b:<8} {arm:<4} {r['backend']:<12} {str(r['banner']):>7} "
              f"{str(r['nsched']):>6} {str(r['passed']):>5} {str(r['max_abs_err']):>5} "
              f"{'OK' if r['gate'] else 'FAIL':>5}")

    base = {}
    for arm in ("E", "P"):
        if ("B0", arm) not in runs:
            print(f"  !! no B0 baseline for arm {arm}"); continue
        for name, tiles in group_tiles(runs[("B0", arm)]).items():
            base[(name, arm)] = tiles[0][2]

    tile_rows, cell_rows = [], []
    for (b, arm), r in sorted(runs.items()):
        if b == "B0" or not r["gate"]:
            continue
        Q = QUANT[r["backend"]]
        for name, tiles in group_tiles(r).items():
            if len(tiles) < 2:
                continue
            widths = [t[1] for t in tiles]
            us = [t[2] for t in tiles]
            sh = tiles[0][3]
            OC = sum(widths)
            key = str(widths)
            lab = labels.get(key, {})
            aligned = lab.get("rvv_aligned") if r["backend"] == "rvv" else lab.get("gem_aligned")
            slabs = [ceil_div(w, Q) for w in widths]
            pred_slab = sum(slabs) / ceil_div(OC, Q)
            tf = base.get((name, arm))
            wr = sum(us) / tf if tf else float("nan")
            # gemmini transpose-floor alternative: each tile re-runs the whole
            # NCHW->NHWC input transpose, so cost grows with (k-1)*floor, not slabs.
            n_in = int(sh.get("IC", 0)) * int(sh.get("IH", 0)) * int(sh.get("IW", 0))
            floor_us = GEM_FLOOR_NS_PER_ELEM * n_in
            pred_floor = 1.0 + (len(widths) - 1) * floor_us / tf if tf else float("nan")
            for i, (ti, w, t_us, _sh) in enumerate(tiles):
                tile_rows.append(dict(
                    binary=b, arm=arm, backend=r["backend"], op=name, OC=OC,
                    partition=key, tile=i, width=w, us=round(t_us, 2),
                    slabs=ceil_div(w, Q),
                    us_per_slab=round(t_us / ceil_div(w, Q), 2)))
            cell_rows.append(dict(
                binary=b, arm=arm, backend=r["backend"], op=name, OC=OC,
                partition=key, k=len(widths),
                even=lab.get("even"), aligned=aligned,
                tile_us=";".join(f"{x:.1f}" for x in us),
                sum_tiles_us=round(sum(us), 1),
                t_unsplit_us=round(tf, 1) if tf else "",
                work_ratio=round(wr, 4) if tf else "",
                pred_slab=round(pred_slab, 4),
                err_slab=round(wr - pred_slab, 4) if tf else "",
                pred_gem_floor=round(pred_floor, 4) if tf else "",
                err_gem_floor=round(wr - pred_floor, 4) if tf else ""))

    for rows, suffix in ((cell_rows, "cells"), (tile_rows, "tiles")):
        if not rows:
            continue
        p = f"{outpre}_{suffix}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"  -> {p}  ({len(rows)} rows)")

    for backend in ("rvv", "gemmini_q31"):
        sub = [r for r in cell_rows if r["backend"] == backend and r["work_ratio"] != ""]
        if not sub:
            continue
        e = sorted(abs(r["err_slab"]) for r in sub)
        print(f"\n  {backend}: slab-model |err| mean {sum(e)/len(e):.3f}  "
              f"median {e[len(e)//2]:.3f}  max {e[-1]:.3f}  over {len(e)} cells")


main()
