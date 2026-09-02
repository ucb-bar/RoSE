#!/usr/bin/env python3
"""End-to-end makespan + the padding-vs-split attribution.

  analyze_e2e.py

Three measured runs, all on f2_quad_hetero_norose_tacit_q31_60mhz:
  e2eU  unsplit, greedy list schedule over 4 harts   -- the like-for-like baseline
  e2eM  best per-axis assignment, same scheduler     -- the ask
  padE  every conv pre-padded as a SINGLE OH tile, SERIAL on one rvv hart
        -- the control that separates the pre-padding effect from the split

Makespan is the XPURT trace's span (max end - min start) in mtime ticks. One
tick is one microsecond to within 1%: on the serial runs, where the makespan
must equal the summed dispatch time by construction, span/sum measures
0.9895-0.9929 us/tick. The residual is the per-dispatch walker overhead, and
that is reported rather than hidden.
"""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ul import parse_uartlog, conv_rows

RES = "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/results"
MB = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster"
ARM_MAXERR = {"E": "0", "P": "2"}


def trace(path):
    txt = open(path, errors="replace").read()
    m = re.search(r"=== MODELBLASTER_XPURT_TRACE_BEGIN ===\n(.*?)"
                  r"=== MODELBLASTER_XPURT_TRACE_END", txt, re.S)
    if not m:
        return None
    rows = list(csv.DictReader([l for l in m.group(1).strip().split("\n") if l.strip()]))
    s = min(int(r["actual_start_cycles"]) for r in rows)
    e = max(int(r["actual_end_cycles"]) for r in rows)
    return {"n": len(rows), "span": e - s,
            "harts": sorted({r["worker_hart"] for r in rows}),
            "by_hart": {h: sum(int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])
                               for r in rows if r["worker_hart"] == h)
                        for h in sorted({r["worker_hart"] for r in rows})}}


def load(tag):
    d = f"{RES}/oh/res_{tag}"
    uls = glob.glob(os.path.join(d, "**", "uartlog"), recursive=True)
    if not uls:
        return None
    p = max(uls, key=os.path.getsize)
    r = parse_uartlog(p)
    r["trace"] = trace(p)
    return r


def main():
    print("=== GATES ===")
    print(f"{'run':<6} {'entries':>8} {'sched':>6} {'PASS':>5} {'err':>5} {'GATE':>5}")
    runs, sched_n = {}, {}
    for tag, exdir, arm in (("e2eU", "dronet_armB", "P"), ("e2eM", "dronet_e2eM", "P"),
                            ("padE", "dronet_padE", "E")):
        r = load(tag)
        if not r:
            print(f"{tag:<6} {'MISSING':>8}"); continue
        g = json.load(open(f"{MB}/examples/{exdir}/int8/generated/graph.json"))
        n = sum(1 for o in g["ops"] if o.get("dispatch_id") is not None)
        # e2e runs are hetero (both kinds present), so the accuracy gate is the
        # gemmini one -- any gemmini dispatch puts the whole model on the
        # HW-im2col arm's max_abs_err=2 budget.
        ok = (r["banner"] == n and r["passed"] and not r["illegal"]
              and r["max_abs_err"] == ARM_MAXERR[arm])
        runs[tag], sched_n[tag] = r, n
        print(f"{tag:<6} {str(r['banner']):>8} {n:>6} {str(r['passed']):>5} "
              f"{str(r['max_abs_err']):>5} {'OK' if ok else 'FAIL':>5}")

    pred = {t: json.load(open(f"{RES}/oh/costs_{t}.json")) and
            json.load(open(f"/scratch/dima/rose-infra/RoSE/experiments/shard_dim/"
                           f"ohsched/{t}.json"))["metadata"]["makespan"] * 1e6
            for t in ("e2eU", "e2eM") if t in runs}

    print("\n=== END-TO-END MAKESPAN (measured, quad hetero bitstream) ===")
    print(f"{'run':<6} {'disp':>5} {'harts':<12} {'measured_us':>12} {'predicted_us':>13} "
          f"{'gap':>8} {'gap/disp_us':>12}")
    for t in ("e2eU", "e2eM"):
        if t not in runs or not runs[t]["trace"]:
            continue
        tr = runs[t]["trace"]
        p = pred.get(t)
        gap = tr["span"] - p if p else float("nan")
        print(f"{t:<6} {tr['n']:>5} {','.join(tr['harts']):<12} {tr['span']:>12} "
              f"{p:>13.1f} {gap:>8.1f} {gap/tr['n']:>12.2f}")
    if "e2eU" in runs and "e2eM" in runs and runs["e2eU"]["trace"] and runs["e2eM"]["trace"]:
        u, m = runs["e2eU"]["trace"]["span"], runs["e2eM"]["trace"]["span"]
        pu, pm = pred["e2eU"], pred["e2eM"]
        print(f"\n  measured speedup  unsplit -> best-axis : {u}/{m} = {u/m:.3f}x "
              f"({u-m:+.0f} us)")
        print(f"  predicted speedup                        : {pu:.0f}/{pm:.0f} = {pu/pm:.3f}x")
        print(f"  per-hart busy us, e2eU: {runs['e2eU']['trace']['by_hart']}")
        print(f"  per-hart busy us, e2eM: {runs['e2eM']['trace']['by_hart']}")

    # ---- padding vs split attribution -------------------------------------
    print("\n=== PADDING-MATERIALISATION CONTROL (rvv, serial) ===")
    print("padonly = every conv pre-padded, k=1 (no split). Its ratio is the")
    print("pre-padding effect alone; OH k=2 contains padding AND split.\n")
    b0 = conv_rows(parse_uartlog(max(glob.glob(f"{RES}/aln/res_alnB0E/**/uartlog",
                                               recursive=True), key=os.path.getsize)))
    cells = {(r["op"], r["axis"]): float(r["work_ratio"])
             for r in csv.DictReader(open(f"{RES}/ohvsoc_cells.csv"))
             if r["backend"] == "rvv" and r["k"] == "2" and r["work_ratio"]}
    if "padE" in runs:
        pr = conv_rows(runs["padE"])
        print(f"{'conv':<18} {'K':>2} {'S':>2} {'unsplit_us':>11} {'padonly_us':>11} "
              f"{'padonly':>8} {'OH k=2':>8} {'split part':>11}")
        geom = {o["name"]: o["shape"] for o in
                json.load(open(f"{MB}/examples/dronet_armB/int8/generated/graph.json"))["ops"]
                if o.get("op") == "conv2d_s8"}
        for nm in sorted(pr, key=lambda n: int(n.split(".")[-1])):
            if nm not in b0:
                continue
            t_un, t_pad = b0[nm][0][1], pr[nm][0][1]
            rp = t_pad / t_un
            roh = cells.get((nm, "OH"))
            g = geom[nm]
            split = (roh - rp) if roh else float("nan")
            print(f"{nm:<18} {g['KH']:>2} {g['SH']:>2} {t_un:>11.1f} {t_pad:>11.1f} "
                  f"{rp:>8.3f} {(('%.3f'%roh) if roh else '-'):>8} {split:>+11.3f}")


main()
