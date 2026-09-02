#!/usr/bin/env python3
"""Compare NHWC-island arms against the NCHW baseline, per op family.

  analyze.py <baseline_tag> <arm_tag> [<arm_tag> ...]

Joins on MODULE NAME, never on dispatch_id: assign_layouts inserts two relayout
dispatches and therefore renumbers every op after the first, exactly as a split
does. The relayouts have no counterpart in the baseline and are reported as an
added cost, which is the honest way round -- they are the price of the island,
not free.
"""
import re, sys, csv, os

R = "/scratch/dima/rose-infra/RoSE"
OUT = f"{R}/experiments/nhwc_island/results"


def parse(tag):
    p = None
    for root, _, files in os.walk(f"{OUT}/res_{tag}"):
        for f in files:
            if f == "uartlog":
                p = os.path.join(root, f)
    if not p:
        raise SystemExit(f"no uartlog for {tag}")
    txt = open(p, errors="replace").read()
    m = re.search(r"schedule=(\S+) entries=(\d+)", txt)
    sched, entries = (m.group(1), int(m.group(2))) if m else (None, None)
    prof = {}
    for blk in re.finditer(
            r"MODELBLASTER_PROFILE_BEGIN \[(\w+)\] ===\n(.*?)=== MODELBLASTER_PROFILE_END",
            txt, re.S):
        for r in csv.DictReader([l for l in blk.group(2).strip().split("\n") if l.strip()]):
            prof[r["name"]] = {"op": r["op"], "shape": r["shape"],
                               "cycles": int(r["cycles"])}
    v = re.search(r"max_abs_err=(\S+)", txt)
    return {"tag": tag, "sched": sched, "entries": entries, "prof": prof,
            "max_abs_err": v.group(1) if v else None, "path": p,
            "wall": int(m2.group(1)) if (m2 := re.search(
                r"MODELBLASTER_WALL_CYCLES \[\w+\] === (\d+)", txt)) else None}


def fam(rec):
    if rec["op"] in ("nchw_to_nhwc_s8", "nhwc_to_nchw_s8"):
        return "relayout"
    return rec["op"]


def main():
    base = parse(sys.argv[1])
    print(f"baseline {base['tag']}: sched={base['sched']} entries={base['entries']} "
          f"max_abs_err={base['max_abs_err']} wall={base['wall']}")
    b_fam, b_tot = {}, 0
    for n, r in base["prof"].items():
        b_fam[fam(r)] = b_fam.get(fam(r), 0) + r["cycles"]
        b_tot += r["cycles"]
    for tag in sys.argv[2:]:
        a = parse(tag)
        print(f"\n=== {tag} vs {base['tag']} ===")
        print(f"  sched={a['sched']} entries={a['entries']} "
              f"max_abs_err={a['max_abs_err']} "
              f"(baseline {base['max_abs_err']}) "
              f"{'BIT-IDENTICAL' if a['max_abs_err']==base['max_abs_err'] else '*** DIFFERS ***'}")
        a_fam, a_tot = {}, 0
        for n, r in a["prof"].items():
            a_fam[fam(r)] = a_fam.get(fam(r), 0) + r["cycles"]
            a_tot += r["cycles"]
        print(f"  {'family':<18}{'baseline':>12}{'arm':>12}{'speedup':>10}")
        for k in sorted(set(b_fam) | set(a_fam)):
            bb, aa = b_fam.get(k, 0), a_fam.get(k, 0)
            sp = f"{bb/aa:.2f}x" if aa else ("-" if not bb else "inf")
            print(f"  {k:<18}{bb:>12}{aa:>12}{sp:>10}")
        print(f"  {'TOTAL':<18}{b_tot:>12}{a_tot:>12}{b_tot/a_tot:>9.2f}x")
        cb = b_fam.get("conv2d_s8", 0)
        ca = a_fam.get("conv2d_s8", 0) + a_fam.get("relayout", 0)
        print(f"  conv-section (conv2d_s8 baseline vs conv2d_s8+relayout arm): "
              f"{cb} -> {ca} = {cb/ca:.2f}x")
        print(f"  per-conv:")
        for n in sorted(base["prof"]):
            if base["prof"][n]["op"] != "conv2d_s8":
                continue
            bb = base["prof"][n]["cycles"]; aa = a["prof"].get(n, {}).get("cycles")
            print(f"    {n:<18}{bb:>10}{(aa if aa else 0):>10}"
                  f"{(bb/aa if aa else 0):>9.2f}x  {base['prof'][n]['shape']}")
        for n, r in sorted(a["prof"].items()):
            if fam(r) == "relayout":
                print(f"    +{n:<17}{'':>10}{r['cycles']:>10}          {r['shape']}")


main()
