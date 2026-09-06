#!/usr/bin/env python3
"""Supplementary arm: is the MILP's poor showing just its 120 s budget?

The wl_sweep workloads all carry `scheduler.time_limit: 120`, and that limit
binds MOSEK's optimizer only -- not cvxpy's model build, which is where most of
the wall clock goes.  This re-runs the cells whose model actually builds with a
600 s optimizer limit instead, so the makespan difference can be attributed to
the search budget rather than to the formulation.

  milp600_supplement.py --cells <main cells dir> --cells600 <600s cells dir> \
                        --out <results dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def load(d: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(f) as fh:
            c = json.load(fh)
        out[c["cell_id"]] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cells600", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    main_cells = load(args.cells)
    c600 = load(args.cells600)
    rows = []
    for cid, c in sorted(c600.items()):
        base = cid.rsplit("__", 1)[0]
        g = main_cells.get(base + "__greedy")
        m120 = main_cells.get(cid)
        row = {
            "arm": c.get("arm"), "family": c.get("family"), "pair": c.get("pair"),
            "ops": (g or {}).get("num_operations"),
            "greedy_ms": (g or {}).get("makespan_pred_ms"),
            "milp120_ms": (m120 or {}).get("makespan_pred_ms"),
            "milp120_gap": (m120 or {}).get("mip_rel_gap"),
            "milp120_wall_s": (m120 or {}).get("wall_s"),
            "milp120_status": (m120 or {}).get("solver_status")
                              or (m120 or {}).get("failure_reason"),
            "milp600_ms": c.get("makespan_pred_ms"),
            "milp600_gap": c.get("mip_rel_gap"),
            "milp600_wall_s": c.get("wall_s"),
            "milp600_build_s": c.get("build_s"),
            "milp600_solve_s": c.get("solve_s"),
            "milp600_status": c.get("solver_status") or c.get("failure_reason"),
        }
        for k, ms in (("milp120_vs_greedy_pct", row["milp120_ms"]),
                      ("milp600_vs_greedy_pct", row["milp600_ms"])):
            row[k] = (100.0 * (row["greedy_ms"] - ms) / row["greedy_ms"]
                      if (ms and row["greedy_ms"]) else None)
        rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(os.path.join(args.out, "milp600_supplement.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "milp600_supplement.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    print("== MILP optimizer budget: the workloads' 120 s vs 600 s ==")
    print("  {:6s}{:17s}{:9s}{:>6s}{:>11s}{:>11s}{:>9s}{:>11s}{:>9s}".format(
        "arm", "family", "pair", "ops", "greedy_ms", "milp120", "d%",
        "milp600", "d%"))
    for r in rows:
        def f(v, w=11, p=2):
            return ("{:>%d}" % w).format("n/a" if v is None else f"{v:.{p}f}")
        print("  {:6s}{:17s}{:9s}{:>6}{}{}{}{}{}".format(
            r["arm"] or "", r["family"] or "", r["pair"] or "", r["ops"] or "",
            f(r["greedy_ms"]), f(r["milp120_ms"]), f(r["milp120_vs_greedy_pct"], 9),
            f(r["milp600_ms"]), f(r["milp600_vs_greedy_pct"], 9)))
    print(f"\nwrote {os.path.join(args.out, 'milp600_supplement.csv')}")


if __name__ == "__main__":
    main()
