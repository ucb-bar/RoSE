#!/usr/bin/env python3
"""
Turn the per-cell JSONs from sweep_driver.py into the combined table, the
solver-vs-greedy comparison, and the two plots.

greedy is the incumbent: every wl_sweep cell measured on F2 today was
scheduled with it.  So every comparison here is "solver X against greedy on
the same workload, same arm, same machine pair".

Usage:
  analyse_sweep.py --cells <dir of per-cell json> --out <results dir> \
                   --plots <plot dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st

SOLVERS = ["greedy", "greedy_periodic", "decomposed", "milp"]
FIELDS = [
    "cell_id", "family", "pair", "arm", "solver", "ok", "solver_status",
    "timed_out", "returncode",
    "makespan_pred_ms", "num_operations", "num_periodic_operations",
    "total_busy_ms", "wall_s", "build_s", "solve_s", "mosek_time_s",
    "mip_rel_gap", "mip_best_obj", "mip_best_bound",
    "num_period_misses", "worst_late_ms",
    "util_CPU_P", "util_CPU_E", "busy_CPU_P_ms", "busy_CPU_E_ms",
    "n_CPU_P", "n_CPU_E", "hw_CPU_P", "hw_CPU_E", "failure_reason", "error",
]


def flatten(r: dict) -> dict:
    out = {k: r.get(k) for k in FIELDS}
    ubk = r.get("util_by_kind") or {}
    bbk = r.get("busy_by_kind_ms") or {}
    nbk = r.get("n_machines_by_kind") or {}
    phw = r.get("profile_hw") or {}
    out["util_CPU_P"] = ubk.get("CPU_P")
    out["util_CPU_E"] = ubk.get("CPU_E")
    out["busy_CPU_P_ms"] = bbk.get("CPU_P")
    out["busy_CPU_E_ms"] = bbk.get("CPU_E")
    out["n_CPU_P"] = nbk.get("CPU_P", 0)
    out["n_CPU_E"] = nbk.get("CPU_E", 0)
    out["hw_CPU_P"] = phw.get("CPU_P")
    out["hw_CPU_E"] = phw.get("CPU_E")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    cells = []
    for f in sorted(glob.glob(os.path.join(args.cells, "*.json"))):
        with open(f) as fh:
            cells.append(json.load(fh))

    # ---------- combined table ----------
    rows = [flatten(r) for r in cells]
    rows.sort(key=lambda r: (r["arm"] or "", r["family"] or "", r["pair"] or "",
                             SOLVERS.index(r["solver"]) if r["solver"] in SOLVERS else 9))
    csv_path = os.path.join(args.out, "combined_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out, "all_cells.json"), "w") as fh:
        json.dump(sorted(cells, key=lambda r: r["cell_id"]), fh, indent=1)

    # ---------- per-workload comparison ----------
    by_wl: dict[tuple, dict] = {}
    for r in cells:
        key = (r["arm"], r.get("family"), r.get("pair"))
        by_wl.setdefault(key, {})[r["solver"]] = r

    comp = []
    for key, d in sorted(by_wl.items()):
        arm, fam, pair = key
        g = d.get("greedy")
        if not (g and g.get("ok")):
            continue
        base_ms = g["makespan_pred_ms"]
        entry = {
            "arm": arm, "family": fam, "pair": pair,
            "num_operations": g["num_operations"],
            "greedy_makespan_ms": base_ms,
            "greedy_wall_s": g["wall_s"],
        }
        best_ms, best_solver = base_ms, "greedy"
        for s in SOLVERS:
            r = d.get(s)
            if not (r and r.get("ok")):
                entry[f"{s}_makespan_ms"] = None
                entry[f"{s}_improvement_pct"] = None
                entry[f"{s}_wall_s"] = (r or {}).get("wall_s")
                entry[f"{s}_status"] = (r or {}).get("solver_status") or "FAILED"
                continue
            ms = r["makespan_pred_ms"]
            entry[f"{s}_makespan_ms"] = ms
            # A solver is only comparable to greedy on makespan if it scheduled
            # the SAME operations. trim_periodic_after_nonperiodic_makespan cuts
            # periodic instances whose window opens after the non-periodic work
            # ends, so a solver that finishes the non-periodic part later keeps
            # MORE periodic instances and is then measured over a longer span --
            # a penalty for scheduling more work, not a worse schedule.
            entry[f"{s}_ops"] = r["num_operations"]
            entry[f"{s}_comparable"] = (r["num_operations"] == g["num_operations"])
            entry[f"{s}_improvement_pct"] = (
                100.0 * (base_ms - ms) / base_ms
                if (base_ms and entry[f"{s}_comparable"]) else None)
            entry[f"{s}_improvement_pct_raw"] = (
                100.0 * (base_ms - ms) / base_ms if base_ms else None)
            entry[f"{s}_wall_s"] = r["wall_s"]
            entry[f"{s}_status"] = r.get("solver_status")
            entry[f"{s}_period_misses"] = r.get("num_period_misses")
            if ms < best_ms - 1e-9 and entry[f"{s}_comparable"]:
                best_ms, best_solver = ms, s
        entry["best_solver"] = best_solver
        entry["best_makespan_ms"] = best_ms
        entry["best_improvement_pct"] = 100.0 * (base_ms - best_ms) / base_ms if base_ms else 0.0
        # op-count consistency: a solver that scheduled a different number of
        # operations is not comparable on makespan.
        opsets = {s: d[s]["num_operations"] for s in SOLVERS
                  if d.get(s) and d[s].get("ok")}
        entry["ops_consistent"] = len(set(opsets.values())) <= 1
        entry["ops_by_solver"] = opsets
        comp.append(entry)

    with open(os.path.join(args.out, "comparison.json"), "w") as fh:
        json.dump(comp, fh, indent=1)

    ccols = (["arm", "family", "pair", "num_operations", "ops_consistent"]
             + [f"{s}_makespan_ms" for s in SOLVERS]
             + [f"{s}_improvement_pct" for s in SOLVERS]
             + [f"{s}_wall_s" for s in SOLVERS]
             + [f"{s}_status" for s in SOLVERS]
             + [f"{s}_period_misses" for s in SOLVERS]
             + [f"{s}_comparable" for s in SOLVERS]
             + [f"{s}_ops" for s in SOLVERS]
             + ["best_solver", "best_improvement_pct"])
    with open(os.path.join(args.out, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ccols, extrasaction="ignore")
        w.writeheader()
        w.writerows(comp)

    # ---------- aggregate ----------
    agg = {}
    for s in SOLVERS:
        imps = [e[f"{s}_improvement_pct"] for e in comp
                if e.get(f"{s}_improvement_pct") is not None]
        walls = [e[f"{s}_wall_s"] for e in comp if e.get(f"{s}_wall_s") is not None]
        wins = sum(1 for e in comp if e["best_solver"] == s)
        strict_better = sum(1 for e in comp
                            if (e.get(f"{s}_improvement_pct") or 0) > 1e-9)
        strict_worse = sum(1 for e in comp
                           if (e.get(f"{s}_improvement_pct") is not None
                               and e[f"{s}_improvement_pct"] < -1e-9))
        ties = sum(1 for e in comp
                   if e.get(f"{s}_improvement_pct") is not None
                   and abs(e[f"{s}_improvement_pct"]) <= 1e-9)
        agg[s] = {
            "n_cells": len(imps),
            "n_failed": sum(1 for e in comp if e.get(f"{s}_makespan_ms") is None),
            "n_excluded_op_count_mismatch": sum(
                1 for e in comp if e.get(f"{s}_makespan_ms") is not None
                and not e.get(f"{s}_comparable", True)),
            "mean_improvement_pct": st.mean(imps) if imps else None,
            "median_improvement_pct": st.median(imps) if imps else None,
            "min_improvement_pct": min(imps) if imps else None,
            "max_improvement_pct": max(imps) if imps else None,
            "outright_wins": wins,
            "strictly_better_than_greedy": strict_better,
            "strictly_worse_than_greedy": strict_worse,
            "tied_with_greedy": ties,
            "mean_wall_s": st.mean(walls) if walls else None,
            "median_wall_s": st.median(walls) if walls else None,
            "max_wall_s": max(walls) if walls else None,
            "total_wall_s": sum(walls) if walls else None,
        }
    # per family
    fam_agg = {}
    fams = sorted({e["family"] for e in comp})
    for fam in fams:
        sub = [e for e in comp if e["family"] == fam]
        fam_agg[fam] = {}
        for s in SOLVERS:
            imps = [e[f"{s}_improvement_pct"] for e in sub
                    if e.get(f"{s}_improvement_pct") is not None]
            fam_agg[fam][s] = {
                "n": len(imps),
                "mean_improvement_pct": st.mean(imps) if imps else None,
                "wins": sum(1 for e in sub if e["best_solver"] == s),
            }
    with open(os.path.join(args.out, "aggregate.json"), "w") as fh:
        json.dump({"overall": agg, "by_family": fam_agg,
                   "n_workload_cells": len(comp)}, fh, indent=1)

    # ---------- plots ----------
    # Palette: slots 1-3 of the reference categorical theme (blue / orange /
    # aqua).  Those three are the documented all-pairs-validated subset, which
    # is what a scatter needs; greedy is the baseline (0% by construction) and
    # is drawn as a reference rule rather than a fourth hue, so no fourth slot
    # is needed.  Aqua sits below 3:1 on a light surface, so the relief rule
    # applies -- the CSV table beside these plots is that relief.
    SER = {"greedy_periodic": "#2a78d6", "decomposed": "#eb6834", "milp": "#1baf7a"}
    CMP = [s for s in SOLVERS if s != "greedy"]
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        def _style(ax):
            ax.set_facecolor("#fcfcfb")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
                ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
            ax.grid(axis="y", color=MUTED, lw=0.5, alpha=0.5)
            ax.set_axisbelow(True)

        # ---- Figure 1: improvement over greedy, per cell, grouped by family
        fig, ax = plt.subplots(figsize=(13, 6.2), facecolor="#fcfcfb")
        _style(ax)
        # keep the offset well inside one category slot: at +/-0.22 a point sits
        # a fifth of the way to the next family and reads as belonging to it
        jit = {"greedy_periodic": -0.12, "decomposed": 0.0, "milp": 0.12}
        for si, s in enumerate(CMP):
            xs, ys = [], []
            for fi, fam in enumerate(fams):
                for e in comp:
                    if e["family"] != fam:
                        continue
                    v = e.get(f"{s}_improvement_pct")
                    if v is None:
                        continue
                    xs.append(fi + jit[s])
                    ys.append(v)
            ax.scatter(xs, ys, s=34, color=SER[s], label=s, alpha=0.85,
                       edgecolors="#fcfcfb", linewidths=0.8, zorder=3)
        ax.axhline(0, color=INK, lw=1.4, zorder=2)
        ax.annotate("greedy (incumbent)", xy=(len(fams) - 0.45, 0), xytext=(0, 5),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=INK2)
        ax.set_xticks(range(len(fams)))
        # horizontal, centred: rotated+right-aligned labels put the tick at the
        # label's right edge, which reads as if every point were shifted one
        # family to the right
        ax.set_xticklabels(fams, rotation=0, ha="center", color=INK2, fontsize=8)
        ax.set_xlim(-0.6, len(fams) - 0.4)
        ax.set_ylabel("predicted makespan vs greedy (%)   positive = shorter",
                      color=INK2, fontsize=9)
        ax.set_title("Where the scheduling algorithm actually matters\n"
                     "one dot per workload cell (machine pair x arm); "
                     "predicted makespan from the xpu-rt cost model",
                     color=INK, fontsize=12, loc="left", pad=12)
        leg = ax.legend(frameon=False, fontsize=9, loc="upper left",
                        labelcolor=INK2, ncol=3)
        # direct-label the extremes so identity never rests on colour alone
        for s in CMP:
            best = max((e for e in comp if e.get(f"{s}_improvement_pct") is not None),
                       key=lambda e: e[f"{s}_improvement_pct"], default=None)
            worst = min((e for e in comp if e.get(f"{s}_improvement_pct") is not None),
                        key=lambda e: e[f"{s}_improvement_pct"], default=None)
            for e in (best, worst):
                if e is None or abs(e[f"{s}_improvement_pct"]) < 5:
                    continue
                v = e[f"{s}_improvement_pct"]
                ax.annotate(f"{s} {e['family']}/{e['pair']} {v:+.0f}%",
                            xy=(fams.index(e["family"]) + jit[s], v),
                            xytext=(7, 5 if v > 0 else -9),
                            textcoords="offset points",
                            fontsize=7, color=INK2, va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(args.plots, "makespan_by_solver.png"), dpi=150,
                    facecolor="#fcfcfb")
        plt.close(fig)

        # ---- Figure 2: what it costs vs what it buys
        fig, ax = plt.subplots(figsize=(11, 6), facecolor="#fcfcfb")
        _style(ax)
        ax.grid(axis="x", color=MUTED, lw=0.5, alpha=0.5)
        for s in CMP:
            x = [e[f"{s}_wall_s"] for e in comp
                 if e.get(f"{s}_wall_s") and e.get(f"{s}_improvement_pct") is not None]
            y = [e[f"{s}_improvement_pct"] for e in comp
                 if e.get(f"{s}_wall_s") and e.get(f"{s}_improvement_pct") is not None]
            ax.scatter(x, y, s=40, color=SER[s], label=f"{s}  (n={len(x)})",
                       alpha=0.8, edgecolors="#fcfcfb", linewidths=0.8, zorder=3)
            if x:
                ax.annotate(s, xy=(st.median(x), max(y) if y else 0),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)
        ax.axhline(0, color=INK, lw=1.4, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("wall-clock time to produce the schedule (s, log)",
                      color=INK2, fontsize=9)
        ax.set_ylabel("predicted makespan vs greedy (%)   positive = shorter",
                      color=INK2, fontsize=9)
        ax.set_title("Cost against benefit\n"
                     "greedy itself takes 2-25 s; the zero line is greedy",
                     color=INK, fontsize=12, loc="left", pad=12)
        ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.plots, "solvetime_vs_quality.png"), dpi=150,
                    facecolor="#fcfcfb")
        plt.close(fig)
        # ---- Figure 3: where the MILP stops being buildable
        # Status encoding (reserved status palette, never the series hues), and
        # every class carries a marker shape as well as a colour so the meaning
        # never rests on colour alone.
        # Reserved status palette + one neutral: "proved infeasible" is not a
        # solver failure, it is the solver telling you the workload's own
        # periodic windows cannot be met, so it does not wear a failure colour.
        STAT = {
            "solved to proven optimal":  ("#0ca30c", "o"),
            "feasible, gap left open":   ("#fab219", "s"),
            "out of memory":             ("#d03b3b", "X"),
            "hit the 1200 s wall clock": ("#ec835a", "P"),
            "proved INFEASIBLE (workload, not solver)": ("#52514e", "D"),
        }
        mcells = [c for c in cells if c.get("solver") == "milp"]
        if mcells:
            def _klass(c):
                if c.get("ok"):
                    g = c.get("mip_rel_gap")
                    return ("solved to proven optimal"
                            if (g is not None and g < 1e-3)
                            else "feasible, gap left open")
                fr = c.get("failure_reason") or ""
                if fr == "milp_proved_infeasible":
                    return "proved INFEASIBLE (workload, not solver)"
                if fr.startswith("out_of_memory"):
                    return "out of memory"
                if fr == "wall_clock_timeout" or c.get("timed_out"):
                    return "hit the 1200 s wall clock"
                return "out of memory"

            fig, ax = plt.subplots(figsize=(11, 6), facecolor="#fcfcfb")
            _style(ax)
            ax.grid(axis="x", color=MUTED, lw=0.5, alpha=0.5)
            opsmap = {(e["arm"], e["family"], e["pair"]): e["num_operations"]
                      for e in comp}
            for k, (col, mk) in STAT.items():
                xs = [opsmap.get((c["arm"], c.get("family"), c.get("pair")))
                      for c in mcells if _klass(c) == k]
                ys = [c.get("milp_n_constraints") for c in mcells if _klass(c) == k]
                pts = [(x, y) for x, y in zip(xs, ys) if x and y]
                if not pts:
                    continue
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=52,
                           color=col, marker=mk, label=f"{k}  (n={len(pts)})",
                           alpha=0.9, edgecolors="#fcfcfb", linewidths=0.8, zorder=3)
            ax.set_xlabel("operations in the workload", color=INK2, fontsize=9)
            ax.set_ylabel("cvxpy scalar constraints in the MILP model",
                          color=INK2, fontsize=9)
            ax.set_yscale("log")
            # The wall sits on the y axis, not the x: draw it where the data
            # puts it -- between the largest model that solved and the smallest
            # that ran out of room.
            solved_max = max([c.get("milp_n_constraints") for c in mcells
                              if c.get("ok") and c.get("milp_n_constraints")],
                             default=None)
            failed_min = min([c.get("milp_n_constraints") for c in mcells
                              if not c.get("ok") and c.get("milp_n_constraints")
                              and c.get("failure_reason", "").startswith("out_of_memory")],
                             default=None)
            if solved_max and failed_min and failed_min > solved_max:
                ax.axhspan(solved_max, failed_min, color=MUTED, alpha=0.35, zorder=1)
                ax.annotate(f"budget wall: {solved_max/1000:.0f}k-{failed_min/1000:.0f}k "
                            f"constraints at 24 GB / 1200 s",
                            xy=(ax.get_xlim()[0], (solved_max * failed_min) ** 0.5),
                            xytext=(8, 0), textcoords="offset points",
                            fontsize=8, color=INK2, va="center")
            ax.set_title("What decides whether the MILP is usable is the pruned "
                         "constraint count, not the workload size\n"
                         "depth_chain has 270 operations and 2k constraints (its "
                         "periodic windows do not overlap) and solves to proven "
                         "optimal in 5 s;\ncontrol_mix has 295 and 669k, and runs "
                         "out of address space inside cvxpy",
                         color=INK, fontsize=11, loc="left", pad=12)
            # headroom so the legend sits over empty plot, not over marks
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 6)
            ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plots, "milp_scaling_wall.png"), dpi=150,
                        facecolor="#fcfcfb")
            plt.close(fig)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"plotting failed: {type(e).__name__}: {e}")

    # ---------- console summary ----------
    print(f"cells: {len(cells)}  workload-cells with greedy baseline: {len(comp)}")
    print("\n== overall vs greedy ==")
    for s in SOLVERS:
        a = agg[s]
        print(f"  {s:16s} n={a['n_cells']:3d} failed={a['n_failed']:3d} "
              f"mean={_f(a['mean_improvement_pct'])}% med={_f(a['median_improvement_pct'])}% "
              f"range=[{_f(a['min_improvement_pct'])},{_f(a['max_improvement_pct'])}]% "
              f"wins={a['outright_wins']:3d} better={a['strictly_better_than_greedy']:3d} "
              f"worse={a['strictly_worse_than_greedy']:3d} "
              f"wall_med={_f(a['median_wall_s'])}s max={_f(a['max_wall_s'])}s")
    print("\n== by family (mean improvement vs greedy, %) ==")
    hdr = "  {:20s}".format("family") + "".join(f"{s:>18s}" for s in SOLVERS)
    print(hdr)
    for fam in fams:
        line = f"  {fam:20s}"
        for s in SOLVERS:
            v = fam_agg[fam][s]["mean_improvement_pct"]
            line += f"{_f(v):>18s}"
        print(line)
    # failure breakdown -- an experiment that reports only what succeeded
    # cannot be reasoned about
    import collections
    fails = collections.Counter()
    for c in cells:
        if not c.get("ok"):
            fails[(c.get("solver"), c.get("failure_reason") or c.get("error", "?"))] += 1
    if fails:
        print("\n== failures ==")
        for (s_, why), n in sorted(fails.items(), key=lambda kv: -kv[1]):
            print(f"  {s_:16s} {why:34s} {n:4d}")
    # MILP cost detail
    mi = [c for c in cells if c.get("solver") == "milp" and c.get("ok")]
    if mi:
        print("\n== MILP cells that produced a schedule ==")
        print("  {:6s}{:17s}{:9s}{:>6s}{:>9s}{:>9s}{:>9s}{:>11s}  {}".format(
            "arm", "family", "pair", "ops", "build_s", "solve_s", "wall_s",
            "gap", "status"))
        for c in sorted(mi, key=lambda c: c.get("num_operations", 0)):
            print("  {:6s}{:17s}{:9s}{:>6d}{:>9s}{:>9s}{:>9.1f}{:>11s}  {}".format(
                c["arm"], c["family"], c["pair"], c["num_operations"],
                _f(c.get("build_s")), _f(c.get("solve_s")), c["wall_s"],
                _f(c.get("mip_rel_gap")), c.get("solver_status")))
    print(f"\nwrote {csv_path}")


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    main()
