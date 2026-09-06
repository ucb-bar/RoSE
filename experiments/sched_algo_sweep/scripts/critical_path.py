#!/usr/bin/env python3
"""Per-workload DAG critical path, and what it predicts about solver choice.

The longest path through a workload's dispatch DAG is a lower bound on the
makespan that NO scheduler can beat.  When greedy already sits close to it,
there is nothing for a better algorithm to win, and the sweep's "every solver
ties at 0.00%" families are exactly those.  This computes the bound from the
greedy schedules (the DAG and the per-dispatch durations are the same for every
solver -- only the placement differs) and joins it to the measured wins.

  critical_path.py --work <sweep work dir> --comparison <comparison.json> \
                   --out <results dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def critical_path(sched_path: str) -> tuple[float, float]:
    with open(sched_path) as fh:
        disp = json.load(fh)["dispatches"]
    dur = {k: v["duration"] for k, v in disp.items()}
    deps = {k: v["dependencies"] for k, v in disp.items()}
    memo: dict[str, float] = {}

    def lp(k: str) -> float:
        if k in memo:
            return memo[k]
        memo[k] = dur[k] + max([lp(x) for x in deps[k] if x in dur], default=0.0)
        return memo[k]

    cp = max(lp(k) for k in dur)
    ms = max(v["start_time"] + v["duration"] for v in disp.values())
    return cp, ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    with open(args.comparison) as fh:
        comp = json.load(fh)
    idx = {(e["arm"], "networks_%s_%s" % (e["family"], e["pair"])): e for e in comp}

    rows = []
    for d in sorted(glob.glob(os.path.join(args.work, "*__greedy"))):
        cid = os.path.basename(d)[: -len("__greedy")]
        arm, base = cid.split("__", 1)
        p = os.path.join(d, "schedules", f"scheduled_{base}_greedy_profiled.json")
        if not os.path.exists(p):
            continue
        e = idx.get((arm, base))
        if not e:
            continue
        cp, ms = critical_path(p)
        rows.append({
            "arm": arm, "family": e["family"], "pair": e["pair"],
            "num_operations": e["num_operations"],
            "greedy_makespan_ms": ms,
            "dag_critical_path_ms": cp,
            "critical_path_fraction": cp / ms if ms else None,
            "best_available_win_pct": e["best_improvement_pct"],
            "best_solver": e["best_solver"],
        })

    hi = [r for r in rows if (r["critical_path_fraction"] or 0) > args.threshold]
    lo = [r for r in rows if (r["critical_path_fraction"] or 0) <= args.threshold]
    summary = {
        "threshold": args.threshold,
        "n": len(rows),
        "dominated_by_one_chain": {
            "n": len(hi),
            "mean_best_win_pct": st.mean([r["best_available_win_pct"] for r in hi]) if hi else None,
            "max_best_win_pct": max([r["best_available_win_pct"] for r in hi], default=None),
        },
        "slack_available": {
            "n": len(lo),
            "mean_best_win_pct": st.mean([r["best_available_win_pct"] for r in lo]) if lo else None,
            "max_best_win_pct": max([r["best_available_win_pct"] for r in lo], default=None),
        },
        "reading": ("The critical-path fraction is a NECESSARY condition, not a "
                    "sufficient one: above the threshold no algorithm can help, "
                    "below it one might. Use it as a cheap pre-filter before "
                    "spending a solve on an alternative scheduler."),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "critical_path.json"), "w") as fh:
        json.dump({"summary": summary, "rows": sorted(
            rows, key=lambda r: -(r["critical_path_fraction"] or 0))}, fh, indent=1)

    print(f"n={len(rows)} workload cells")
    print(f"  critical path > {args.threshold:.0%} of greedy's makespan "
          f"(one chain dominates): n={len(hi)} "
          f"mean best win={summary['dominated_by_one_chain']['mean_best_win_pct']:.2f}% "
          f"max={summary['dominated_by_one_chain']['max_best_win_pct']:.2f}%")
    print(f"  critical path <= {args.threshold:.0%} (slack exists): n={len(lo)} "
          f"mean best win={summary['slack_available']['mean_best_win_pct']:.2f}% "
          f"max={summary['slack_available']['max_best_win_pct']:.2f}%")
    print(f"wrote {os.path.join(args.out, 'critical_path.json')}")


if __name__ == "__main__":
    main()
