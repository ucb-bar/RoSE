#!/usr/bin/env python3
"""Surgically retune only the periods that are broken or pointless.

POLICY. Keep as many workloads byte-identical as possible. A period is touched
only when it is one of:

  INFEASIBLE  one instance cannot fit its own window (duty = cost/period > 1).
              The schedule is not hard, it is impossible: every heuristic
              silently returns a miss-laden schedule and only the MILP says so.
  TRIVIAL     duty < 0.10, i.e. the window is more than 10x the work. Nothing
              contends, so the cell cannot distinguish one scheduler from
              another and costs FPGA time to learn nothing.

Everything in between is left exactly as it is -- that band is the experiment.

Retuned periods target duty ~= 0.5 (window = 2x the measured single-hart cost
on this pair's slowest backend), which is feasible with real headroom while
still applying pressure.

NOT touched: networks with no period (aperiodic/one-shot). scale_ladder is
entirely aperiodic by design -- it studies duration spread, not deadlines --
so it scores zero utilisation and must not be "fixed".

NOT touched either: any network in a dependency CHAIN (a workload with
`edges`). Chain members deliberately share one period so workload_factory
pairs instance i to instance i; retuning one member alone would give the two
ends different rates and silently drop them onto its different-rates
"newest closed producer" rule instead -- changing what the depth families
measure. A chain's duty is the WHOLE chain's cost over the shared period, and
by that measure depth_chain is 0.67 on gempair, i.e. challenging, not
trivial. Judging the fast member alone is what made it look pointless.

  tune_windows.py [--apply]      default is a dry run
"""
import argparse, glob, json, os

W = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/data/toplevel"
PAIRS = {"rvvpair": (0, 2), "gempair": (2, 0), "hetero": (1, 1), "quad": (2, 2)}
TARGET_DUTY = 0.5
INFEASIBLE, TRIVIAL = 1.0, 0.10

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mk_workloads import MEASURED


def tune(path, apply_):
    b = os.path.basename(path)[9:-5]
    pair = b.rsplit("_", 1)[1]
    p_ct, e_ct = PAIRS[pair]
    d = json.load(open(path))
    # every network named by an edge is a chain member -- off limits
    chained = set()
    for edge in d.get("edges", []):
        chained.add(edge.get("from"))
        chained.add(edge.get("to"))
    changes = []
    for name, e in d["networks"].items():
        if name in chained:
            continue
        per = e.get("period")
        if per is None or name not in MEASURED:
            continue
        g, v = MEASURED[name]
        cands = ([g] if p_ct else []) + ([v] if e_ct else [])
        slow = max(cands)
        duty = slow / per
        why = ("INFEASIBLE" if duty > INFEASIBLE else
               "trivial" if duty < TRIVIAL else None)
        if why is None:
            continue
        new = round(slow / TARGET_DUTY, 3)
        changes.append((name, per, new, duty, why))
        if apply_:
            e["period"] = new
            e["window_duration"] = new
    if changes and apply_:
        json.dump(d, open(path, "w"), indent=1)
    return b, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    touched = n = 0
    for sub in ("wl_sweep", "wl_sweep_shard"):
        for f in sorted(glob.glob(f"{W}/{sub}/*.json")):
            b, ch = tune(f, a.apply)
            n += 1
            if not ch:
                continue
            touched += 1
            for name, old, new, duty, why in ch:
                print(f"  {sub:<15}{b:<30}{name:<17}{why:<11}"
                      f"duty {duty:5.2f}  {old} -> {new}")
    print(f"\n  {touched} of {n} workload files {'changed' if a.apply else 'would change'}"
          f"; {n - touched} left byte-identical")


main()
