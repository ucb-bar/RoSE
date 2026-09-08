#!/usr/bin/env python3
"""Winner per cell computed from the LOCALLY emitted schedules, not from
`all_results.json`.

Needed because the workload specs were retuned after the sweep ran (see
README "Fidelity"): on the eight `tight_loop` workload-arms the recorded
ranking is about a different, infeasible problem, and eight further cells drift
by up to 2.5%. Every one of the twelve solvers is emitted and built for every
cell regardless, so this file only changes which ELF carries the "winner"
label -- it never changes what exists.

Same rule as pick_winners.py: discard invalid, prefer zero-miss, then min
makespan.
"""
import argparse, collections, glob, json, os
from pick_winners import pick

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", default=os.path.join(HERE, "schedules"))
    ap.add_argument("--recorded", default=os.path.join(HERE, "winners_recorded.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(a.schedules, "*.meta.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "sched_hash" not in r:
            continue
        cells[(r["arm"], r["workload"])][r["solver"]] = r
    rec = {(w["arm"], w["workload"]): w for w in json.load(open(a.recorded))}
    out, flips = [], 0
    for (arm, wl), m in sorted(cells.items()):
        s, r, relaxed = pick(m)
        g = m.get("greedy")
        fam = wl.replace("networks_", "").rsplit("_", 1)[0]
        row = dict(arm=arm, workload=wl, family=fam, config=wl.rsplit("_", 1)[1],
                   n_solvers_emitted=len(m), winner=s,
                   winner_objective=r["objective"], winner_misses=r["misses"],
                   relaxed=relaxed,
                   greedy_objective=g["objective"] if g else None,
                   greedy_misses=g["misses"] if g else None,
                   impr_pct=(round((g["objective"] - r["objective"]) / g["objective"] * 100, 3)
                             if g and g["objective"] else None),
                   recorded_winner=rec[(arm, wl)]["winner"],
                   sched_hash=r["sched_hash"])
        row["winner_changed"] = row["winner"] != row["recorded_winner"]
        flips += row["winner_changed"]
        out.append(row)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(out)} cells -> {a.out}; winner differs from the recorded one on {flips}")
    print("local winners:", collections.Counter(x["winner"] for x in out).most_common())
    print("cells still with no zero-miss solver:",
          [f"{x['family']}_{x['config']}/{x['arm']}" for x in out if x["relaxed"]])


if __name__ == "__main__":
    main()
