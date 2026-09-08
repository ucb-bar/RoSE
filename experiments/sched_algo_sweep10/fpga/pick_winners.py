#!/usr/bin/env python3
"""Winner-per-(workload, arm) selection for the FPGA validation set.

Rule (feasible-first, then fastest):
  1. Discard solver entries that errored (the 24 CP-SAT INFEASIBLE returns on
     tight_loop) or whose independent validation audit is non-zero.
  2. If ANY surviving entry has `misses == 0`, the pool is exactly those:
     a schedule that misses a periodic window is not a valid winner.
  3. Otherwise (no solver can satisfy the workload at all) the pool is every
     surviving entry and the cell is marked `relaxed` -- ranked by misses
     first, then makespan. This is what tight_loop needed in the sweep.
  4. Within the pool: min `objective`, tie-broken by wall time then name.

`best-of-fast` is a *selection* over six heuristics, never a distinct schedule,
so it can only tie its own best member -- and the members are in the pool
individually, so it never wins and never needs mapping through `picked`.
"""
import argparse, collections, json


def valid(r):
    if "error" in r:
        return False
    v = r.get("validation") or {}
    return all(not v.get(k, 0) for k in ("prec_viol", "overlap_viol",
                                         "inf_dur_assign", "neg_start",
                                         "before_min_start"))


def pick(entries):
    """entries: {solver: row} -> (solver, row, relaxed)"""
    cands = [(s, r) for s, r in entries.items() if valid(r)]
    if not cands:
        return None, None, False
    zero = [(s, r) for s, r in cands if r["misses"] == 0]
    relaxed = not zero
    pool = zero or cands
    pool.sort(key=lambda x: (x[1]["misses"], x[1]["objective"], x[1]["wall_s"], x[0]))
    return pool[0][0], pool[0][1], relaxed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = collections.defaultdict(dict)
    for r in json.load(open(a.results)):
        cells[(r["arm"], r["workload"])][r["solver"]] = r
    out = []
    for (arm, wl), m in sorted(cells.items()):
        s, r, relaxed = pick(m)
        fam = wl.replace("networks_", "").rsplit("_", 1)[0]
        cfg = wl.rsplit("_", 1)[1]
        g = m["greedy"]
        out.append(dict(arm=arm, workload=wl, family=fam, config=cfg,
                        winner=s, winner_objective=r["objective"],
                        winner_misses=r["misses"], relaxed=relaxed,
                        greedy_objective=g["objective"], greedy_misses=g["misses"],
                        ops=g["ops"], periodic_ops=g["periodic_ops"],
                        impr_pct=round((g["objective"] - r["objective"])
                                       / g["objective"] * 100, 3)))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(out)} cells -> {a.out}")
    print(collections.Counter(x["winner"] for x in out).most_common())


if __name__ == "__main__":
    main()
