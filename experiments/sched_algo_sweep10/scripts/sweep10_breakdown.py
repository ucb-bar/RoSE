"""Slices the aggregate hides: per family, per machine pair, per arm, and the
paired head-to-heads between the solvers that are close enough to need one.

The single "mean improvement over greedy" number is an average over eleven
families whose structure differs by an order of magnitude -- depth_chain has no
scheduling freedom at all (every solver lands within 0.03% of greedy) while
depth_contended has 40%+ of spread. Ranking on the pooled mean alone would let
two families decide the answer for all eleven.
"""
import json, sys
import numpy as np

res = sys.argv[1]
d = json.load(open(f"{res}/all_results.json"))
tbl = {(r["arm"], r["workload"], r["solver"]): r for r in d}
allwls = sorted({(a, w) for a, w, s in tbl})
PAIRS = ["gempair", "hetero", "quad", "rvvpair"]
S = ["best-of-fast", "pso", "sa", "cpsat", "cpsat:warm", "cpsat:warmbest",
     "heft_edf", "heft", "decomposed", "greedy_periodic", "greedy_reserved"]


def fam(w):
    b = w[len("networks_"):] if w.startswith("networks_") else w
    for p in PAIRS:
        if b.endswith("_" + p):
            return b[: -len(p) - 1], p
    return b, "?"


def feas(r):
    return r and r.get("objective") is not None and r["misses"] == 0


def impr(ks, s):
    """% makespan improvement over greedy, only where BOTH are deadline-clean."""
    v = []
    for k in ks:
        g, r = tbl.get((k[0], k[1], "greedy")), tbl.get((k[0], k[1], s))
        if feas(g) and feas(r):
            v.append((g["objective"] - r["objective"]) / g["objective"] * 100)
    return v


out = []
def emit(x=""):
    out.append(x)
    print(x)


def table(title, groups):
    emit(f"\n### {title}\n")
    emit("| group | " + " | ".join(s for s in S) + " |")
    emit("|---" * (len(S) + 1) + "|")
    for label, ks in groups:
        cells = []
        for s in S:
            v = impr(ks, s)
            cells.append(f"{np.mean(v):+.2f} (n={len(v)})" if v else "--")
        emit(f"| {label} | " + " | ".join(cells) + " |")


emit("# Breakdown: where each solver is strong and weak")
emit("\nEvery cell is mean % makespan improvement over `greedy`, counted only on "
     "workload-arms where greedy AND the solver both finished with ZERO missed "
     "periodic windows. `n` is how many of the group's workload-arms that leaves; "
     "a small `n` is itself the finding (the solver was infeasible on the rest).")

fams = sorted({fam(w)[0] for a, w in allwls})
table("By family (both arms pooled)",
      [(f, [k for k in allwls if fam(k[1])[0] == f]) for f in fams])
table("By machine pair (tight_loop excluded)",
      [(p, [k for k in allwls if fam(k[1])[1] == p and fam(k[1])[0] != "tight_loop"])
       for p in PAIRS])
table("By arm (tight_loop excluded)",
      [(a, [k for k in allwls if k[0] == a and fam(k[1])[0] != "tight_loop"])
       for a in ("wl_sweep", "wl_sweep_shard")])

wls = [k for k in allwls if fam(k[1])[0] != "tight_loop"]
emit("\n### Paired head-to-head (tight_loop excluded, both feasible)\n")
emit("Positive = row is FASTER than column, as a % of the column's makespan.\n")
tops = ["cpsat:warmbest", "pso", "sa", "cpsat:warm", "best-of-fast", "cpsat",
        "heft_edf", "greedy"]
emit("| | " + " | ".join(tops) + " |")
emit("|---" * (len(tops) + 1) + "|")
for A in tops:
    cells = []
    for B in tops:
        if A == B:
            cells.append("--")
            continue
        v = [(tbl[(k[0], k[1], B)]["objective"] - tbl[(k[0], k[1], A)]["objective"])
             / tbl[(k[0], k[1], B)]["objective"] * 100
             for k in wls
             if feas(tbl.get((k[0], k[1], A))) and feas(tbl.get((k[0], k[1], B)))]
        cells.append(f"{np.mean(v):+.2f} / {np.median(v):+.2f}" if v else "--")
    emit(f"| **{A}** | " + " | ".join(cells) + " |")
emit("\n(mean / median. A median of 0.00 with a non-zero mean means the two "
     "solvers agree on most workloads and differ only in a hard tail.)")

emit("\n### Cheap portfolio composition\n")
from collections import Counter
c = Counter(tbl[(k[0], k[1], "best-of-fast")].get("picked") for k in wls
            if tbl.get((k[0], k[1], "best-of-fast")))
emit("Which of the six sub-second heuristics actually supplied the portfolio's "
     "answer, over the 80 non-tight_loop workload-arms:\n")
for name, n in c.most_common():
    emit(f"- `{name}`: {n}")
walls = [tbl[(k[0], k[1], "best-of-fast")]["wall_s"] for k in allwls
         if tbl.get((k[0], k[1], "best-of-fast"))]
emit(f"\nCost of running all six: mean {np.mean(walls):.3f} s, "
     f"median {np.median(walls):.3f} s, max {np.max(walls):.3f} s.")

open(f"{res}/breakdown.md", "w").write("\n".join(out) + "\n")
print(f"\nwrote {res}/breakdown.md")
