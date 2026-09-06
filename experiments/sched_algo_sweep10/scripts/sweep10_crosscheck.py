"""Cross-check against the Sep-3 `wl_sweep_baseline.json` run of the same bench.

Same script, same workloads, same seeds -- so the deterministic solvers must
reproduce EXACTLY. Where they do not, the workload data changed. Where the
stochastic ones differ, that is either the solver code moving or search noise,
and the two have to be told apart rather than averaged over.
"""
import json, sys
import numpy as np

base = json.load(open(sys.argv[1]))
res = json.load(open(sys.argv[2] + "/all_results.json"))
tbl = {(r["arm"], r["workload"], r["solver"]): r for r in res}
DET = {"greedy", "greedy_periodic", "greedy_reserved", "decomposed", "heft", "heft_edf"}
STO = {"pso", "sa", "cpsat", "cpsat:warm"}

lines, rows = [], []
for wl, v in sorted(base.items()):
    if "rows" not in v:
        continue
    for r in v["rows"]:
        m = r.get("method")
        now = tbl.get(("wl_sweep", wl, m))
        if not now:
            continue
        if r.get("objective") is None or now.get("objective") is None:
            rows.append((wl, m, r.get("objective"), now.get("objective"), None))
            continue
        d = (now["objective"] - r["objective"]) / r["objective"] * 100
        rows.append((wl, m, r["objective"], now["objective"], d))

det = [x for x in rows if x[1] in DET and x[4] is not None]
sto = [x for x in rows if x[1] in STO and x[4] is not None]
lines.append(f"compared {len(rows)} (workload, solver) cells present in both runs "
             f"(base arm only; the Sep-3 run had no shard arm)")
# The Sep-3 file stores objectives rounded to 3 decimals, so a bit-identical
# schedule can still differ by ~1e-4 relative. 0.01% is the reproduction floor.
TOL = 0.01
exact = [x for x in det if abs(x[4]) < TOL]
lines.append(f"\nDETERMINISTIC solvers ({len(det)} cells): {len(exact)} reproduce "
             f"to <{TOL}% (the Sep-3 file's 3-decimal rounding floor), "
             f"{len(det)-len(exact)} genuinely differ")
for m in sorted(DET):
    v = [x for x in det if x[1] == m]
    bad = [x for x in v if abs(x[4]) >= TOL]
    lines.append(f"  {m:16s} {len(v)-len(bad):2d}/{len(v):2d} reproduce, "
                 f"{len(bad)} differ" + (f" (worst {max(abs(y[4]) for y in bad):.2f}%)"
                                          if bad else ""))
for x in sorted(det, key=lambda y: -abs(y[4]))[:12]:
    if abs(x[4]) >= TOL:
        lines.append(f"  {x[0]:34s}{x[1]:16s} sep3={x[2]:10.3f} now={x[3]:10.3f} "
                     f"{x[4]:+8.3f}%")
lines.append(f"\nSTOCHASTIC solvers ({len(sto)} cells): mean change "
             f"{np.mean([x[4] for x in sto]):+.3f}%, median "
             f"{np.median([x[4] for x in sto]):+.3f}% (negative = faster now)")
for m in sorted(STO):
    v = [x[4] for x in sto if x[1] == m]
    if v:
        lines.append(f"  {m:12s} n={len(v):3d} mean {np.mean(v):+8.3f}%  "
                     f"median {np.median(v):+8.3f}%  best {min(v):+8.3f}%  "
                     f"worst {max(v):+8.3f}%")
lines.append("\nlargest CP-SAT改善 (most negative = biggest improvement since Sep 3):")
cp = sorted([x for x in sto if x[1].startswith("cpsat")], key=lambda y: y[4])[:10]
for x in cp:
    lines.append(f"  {x[0]:34s}{x[1]:12s} sep3={x[2]:10.3f} now={x[3]:10.3f} {x[4]:+8.3f}%")
txt = "\n".join(lines).replace("改善", " improvement")
open(sys.argv[2] + "/crosscheck.txt", "w").write(txt + "\n")
print(txt)
