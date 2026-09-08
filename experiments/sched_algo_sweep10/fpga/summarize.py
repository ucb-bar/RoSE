#!/usr/bin/env python3
"""Coverage / size / dedupe summary over elf_plan.json + manifest.json."""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ELFDIR = os.path.abspath(os.path.join(HERE, "..", "elf"))
plan = json.load(open(os.path.join(HERE, "elf_plan.json")))
man = json.load(open(os.path.join(HERE, "manifest.json")))

built = {p["tag"]: os.path.getsize(os.path.join(ELFDIR, p["tag"] + ".elf"))
         for p in plan if os.path.exists(os.path.join(ELFDIR, p["tag"] + ".elf"))}
print(f"rows in manifest           {len(man)}")
print(f"distinct schedules (plan)  {len(plan)}")
print(f"ELFs on disk               {len(built)}")
print(f"rows with an ELF           {sum(1 for r in man if r['elf_tag'] in built)}")
print()
print("per-config distinct schedules / rows / ELFs built")
cfgs = ["rvvpair", "gempair", "hetero", "quad"]
for cfg in cfgs:
    P = [p for p in plan if p["config"] == cfg]
    M = [r for r in man if r["config"] == cfg]
    B = [p for p in P if p["tag"] in built]
    print(f"  {cfg:9s} distinct {len(P):4d}  rows {len(M):5d}  built {len(B):4d}")
print()
print("per-arm")
for at in ("base", "shard"):
    P = [p for p in plan if p["arm_tag"] == at]
    print(f"  {at:6s} distinct {len(P):4d}  built {sum(1 for p in P if p['tag'] in built):4d}")
print()
print("dedupe: distinct schedules per cell (of 12 rows)")
per = collections.Counter()
cell = collections.defaultdict(set)
for p in plan:
    cell[(p["arm"], p["workload"])].add(p["sched_hash"])
for k, v in cell.items():
    per[len(v)] += 1
for k in sorted(per):
    print(f"  {k:2d} distinct: {per[k]} cells")
tot = sum(len(v) for v in cell.values())
print(f"  total {tot} distinct over {len(cell)} cells "
      f"(mean {tot/len(cell):.2f} of 12)")
print()
print("solver-group composition (how often each solver shares its ELF)")
sh = collections.Counter()
for r in man:
    sh[len(r["shared_with"])] += 1
for k in sorted(sh):
    print(f"  shares with {k}: {sh[k]} rows")
print()
if built:
    sz = sorted(built.values())
    import statistics
    print(f"ELF sizes: median {statistics.median(sz)/1e6:.2f} MB  "
          f"mean {statistics.mean(sz)/1e6:.2f} MB  max {max(sz)/1e6:.2f} MB")
    big = sorted(((v, k) for k, v in built.items() if v > 20e6), reverse=True)
    print(f"  >20 MB: {len(big)} ELFs")
    for v, k in big[:5]:
        print(f"    {v/1e6:6.2f} MB  {k}")
print()
print("sweep10 reproduction (emit-time objective vs the recorded row)")
rep = collections.Counter(r["reproduces"] for r in man if r["sweep10_objective"] is not None)
print(f"  exact {rep[True]}   differs {rep[False]}   "
      f"recorded-error rows {sum(1 for r in man if r['sweep10_error'])}")
