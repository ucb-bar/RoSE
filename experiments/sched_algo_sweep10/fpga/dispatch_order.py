#!/usr/bin/env python3
"""Emit the ELF tags to dispatch, in priority order, one per line.

  1. the winner ELF for each (workload, arm)
  2. the greedy baseline ELF for each (workload, arm)
  3. everything else

Only tags whose ELF is actually on disk are listed. Interleaved across the four
machine configurations so a partial run still covers the whole spectrum rather
than finishing rvvpair and never reaching quad.
"""
import argparse, collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ELFDIR = os.path.abspath(os.path.join(HERE, "..", "elf"))
CFG = ["rvvpair", "gempair", "hetero", "quad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(HERE, "elf_plan.json"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = {p["tag"]: p for p in json.load(open(a.plan))}
    man = json.load(open(a.manifest))
    win = {r["elf_tag"] for r in man if r["is_recorded_winner"]}
    gre = {r["elf_tag"] for r in man if r["is_greedy_baseline"]}
    tiers = collections.OrderedDict([("winner", []), ("greedy", []), ("other", [])])
    for tag, p in plan.items():
        if not os.path.exists(os.path.join(ELFDIR, tag + ".elf")):
            continue
        t = "winner" if tag in win else ("greedy" if tag in gre else "other")
        tiers[t].append(tag)
    lines = []
    for t, tags in tiers.items():
        by = collections.defaultdict(list)
        for g in tags:
            by[plan[g]["config"]].append(g)
        for v in by.values():
            v.sort()
        i = 0
        while any(by[c] for c in CFG):
            c = CFG[i % 4]; i += 1
            if by[c]:
                lines.append(f"{t}\t{by[c].pop(0)}")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} tags -> {a.out}")
    print(collections.Counter(l.split('\t')[0] for l in lines).most_common())


if __name__ == "__main__":
    main()
