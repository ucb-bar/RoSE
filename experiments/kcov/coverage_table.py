#!/usr/bin/env python3
"""Curated-kernel coverage per (model, backend), from kernel_picks.json.

  coverage_table.py <tag>...        # tags under experiments/kcov/prof/
The picks file is the ONLY reliable evidence that a curated kernel was
selected: a curated file that fails to compile in the verify harness makes the
curator fall back to another algorithm silently.
"""
import collections, json, os, sys
P = "/scratch/dima/rose-infra/RoSE/experiments/kcov/prof"
for tag in sys.argv[1:]:
    f = os.path.join(P, tag + ".picks.json")
    if not os.path.exists(f):
        print("%-22s (no picks file)" % tag); continue
    d = json.load(open(f))
    picks = d["picks"]
    cur = {k: v["algorithm"] for k, v in picks.items() if v["source"] != "reference"}
    ref = sorted(k for k, v in picks.items() if v["source"] == "reference")
    print("%-22s target=%-12s curated %2d/%2d" % (tag, d.get("target", "?"), len(cur), len(picks)))
    for k in sorted(cur):
        print("     + %-26s %s" % (k, cur[k]))
    if ref:
        print("     reference: " + ", ".join(ref))
