#!/usr/bin/env python3
"""{backend: {dispatch_id: us}} from the two serial runs of ONE split tree.

  mk_costs.py OUT.json <E_results_dir> <P_results_dir>

The serial runs put every dispatch of a graph on one hart, so between the E
(rvv) and P (gemmini_q31) runs of the SAME tree every dispatch has a measured
cost on both backends -- which is exactly the table a placement needs, taken
from this hardware rather than from a model.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ul import parse_uartlog

out, dirs = sys.argv[1], sys.argv[2:]
costs = {}
for d, be in zip(dirs, ("rvv", "gemmini_q31")):
    uls = glob.glob(os.path.join(d, "**", "uartlog"), recursive=True)
    r = parse_uartlog(max(uls, key=os.path.getsize))
    costs[be] = {str(k): round(v["us"], 3) for k, v in r["prof"].items()}
    print(f"  {be}: {len(costs[be])} dispatches from {d}")
json.dump(costs, open(out, "w"), indent=1)
print(f"[mk_costs] -> {out}")
