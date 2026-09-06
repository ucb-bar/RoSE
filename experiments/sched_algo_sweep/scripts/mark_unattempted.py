#!/usr/bin/env python3
"""Write an explicit per-cell record for grid cells that were never attempted.

The MILP arm was given a bounded budget (see the driver's --mem-limit-gb /
--hard-timeout / --deadline-s).  Cells past that budget have no cell JSON at
all, and a missing row reads as "we forgot" rather than "we stopped on
purpose".  This fills them in with `not_attempted_budget` so the combined table
covers the whole 44 x 2 x 4 grid.

  mark_unattempted.py --cells <dir> --wl-dir-base <..> --wl-dir-shard <..>
"""
from __future__ import annotations

import argparse
import json
import os

SOLVERS = ["greedy", "greedy_periodic", "decomposed", "milp"]
PAIRS = ("gempair", "hetero", "quad", "rvvpair")


def split_basename(base: str) -> tuple[str, str]:
    stem = base[len("networks_"):] if base.startswith("networks_") else base
    for p in PAIRS:
        if stem.endswith("_" + p):
            return stem[: -(len(p) + 1)], p
    return stem, "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--wl-dir-base", required=True)
    ap.add_argument("--wl-dir-shard", required=True)
    ap.add_argument("--reason", default="not_attempted_budget")
    args = ap.parse_args()

    arms = {"base": args.wl_dir_base, "shard": args.wl_dir_shard}
    made = 0
    for arm, d in arms.items():
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            base = fn[:-5]
            fam, pair = split_basename(base)
            for s in SOLVERS:
                cid = f"{arm}__{base}__{s}"
                p = os.path.join(args.cells, cid + ".json")
                if os.path.exists(p):
                    continue
                with open(p, "w") as fh:
                    json.dump({
                        "cell_id": cid, "arm": arm, "basename": base,
                        "family": fam, "pair": pair, "solver": s,
                        "ok": False,
                        "error": args.reason,
                        "failure_reason": args.reason,
                    }, fh, indent=1)
                made += 1
    print(f"wrote {made} not-attempted cell records")


if __name__ == "__main__":
    main()
