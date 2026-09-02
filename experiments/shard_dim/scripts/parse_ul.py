#!/usr/bin/env python3
"""Shared uartlog parser for the OH/OC split study (same fields analyze_alignment.py uses)."""
import csv, io, re


def parse_uartlog(path):
    txt = open(path, errors="replace").read()
    m = re.search(r"schedule=(\S+) entries=(\d+)", txt)
    banner = int(m.group(2)) if m else None
    prof = {}
    for blk in re.finditer(
            r"MODELBLASTER_PROFILE_BEGIN \[(\w+)\] ===\n(.*?)=== MODELBLASTER_PROFILE_END",
            txt, re.S):
        for r in csv.DictReader([l for l in blk.group(2).strip().split("\n") if l.strip()]):
            prof[int(r["dispatch_id"])] = {
                "backend": r["backend"], "name": r["name"], "op": r["op"],
                "shape": dict(p.split("=") for p in r["shape"].split(";") if "=" in p),
                "us": int(r["cycles"]) / 1000.0}
    v = re.search(r"MODELBLASTER_VERIFY \[\w+\] === max_abs_err=(\S+)", txt)
    return {"banner": banner, "prof": prof,
            "passed": "*** PASSED ***" in txt,
            "max_abs_err": v.group(1) if v else None,
            "illegal": bool(re.search(r"[Ii]llegal instruction|Unhandled trap|Fatal", txt)),
            "path": path}


def conv_rows(run):
    """{base_op_name: [(tile_idx, us, shape), ...]} sorted by tile."""
    grp = {}
    for did, rec in sorted(run["prof"].items()):
        if rec["op"] != "conv2d_s8":
            continue
        base = rec["name"].split(".tile_")[0]
        t = int(rec["name"].split(".tile_")[1]) if ".tile_" in rec["name"] else 0
        grp.setdefault(base, []).append((t, rec["us"], rec["shape"]))
    for k in grp:
        grp[k].sort()
    return grp
