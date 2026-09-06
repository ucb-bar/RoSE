#!/usr/bin/env python3
"""Four-part pass gate for an FPGA cell result.

`*** PASSED ***` on its own means almost nothing here, and the wl_sweep
campaign proved it three separate ways:

  1. IDENTITY. fq copies results out of the run host's persistent sim_slot_*/,
     so a cell can silently collect the PREVIOUS job's uartlog. Five quad
     result directories read as complete while holding zero trace rows.
  2. FAULT. A run can trap and still leave a plausible-looking log.
  3. TRACE. The dispatch table is pre-initialised to -1/-1 so an entry NO
     worker ran is distinguishable from one that ran on hart 0; an empty or
     all -1 table is not a result.
  4. NUMERICS -- the one that was missing. `*** PASSED ***` does NOT gate on
     MODELBLASTER_VERIFY, so a run whose registers were corrupted mid-kernel
     reports success with wrong numbers. Every int8 drift > 2 in the whole
     campaign was on the quad pair (6 occurrences, 0 elsewhere) and every one
     of them was reported green.

Drift is REPORTED, and only fails the cell under --strict-drift. That is
deliberate: ~34 cells across all four machine pairs carry small fp32
mlp_control drift of ~4-9 that predates this work and is a different
phenomenon from the quad int8 corruption. Hard-failing on a blanket threshold
would bury the real signal in known noise. Use --strict-drift when you want
numerics to gate, e.g. when re-verifying a fix.

  verify_cell.py [--atol 2] [--strict-drift] [--quiet] <res_dir> [...]

Exit status is 0 only if every directory passes.
"""
import argparse
import glob
import os
import re
import sys

TRACE_FIELDS = 14


def _uartlog(res_dir):
    """Largest uartlog under res_dir; fq nests them a few levels deep."""
    uls = [u for u in glob.glob(os.path.join(res_dir, "**", "uartlog"),
                                recursive=True)
           + [os.path.join(res_dir, "uartlog")] if os.path.exists(u)]
    return max(uls, key=os.path.getsize) if uls else None


def check(res_dir, atol=2.0, strict_drift=False):
    """-> (status, detail). status 'OK' only when every gate holds."""
    tag = os.path.basename(res_dir.rstrip("/"))
    tag = tag[4:] if tag.startswith("res_") else tag
    d = {"tag": tag, "rows": 0, "drift": {}, "faults": 0}

    ul = _uartlog(res_dir)
    if ul is None:
        return "NO-UARTLOG", d
    txt = open(ul, errors="replace").read()
    lines = txt.splitlines()

    m = re.search(r"xpurt-runner: schedule=(\S+)", txt)
    d["embedded"] = m.group(1) if m else ""
    if not m:
        return "NO-IDENTITY", d
    if m.group(1) != tag:
        return "STALE-TAG", d          # this is someone else's run

    d["faults"] = sum(1 for ln in lines if "mcause" in ln)
    if d["faults"]:
        fm = re.search(r"mcause:\s*(\S+)[^\n]*\n\s*mtval:\s*(\S+)", txt)
        if fm:
            d["mcause"], d["mtval"] = fm.group(1), fm.group(2)
        return "FAULT", d

    rows = [ln.split(",") for ln in lines
            if re.match(r"^\d+,", ln) and len(ln.split(",")) == TRACE_FIELDS]
    rows = [r for r in rows if int(r[-3]) >= 0]      # worker_hart != -1
    d["rows"] = len(rows)
    if not rows:
        return "NO-TRACE", d

    if "*** PASSED ***" not in txt:
        return "NOT-PASSED", d

    for ln in lines:
        if "MODELBLASTER_VERIFY" in ln:
            net = re.search(r"\[([^\]]+)\]", ln)
            err = re.search(r"max_abs_err=(\S+)", ln)
            if net and err:
                try:
                    d["drift"][net.group(1)] = float(err.group(1))
                except ValueError:
                    pass
    d["max_drift"] = max(d["drift"].values(), default=0.0)
    if d["max_drift"] > atol:
        return ("DRIFT" if strict_drift else "OK-DRIFT"), d
    return "OK", d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--atol", type=float, default=2.0,
                    help="max_abs_err tolerated before a cell is flagged")
    ap.add_argument("--strict-drift", action="store_true",
                    help="treat drift over --atol as a FAILURE, not a warning")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    bad = 0
    for res in a.dirs:
        st, d = check(res, a.atol, a.strict_drift)
        ok = st in ("OK", "OK-DRIFT")
        bad += not ok
        if a.quiet and ok:
            continue
        extra = ""
        if st in ("OK", "OK-DRIFT", "NO-TRACE"):
            extra = f"rows={d['rows']}"
            if d["drift"]:
                worst = max(d["drift"], key=lambda k: d["drift"][k])
                extra += f" max_drift={d['max_drift']:g} ({worst})"
        elif st == "FAULT":
            extra = f"mcause={d.get('mcause','?')} mtval={d.get('mtval','?')}"
        elif st == "STALE-TAG":
            extra = f"log says schedule={d['embedded']}"
        print(f"  {d['tag']:<46} {st:<12} {extra}")
    if not a.quiet:
        print(f"\n  {len(a.dirs) - bad}/{len(a.dirs)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
