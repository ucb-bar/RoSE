#!/usr/bin/env python3
"""Three-part pass gate for a quad_debug FPGA result.

All three must hold, because any one alone can read as success:

  1. embedded identity -- `xpurt-runner: schedule=<tag>` must equal the
     directory's tag. fq copies results out of the run host's persistent
     sim_slot_*/ directory, so a cell can silently collect the PREVIOUS job's
     uartlog and a stale lane looks exactly like a pass.
  2. no fault -- no `mcause` line, and the run reached the trace dump.
  3. a non-empty 14-field dispatch trace, with every row actually executed
     (worker_hart != -1; the table is pre-initialised to -1/-1 precisely so an
     entry that NO worker ran is distinguishable from one that ran on hart 0).

Usage: verify.py [res_dir ...]     (default: every experiments/quad_debug/res_*)
"""
import sys
import os
import glob

HDR = ("entry_id,network,instance,dispatch_id,op,name,core_kind,hart,"
       "predicted_start_ms,predicted_duration_ms,worker_kind_idx,worker_hart,"
       "actual_start_cycles,actual_end_cycles")
ROOT = "/scratch/dima/rose-infra/RoSE/experiments/quad_debug"


def check(res_dir):
    tag = os.path.basename(res_dir)[len("res_"):]
    uls = glob.glob(os.path.join(res_dir, "**", "uartlog"), recursive=True)
    if not uls:
        return tag, "NO-UARTLOG", {}
    txt = open(uls[0], errors="replace").read()
    lines = txt.splitlines()

    got = ""
    for ln in lines:
        if "xpurt-runner: schedule=" in ln:
            got = ln.split("schedule=")[1].split()[0]
            break
    faults = sum(1 for ln in lines if "mcause" in ln)

    rows, unrun = 0, 0
    seen_hdr = False
    for ln in lines:
        if ln.startswith("entry_id,network,"):
            seen_hdr = True
            continue
        if not seen_hdr:
            continue
        f = ln.split(",")
        if len(f) != 14 or not f[0].isdigit():
            continue
        rows += 1
        # dispatch_id == -1 is a zero-cost IR op (view / chunk2_c1). The
        # walker posts its completion sem and `continue`s WITHOUT writing the
        # trace slot, so worker_hart legitimately stays at the -1 the table was
        # pre-initialised to. Only a real dispatch left unrun is a problem.
        if f[3].strip() != "-1" and f[11].strip() == "-1":
            unrun += 1

    verify = [ln for ln in lines if "MODELBLASTER_VERIFY" in ln]
    info = dict(identity=got, faults=faults, rows=rows, unrun=unrun,
                verify=len(verify), header=seen_hdr)
    ok = (got == tag and faults == 0 and seen_hdr and rows > 0 and unrun == 0)
    return tag, ("PASSED" if ok else "FAILED"), info


def main():
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "res_*")))
    if not dirs:
        print("no result dirs yet")
        return
    worst = 0
    for d in dirs:
        tag, verdict, i = check(d)
        if verdict != "PASSED":
            worst = 1
        print(f"{verdict:8s} {tag}")
        if i:
            print(f"         identity={i['identity']!r} faults={i['faults']} "
                  f"trace_rows={i['rows']} unrun_rows={i['unrun']} "
                  f"verify_lines={i['verify']}")
    sys.exit(worst)


main()
