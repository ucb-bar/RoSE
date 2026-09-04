#!/usr/bin/env python3
"""Parse an ExecuTorch runner uartlog into per-model results, and join them
against the ModelBlaster single-RVV-hart baselines from the same bitstream.

ET side  (experiments/executorch/res/<tag>/**/uartlog):
    MB_MODEL_BEGIN=<m> / MB_MODEL_END=<m>      per-model bracket
    MB_ET_HART=<n>, MB_ET_THREADPOOL from=..to=..   placement gates
    MB_INPUT_BAKED=<m> n=..                     real input actually fed
    EXECUTORCH_EXECUTE_CYCLES[<m>][<i>]=<cyc>   rdcycle around method->execute()
    Output[<i>] numel=<n> checksum=<x>
    `>>, <op>, <cycles>`                        per-op (MB_XNN_PROFILE=ON only)

MB side  (experiments/sweep3net/res_<model>_serialE_base/uartlog):
    the MODELBLASTER_PROFILE csv -- summed, this is the rdcycle cost of the
    whole network on ONE rvv hart, the like-for-like counterpart of ET's
    execute() bracket.
"""
import argparse, json, os, re, sys

MB_RES = "/scratch/dima/rose-infra/RoSE/experiments/sweep3net"


def find_uartlog(d):
    for root, _, files in os.walk(d):
        if "uartlog" in files:
            yield os.path.join(root, "uartlog")


def parse_et(path):
    txt = open(path, errors="replace").read()
    out = {"uartlog": path, "models": {}}
    m = re.search(r"MB_ET_HART=(\d+)", txt)
    out["hart"] = int(m.group(1)) if m else None
    m = re.search(r"MB_ET_THREADPOOL from=(\d+) to=(\d+) ok=(\d+)", txt)
    out["threadpool"] = ([int(m.group(1)), int(m.group(2)), int(m.group(3))]
                         if m else None)
    out["fault"] = bool(re.search(r"mcause|ZEPHYR FATAL ERROR", txt))
    out["passed"] = "*** PASSED ***" in txt

    for tag in re.findall(r"MB_MODEL_BEGIN=(\S+)", txt):
        seg = re.search(rf"MB_MODEL_BEGIN={re.escape(tag)}\b(.*?)"
                        rf"(?:MB_MODEL_END={re.escape(tag)}|\Z)", txt, re.S)
        body = seg.group(1) if seg else ""
        cyc = {int(i): int(c) for i, c in
               re.findall(rf"EXECUTORCH_EXECUTE_CYCLES\[{re.escape(tag)}\]\[(\d+)\]=(\d+)", txt)}
        warm = [cyc[k] for k in sorted(cyc) if k > 0]
        r = {"cold_cyc": cyc.get(0),
             "warm_cyc": min(warm) if warm else None,
             "warm_all": [cyc[k] for k in sorted(cyc) if k > 0],
             "iters": len(cyc),
             "ended": bool(re.search(rf"MB_MODEL_END={re.escape(tag)}", txt)),
             "input_baked": bool(re.search(
                 rf"MB_INPUT_BAKED={re.escape(tag)}\b", txt)),
             "checksums": [(int(n), float(c)) for n, c in
                           re.findall(r"numel=(\d+) checksum=(-?[0-9.eE+na]+)", body)
                           if c not in ("nan", "-nan")],
             }
        ops = re.findall(r">>,\s*([^,]+),\s*(\d+)", body)
        if ops:
            agg = {}
            for name, c in ops:
                agg.setdefault(name.strip(), []).append(int(c))
            # per-op lines repeat once per invoke; take the LAST (warm) sample
            r["ops"] = {k: v[-1] for k, v in agg.items()}
            r["op_sum"] = sum(r["ops"].values())
        out["models"][tag] = r
    return out


def mb_baseline(model):
    p = f"{MB_RES}/res_{model}_serialE_base/uartlog"
    if not os.path.exists(p):
        return None
    tot, n, backends, inprof = 0, 0, set(), False
    verify = None
    for line in open(p, errors="replace"):
        if "MODELBLASTER_PROFILE_BEGIN" in line:
            inprof = True; continue
        if "MODELBLASTER_PROFILE_END" in line:
            inprof = False; continue
        if inprof:
            f = line.strip().split(",")
            if len(f) >= 6 and f[-1].isdigit():
                tot += int(f[-1]); n += 1; backends.add(f[0])
        if "MODELBLASTER_VERIFY" in line:
            verify = line.strip()
    return {"uartlog": p, "profile_sum_cycles": tot, "ops": n,
            "backends": sorted(backends), "verify": verify}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resdirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    allres = {}
    for d in a.resdirs:
        logs = list(find_uartlog(d)) if os.path.isdir(d) else [d]
        if not logs:
            print(f"{d}: NO UARTLOG"); continue
        # the fq results dir has both a top-level copy and the per-run tree;
        # take the longest (most complete) one
        path = max(logs, key=lambda p: os.path.getsize(p))
        r = parse_et(path)
        allres[os.path.basename(d.rstrip("/"))] = r
        print(f"=== {d}")
        print(f"  hart={r['hart']} threadpool={r['threadpool']} "
              f"passed={r['passed']} fault={r['fault']}")
        for tag, m in r["models"].items():
            base = tag.rsplit("_", 1)[0]
            mb = mb_baseline(base)
            ratio = (f"{m['warm_cyc']/mb['profile_sum_cycles']:.2f}x"
                     if m.get("warm_cyc") and mb and mb["profile_sum_cycles"] else "-")
            print(f"  {tag:22s} cold={m['cold_cyc']} warm={m['warm_cyc']} "
                  f"iters={m['iters']} ended={m['ended']} baked_in={m['input_baked']}")
            print(f"  {'':22s} checksums={m['checksums']}")
            print(f"  {'':22s} MB_rvv_1hart={mb['profile_sum_cycles'] if mb else None} "
                  f"(ops={mb['ops'] if mb else '-'})  ET/MB={ratio}")
            if "ops" in m:
                top = sorted(m["ops"].items(), key=lambda kv: -kv[1])[:12]
                print(f"  {'':22s} op_sum={m['op_sum']}  top={top}")
    if a.json:
        json.dump(allres, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
