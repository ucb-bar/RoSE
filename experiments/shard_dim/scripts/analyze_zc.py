#!/usr/bin/env python3
"""Zero-copy OH tiles vs the gather/scatter ones, same partitions, same hart.

`zcA`/`zcB` carry exactly the partitions `ohA`/`ohB` carried, built from the
same tree on the same bitstream and pinned to the same gemmini hart, so the
only difference between the two runs is whether the OH wrapper copies the band
or the kernel walks the parent in place.
"""
import collections, glob, os, re, sys

R = "/scratch/dima/rose-infra/RoSE/experiments/shard_dim/results/oh"


def tiles(tag):
    c = glob.glob(f"{R}/res_{tag}/**/uartlog", recursive=True) + [f"{R}/res_{tag}/uartlog"]
    p = next((x for x in c if os.path.exists(x)), None)
    if not p:
        return None, None
    t = open(p, errors="ignore").read()
    per = collections.defaultdict(list)
    for l in t.split("\n"):
        f = l.split(",")
        if len(f) > 6 and f[1] == "dronet" and ".tile_" in f[5]:
            per[f[5].split(".tile_")[0]].append((int(f[-1]) - int(f[-2])) / 1.0)
    err = re.search(r"max_abs_err=([0-9.eE+-]+)", t)
    return per, (err.group(1) if err else "?", "PASSED" in t)


print(f"  {'conv':<17}{'gather/scatter':<22}{'zero-copy':<22}{'sum':<9}{'latency (2 harts)'}")
print("  " + "-" * 88)
tot_o = tot_z = 0.0
for old, new in (("ohAP", "zcA"), ("ohBP", "zcB")):
    po, mo = tiles(old)
    pz, mz = tiles(new)
    if po is None or pz is None:
        print(f"  ({old}/{new}: results missing)")
        continue
    for nm in sorted(set(po) & set(pz), key=lambda s: int(s.split(".")[-1])):
        a, b = po[nm], pz[nm]
        sa, sb = sum(a), sum(b)
        tot_o += sa
        tot_z += sb
        print(f"  {nm:<17}{';'.join(f'{x:.0f}' for x in a):<22}"
              f"{';'.join(f'{x:.0f}' for x in b):<22}"
              f"{sa/sb:>6.2f}x  {max(a)/max(b):>6.2f}x")
    print(f"    [{new}] max_abs_err={mz[0]} PASSED={mz[1]}")
print("  " + "-" * 88)
if tot_z:
    print(f"  total OH tile work: {tot_o:.0f} -> {tot_z:.0f} us   {tot_o/tot_z:.2f}x less")
