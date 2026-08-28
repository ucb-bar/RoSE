#!/usr/bin/env python3
"""Turn a tacit-decoder `trace.vbb.csv` into a per-basic-block cycle attribution
mapped back to functions and source lines.

`--to-vbb` emits one row per *virtual basic block* -- a (start_addr, end_addr)
pair delimited by two consecutive control-flow events -- with:

    count   how many times that block was executed
    mean    mean cycles from entering the block to leaving it
    netvar  sum(interval) - min(interval)*count, i.e. the cycles ABOVE the
            block's own fastest observed execution.  This is the stall/variance
            term: a block with high netvar spent time waiting (cache, vector
            hazard), not computing.

total = count * mean is the block's whole-run cycle contribution.

Usage:
  vbb_analyze.py trace.vbb.csv --elf zephyr.elf --addr2line <prefix>addr2line \
      [--nm <prefix>nm] [--top 40] [--filter-func conv2d]
"""
import argparse, bisect, csv, collections, subprocess, sys


def load_symbols(elf, nm):
    """-> (sorted start addrs, [(start, size, name)])"""
    out = subprocess.run([nm, "-S", "--defined-only", elf],
                         capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4:
            addr, size, typ, name = p
            if typ.lower() in ("t", "w"):          # text / weak text
                syms.append((int(addr, 16), int(size, 16), name))
    syms.sort()
    return [s[0] for s in syms], syms


def sym_for(addr, starts, syms):
    i = bisect.bisect_right(starts, addr) - 1
    if i < 0:
        return "?", 0
    start, size, name = syms[i]
    if size and addr >= start + size:
        return "?", 0
    return name, addr - start


def addr2line_batch(elf, tool, addrs):
    if not addrs:
        return {}
    inp = "\n".join(f"{a:#x}" for a in addrs)
    out = subprocess.run([tool, "-e", elf, "-f", "-i", "-C"],
                         input=inp, capture_output=True, text=True).stdout
    # -f -i gives variable line counts per address; re-query one at a time only
    # for the ones we cannot align. Simpler + reliable: query without -i.
    out = subprocess.run([tool, "-e", elf, "-C"],
                         input=inp, capture_output=True, text=True).stdout.splitlines()
    return {a: (out[i] if i < len(out) else "??") for i, a in enumerate(addrs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vbb")
    ap.add_argument("--elf", required=True)
    ap.add_argument("--addr2line", required=True)
    ap.add_argument("--nm", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--filter-func", default=None,
                    help="only blocks whose symbol contains this substring")
    a = ap.parse_args()
    nm = a.nm or a.addr2line.replace("addr2line", "nm")

    starts, syms = load_symbols(a.elf, nm)

    rows = []
    with open(a.vbb) as f:
        for r in csv.DictReader(f):
            bb = r["bb"].strip()
            lo, hi = bb.split("-")
            lo, hi = int(lo, 16), int(hi, 16)
            cnt = int(r["count"])
            mean = float(r["mean"])
            netvar = float(r["netvar"])
            rows.append((lo, hi, cnt, mean, netvar, cnt * mean))

    grand = sum(r[5] for r in rows)
    print(f"blocks={len(rows)}  attributed cycles={grand:,.0f}\n")

    # ---- per function ------------------------------------------------------
    per_fn = collections.defaultdict(lambda: [0.0, 0, 0.0])
    for lo, hi, cnt, mean, netvar, tot in rows:
        fn, _ = sym_for(lo, starts, syms)
        per_fn[fn][0] += tot
        per_fn[fn][1] += cnt
        per_fn[fn][2] += netvar
    print("=== per function ===")
    print(f"{'cycles':>14} {'%':>6} {'execs':>10} {'stall(netvar)':>14}  function")
    for fn, (tot, cnt, nv) in sorted(per_fn.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"{tot:14,.0f} {100*tot/grand:6.2f} {cnt:10,} {nv:14,.0f}  {fn}")

    # ---- per block ---------------------------------------------------------
    sel = []
    for lo, hi, cnt, mean, netvar, tot in rows:
        fn, off = sym_for(lo, starts, syms)
        if a.filter_func and a.filter_func not in fn:
            continue
        sel.append((tot, lo, hi, cnt, mean, netvar, fn, off))
    sel.sort(reverse=True)
    want = sorted({s[1] for s in sel[:a.top]} | {s[2] for s in sel[:a.top]})
    lines = addr2line_batch(a.elf, a.addr2line, want)
    print(f"\n=== top {a.top} basic blocks"
          f"{' in ' + a.filter_func if a.filter_func else ''} ===")
    print(f"{'cycles':>13} {'%':>6} {'execs':>9} {'mean':>8} {'stall':>12}  block  -> source")
    for tot, lo, hi, cnt, mean, netvar, fn, off in sel[:a.top]:
        print(f"{tot:13,.0f} {100*tot/grand:6.2f} {cnt:9,} {mean:8.2f} {netvar:12,.0f}  "
              f"{fn}+{off:#x} [{lo:#x}-{hi:#x}]")
        print(f"{'':>52}  {lines.get(lo,'??')}")


if __name__ == "__main__":
    main()
