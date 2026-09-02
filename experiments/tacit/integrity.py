#!/usr/bin/env python3
"""TACIT stream integrity check -- the three numbers in F2_TACIT_VERDICT.md 2d.

  1. zero-byte fraction + longest zero run + zero-run length histogram
     (4-byte-quantised runs are the BAR4 lane-fill signature)
  2. framing breaks: places where the packet walk hits a reserved F-header
     (FRes1/FRes2) and has to resync by skipping a byte
  3. summed timestamp deltas, which must equal the traced window's cycle count

Usage: integrity.py <trace.out> [expected_cycles]
"""
import sys
from collections import Counter

FN = ["FTb", "FNt", "FUj", "FIj", "FTrap", "FSync", "FRes1", "FRes2"]
CN = ["CTb", "CNt", "CNa", "CIj"]


def varint(d, i):
    sc = []
    while True:
        if i >= len(d):
            raise IndexError
        b = d[i]; i += 1; sc.append(b)
        if b & 0x80:
            break
    v = 0
    for b in reversed(sc):
        v = (v << 7) | (b & 0x7f)
    return v, i


def walk(d):
    """Walk packets, resyncing on reserved headers. Yields (off,len,kind,fields)
    and counts a framing break each time it has to skip a byte."""
    i = 0
    first = True
    breaks = 0
    while i < len(d):
        st = i
        b = d[i]; i += 1
        c = b & 3
        if c != 2:
            yield (st, i - st, CN[c], {"ts": (b & 0xfc) >> 2}, breaks)
            first = False
            continue
        f = (b >> 2) & 7
        if f in (6, 7):                       # FRes1 / FRes2 -- not emitted by HW
            breaks += 1
            i = st + 1                        # resync: skip one byte
            continue
        fl = {}
        try:
            if f in (0, 1, 3):
                fl["ts"], i = varint(d, i)
            elif f == 2:
                fl["tgt"], i = varint(d, i); fl["ts"], i = varint(d, i)
            elif f == 5:
                if not first:
                    fl["brmode"], i = varint(d, i)
                fl["tgt"], i = varint(d, i); fl["ts"], i = varint(d, i)
            elif f == 4:
                fl["from"], i = varint(d, i); fl["tgt"], i = varint(d, i)
                fl["ts"], i = varint(d, i)
        except IndexError:
            break                             # truncated tail
        yield (st, i - st, FN[f], fl, breaks)
        first = False


def zero_runs(d):
    runs = []
    n = 0
    for b in d:
        if b == 0:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return runs


def main():
    path = sys.argv[1]
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
    d = open(path, "rb").read()
    print(f"file: {path}")
    print(f"bytes: {len(d)}")
    if not d:
        print("EMPTY FILE"); return

    # --- 1. zeros -------------------------------------------------------
    nz = d.count(0)
    runs = zero_runs(d)
    print(f"zero bytes: {nz} ({100.0*nz/len(d):.2f}%)   longest zero run: "
          f"{max(runs) if runs else 0}")
    if runs:
        print("zero-run length histogram (top 8):",
              Counter(runs).most_common(8))

    # --- 2/3. packet walk ------------------------------------------------
    pk = list(walk(d))
    breaks = pk[-1][4] if pk else 0
    kinds = Counter(k for _, _, k, _, _ in pk)
    # A timestamp is a cycle delta between retired control-flow events; anything
    # past ~2^32 is a varint that ran off the end of a truncated packet into the
    # next one, not a real delta. Count those separately -- they are the
    # signature of a corrupt stream, and they poison the sum.
    # The FSync timestamp is NOT a delta within the traced window -- it is the
    # encoder's free-running cycle count at the moment tracing was enabled
    # (i.e. reset -> trace start). Summing it into the window would add the
    # whole boot. It is invisible on a trace enabled at reset (the U250
    # reference's FSync ts is 346) and dominant on one enabled from C
    # (68,162,537 on the F2 PCIM dronet run), which is exactly how it was
    # missed until a trace of the second kind decoded cleanly.
    RUNAWAY = 1 << 32
    ts_all = [fl.get("ts", 0) for _, _, k, fl, _ in pk if k != "FSync"]
    sync_ts = [fl.get("ts", 0) for _, _, k, fl, _ in pk if k == "FSync"]
    runaway = sum(1 for t in ts_all if t >= RUNAWAY)
    tssum = sum(t for t in ts_all if t < RUNAWAY)
    print(f"packets: {len(pk)}   framing breaks: {breaks}"
          f"   ({len(d)/breaks:.0f} bytes/break)" if breaks else
          f"packets: {len(pk)}   framing breaks: 0")
    print("kinds:", kinds.most_common())
    if sync_ts:
        print(f"FSync absolute cycle count at trace start: {sync_ts[0]}")
    print(f"windowed timestamp sum (deltas, FSync excluded): {tssum}")
    print(f"runaway timestamps (>= 2^32, i.e. varint ran off a truncated packet): {runaway}")
    if expected:
        print(f"expected cycles: {expected}   coverage: {100.0*tssum/expected:.1f}%")
    sync = [(off, fl) for off, _, k, fl, _ in pk if k == "FSync"]
    if sync:
        off, fl = sync[0]
        print(f"first FSync at byte {off}: tgt=0x{fl['tgt']:x} -> pc=0x{fl['tgt']<<1:x}")
    else:
        print("no FSync packet found")


if __name__ == "__main__":
    main()
