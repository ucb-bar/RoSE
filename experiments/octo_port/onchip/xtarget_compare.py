#!/usr/bin/env python3
"""Diff the --inspect dumps of two runs of the SAME IR, tensor by tensor.

    python xtarget_compare.py <run_a.log> <run_b.log>

Use it to localize where two targets stop agreeing. Note what it is measuring:
the int8 reference kernels that dequantize to float are compiled with gcc's
default -ffp-contract=fast, which fuses their multiply-accumulates on a target
that has FMA (RISC-V does, baseline x86-64 does not), so two targets running
the same IR are NOT expected to be bit-identical. Build both with
EXTRA_KERNEL_CFLAGS='-ffp-contract=off' if you want them to be.
"""
import re
import sys

import numpy as np

_RE = re.compile(
    r"=== MODELBLASTER_INSPECT_BEGIN \[(?P<name>[^\]]+)\] === "
    r"scale=(?P<scale>[^ ]+) dtype=(?P<dtype>\w+) n=(?P<n>\d+)\n"
    r"(?P<body>.*?)=== MODELBLASTER_INSPECT_END \[\1\] ===", re.DOTALL)
_OUT = re.compile(r"=== MODELBLASTER_OUTPUT_BEGIN ===\n(.*?)"
                  r"=== MODELBLASTER_OUTPUT_END ===", re.DOTALL)


def parse(path):
    text = open(path, errors="replace").read()
    out = {m.group("name"): np.fromstring(m.group("body"), sep="\n")
           for m in _RE.finditer(text)}
    m = _OUT.search(text)
    if m:
        out["<final output>"] = np.fromstring(m.group(1), sep="\n")
    return out


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    a, b = parse(argv[1]), parse(argv[2])
    shared = [n for n in a if n in b]
    if not shared:
        sys.exit("no tensor names in common -- were both runs built from the "
                 "same IR (same --inspect list)?")
    print(f"{'tensor':<16}{'n':>8}  {'ndiff':>7}{'%diff':>8}{'max|d| lsb':>12}")
    for n in shared:
        k = min(len(a[n]), len(b[n]))
        d = (a[n][:k] - b[n][:k]).astype(int)
        nz = int(np.count_nonzero(d))
        print(f"{n:<16}{k:>8}  {nz:>7}{100.0 * nz / max(k, 1):>7.2f}%"
              f"{int(np.abs(d).max()):>12}")


if __name__ == "__main__":
    main(sys.argv)
