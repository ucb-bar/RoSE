#!/usr/bin/env python3
"""Compare a run's --inspect dumps against the PyTorch fp32 capture.

    python inspect_compare.py <run.log> <ir_dir>

`run.log` is the stdout of any runner (native / spike / firesim); `ir_dir` is
the extractor's --out-dir, which holds inspect_ref.npz. Both come from the same
extraction:

    EXTRACT_EXTRA_ARGS='--per-channel --inspect t1,t2,...' FORCE_EXTRACT=1 \
    QUANT=int8 RUNNER=native bash modelblaster/examples/octo_small/run.sh

Prints one row per inspected tensor: element count, the fp32 and dequantized
device maxima, the worst element error, and the cosine. The cosine is the
useful column -- it is magnitude-invariant, so a lowering bug shows up as a
collapse to ~0 at one specific tensor while everything upstream stays ~1.
"""
import re
import sys
from pathlib import Path

import numpy as np

_RE = re.compile(
    r"=== MODELBLASTER_INSPECT_BEGIN \[(?P<name>[^\]]+)\] === "
    r"scale=(?P<scale>[^ ]+) dtype=(?P<dtype>\w+) n=(?P<n>\d+)\n"
    r"(?P<body>.*?)=== MODELBLASTER_INSPECT_END \[\1\] ===", re.DOTALL)


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    text = Path(argv[1]).read_text(errors="replace")
    ref = np.load(Path(argv[2]) / "inspect_ref.npz")
    rows = []
    for m in _RE.finditer(text):
        name = m.group("name")
        vals = np.fromstring(m.group("body"), sep="\n").astype(np.float32)
        if m.group("dtype") == "i8":
            vals = vals * float(m.group("scale"))   # dequantize
        if name not in ref:
            rows.append((name, len(vals), np.nan,
                         float(np.abs(vals).max()), np.nan, np.nan))
            continue
        r = ref[name].astype(np.float32).ravel()
        k = min(len(r), len(vals))
        a, b = r[:k], vals[:k]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        rows.append((name, k, float(np.abs(a).max()), float(np.abs(b).max()),
                     float(np.abs(a - b).max()), cos))
    if not rows:
        sys.exit("no MODELBLASTER_INSPECT blocks in the log -- was the IR "
                 "extracted with --inspect, and rebuilt after?")
    print(f"{'tensor':<16}{'n':>9}  {'fp32 max':>11}{'dev max':>11}"
          f"{'max|d|':>11}{'cos':>9}")
    for name, n, fa, fb, d, cos in rows:
        print(f"{name:<16}{n:>9}  {fa:>11.4g}{fb:>11.4g}{d:>11.4g}{cos:>9.4f}")


if __name__ == "__main__":
    main(sys.argv)
