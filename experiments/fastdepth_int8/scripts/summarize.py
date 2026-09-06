#!/usr/bin/env python3
"""Collate results/*.json into the fp32-vs-int8 table."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

R = {}
for p in glob.glob(os.path.join(C.RESULTS, "*.json")):
    d = json.load(open(p))
    if "metrics" in d:
        R[os.path.basename(p)[:-5]] = d

K = ("d1", "d2", "d3", "rmse", "rel", "log10")
LABEL = {
    "c1":       "int8  1 calib frame, per-tensor weights  (stock)",
    "c32":      "int8  32 calib frames, per-tensor weights",

    "c1pc":     "int8  1 calib, --per-channel  (dense convs only)",
    "c32pc":    "int8  32 calib, --per-channel",
    "c1ca":     "int8  1 calib, clamp-aware ranges",
    "c1pcdw":   "int8  1 calib, per-channel INCL depthwise",
    "c32pcdw":  "int8  32 calib, per-channel incl depthwise",
    "c1dwonly": "int8  1 calib, per-channel DEPTHWISE only (no dense)",
    "c1pcdwca": "int8  1 calib, per-channel incl depthwise + clamp-aware",
    "c8pcdwca": "int8  8 calib, per-channel incl depthwise + clamp-aware",
    "c32pcdwca": "int8 32 calib, per-channel incl depthwise + clamp-aware",
}
for ck, fp in (("epoch0", "fp32_e0_full"), ("epoch12", "fp32_e12_full")):
    if fp not in R:
        continue
    base = R[fp]["metrics"]
    print(f"\n=== {ck}  (n={R[fp]['n']} NYU val frames, per-image protocol) ===")
    print(f"{'arm':<52}" + "".join(f"{k:>9}" for k in K) + f"{'dd1':>9}{'drmse':>8}")
    print(f"{'fp32 reference':<52}" + "".join(f"{base[k]:9.4f}" for k in K))
    for v, lab in LABEL.items():
        tag = f"int8_{ck}_{v}"
        if tag not in R:
            continue
        m = R[tag]["metrics"]
        print(f"{lab:<52}" + "".join(f"{m[k]:9.4f}" for k in K)
              + f"{m['d1'] - base['d1']:+9.4f}{m['rmse'] - base['rmse']:+8.4f}")
