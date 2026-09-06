#!/usr/bin/env python3
"""How faithful is the quantised network to the FLOAT network it came from?

Ground-truth metrics can improve for the wrong reason -- a quantisation that
happens to shrink an over-dispersed prediction will score better on RMSE while
being a worse copy of the model. This measures the int8-vs-fp32 prediction
difference directly, with no ground truth involved, so it cannot be gamed that
way.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

fp = np.load(os.path.join(C.RESULTS, "pred_fp32_e0.npz"))["pred"]
rows = {}
for tag in sys.argv[1:]:
    p = np.load(os.path.join(C.RESULTS, f"pred_int8_e0_{tag}.npz"))["pred"]
    d = (p - fp).reshape(len(fp), -1)
    rows[tag] = {
        "rmse_vs_fp32_m": float(np.mean(np.sqrt((d ** 2).mean(axis=1)))),
        "max_abs_vs_fp32_m": float(np.abs(d).max()),
        "sqnr_db": float(10 * np.log10((fp ** 2).sum() / (d ** 2).sum())),
        "pred_std_ratio": float(p.std() / fp.std()),
    }
    print(f"{tag:<12} per-image RMSE vs fp32 {rows[tag]['rmse_vs_fp32_m']:.4f} m   "
          f"SQNR {rows[tag]['sqnr_db']:6.2f} dB   "
          f"std ratio {rows[tag]['pred_std_ratio']:.3f}")
print(C.save("fidelity_e0", {"arm": "fidelity", "n": len(fp), "metrics": {},
                             "fidelity": rows}))
