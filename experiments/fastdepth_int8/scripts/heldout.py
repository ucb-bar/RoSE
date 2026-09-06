#!/usr/bin/env python3
"""Re-score fp32 and int8 on the val frames NOT used for calibration.

make_calib.py draws its 32 calibration frames from the VAL split
(np.linspace(0, 653, 32)), so the headline int8 numbers are measured partly on
frames whose activation ranges the quantiser saw. The leakage should be tiny
-- calibration only fixes per-tensor max-abs, not weights -- but "should be
tiny" is not a measurement, so this excludes those indices and re-scores from
the saved predictions (no new inference, so it cannot disagree by accident).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

_, gt, names = C.load_val(None)
calib_idx = set(np.linspace(0, len(gt) - 1, 32).astype(int).tolist())
keep = [i for i in range(len(gt)) if i not in calib_idx]
print(f"held-out: {len(keep)} of {len(gt)} frames "
      f"({len(calib_idx)} calibration frames excluded)")

out = {}
for label, f in [("fp32", "pred_fp32_e0.npz")] + \
                [(t, f"pred_int8_e0_{t}.npz") for t in sys.argv[1:]]:
    p = np.load(os.path.join(C.RESULTS, f))["pred"]
    full = C.aggregate([C.per_image(p[i], gt[i]) for i in range(len(gt))])
    held = C.aggregate([C.per_image(p[i], gt[i]) for i in keep])
    out[label] = {"all": full, "heldout": held}
    print(f"{label:<12} all654  d1 {full['d1']:.4f} rmse {full['rmse']:.4f}   "
          f"heldout622  d1 {held['d1']:.4f} rmse {held['rmse']:.4f}   "
          f"shift d1 {held['d1'] - full['d1']:+.4f} rmse {held['rmse'] - full['rmse']:+.4f}")
print(C.save("heldout_e0", {"arm": "heldout", "n": len(keep), "metrics": {},
                            "heldout": out}))
