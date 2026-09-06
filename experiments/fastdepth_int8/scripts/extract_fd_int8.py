#!/usr/bin/env python3
"""Extract an int8 FastDepth IR under a chosen calibration configuration.

Calls the SHIPPED extract_int8() -- no fork of the quantiser -- with
int8_fast.patch() applied so its own golden simulation finishes in minutes
rather than hours. The patch only replaces the conv accumulate loops with an
im2col matmul over the same integers; verify_fast_sim.py proves that
substitution is bit-exact.

  --num-calibration N   frames from the calibration bank fed to extract_int8
  --calib PATH          the bank (npz, key "samples", ImageNet-normalised)
  --per-channel         per-output-channel weight scales (conv2d_s8_pc)
  --per-channel-depthwise  extend that to depthwise_conv2d_s8_pc
  --clamp-aware         MB_INT8_CLAMP_AWARE_RANGES=1
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--num-calibration", type=int, default=1)
ap.add_argument("--calib", default=None)
ap.add_argument("--per-channel", action="store_true")
ap.add_argument("--per-channel-depthwise", action="store_true")
ap.add_argument("--clamp-aware", action="store_true")
ap.add_argument("--slow", action="store_true", help="skip the fast patch")
a = ap.parse_args()

if a.clamp_aware:
    os.environ["MB_INT8_CLAMP_AWARE_RANGES"] = "1"
C.apply_trained_env(a.calib)
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
import int8_fast
from modelblaster.models import fastdepth as fdm
from modelblaster.pipeline.extract_graph import extract

if not a.slow:
    int8_fast.patch()

model = fdm.get_model()
sample = fdm.get_sample_input()
cal = None
if a.num_calibration > 1:
    cal = list(fdm.get_calibration_samples(a.num_calibration))
    sample = cal[0]
    print(f"[extract] {len(cal)} calibration samples", flush=True)
t0 = time.time()
extract(model, sample, name="fastdepth", out_dir=a.out_dir, quant="int8",
        calibration_samples=cal, per_channel=a.per_channel,
        per_channel_depthwise=a.per_channel_depthwise)
print(f"[extract] {time.time() - t0:.1f}s -> {a.out_dir}")
