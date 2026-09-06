#!/usr/bin/env python3
"""Host-compile kernel_depthwise_conv2d_s8_pc and check it against the golden.

A new op kind is not "added" until its C reference actually runs and agrees
with the Python simulator that produces the golden -- otherwise the IR is
emitting an op nothing can execute, and the bit-exactness we verify is
between two halves of the same Python.

Uses the REAL quant parameters, weights and activations from an extracted
FastDepth IR (every depthwise_conv2d_s8_pc op in it), not synthetic ones, so
the per-channel multiplier/shift arrays under test are the ones the model
would actually deploy with.
"""
import ctypes, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
import int8_fast  # noqa: F401  (imports extract_graph too)
from modelblaster.pipeline import verify_kernel as VK
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
from modelblaster.pipeline.extract_graph import _sim_depthwise_conv2d_s8_pc

ir_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.ROOT, "ir", "epoch0_c1pcdw")
ir = json.load(open(os.path.join(ir_dir, "graph.json")))
W = dict(np.load(os.path.join(ir_dir, "weights.npz")))
io = np.load(os.path.join(ir_dir, "io.npz"))

spec = KERNEL_SPECS["depthwise_conv2d_s8_pc"]
import tempfile
work = tempfile.mkdtemp(prefix="dwpc_")
so = VK.host_compile(spec.reference_impl, "dwpc", work)
lib = ctypes.CDLL(so)
fn = lib.kernel_depthwise_conv2d_s8_pc
fn.argtypes = spec.argtypes_factory()
fn.restype = None

# real activations for every op in the graph
x = io["input"].reshape(1, 3, 224, 224).astype(np.int8)
_, act = int8_fast.run_ir(ir, W, x, keep_all=True)

ok, n = True, 0
for op in ir["ops"]:
    if op["op"] != "depthwise_conv2d_s8_pc":
        continue
    sh, q = op["shape"], op["quant"]
    Cc = int(sh.get("groups", sh["OC"]))
    a0 = np.ascontiguousarray(act[op["inputs"][0]].reshape(-1).astype(np.int8))
    w = np.ascontiguousarray(W[op["weight"]].reshape(-1).astype(np.int8))
    b = np.ascontiguousarray(W[op["bias"]].reshape(-1).astype(np.int32))
    m = np.ascontiguousarray(W[q["output_multiplier_per_oc_key"]].astype(np.int32))
    s = np.ascontiguousarray(W[q["output_shift_per_oc_key"]].astype(np.int32))
    out = np.zeros(sh["N"] * Cc * sh["OH"] * sh["OW"], dtype=np.int8)
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    fn(a0.ctypes.data_as(i8p), w.ctypes.data_as(i8p), b.ctypes.data_as(i32p),
       out.ctypes.data_as(i8p),
       sh["N"], Cc, sh["IH"], sh["IW"], sh["KH"], sh["KW"],
       sh["SH"], sh["SW"], sh["PH"], sh["PW"],
       q["input_offset"], q["filter_offset"], q["output_offset"],
       m.ctypes.data_as(i32p), s.ctypes.data_as(i32p),
       q["activation_min"], q["activation_max"])
    gold = _sim_depthwise_conv2d_s8_pc(a0, sh, q, W[op["weight"]], W[op["bias"]], m, s)
    d = int(np.abs(out.astype(np.int32) - gold.astype(np.int32)).max())
    n += 1
    if d:
        ok = False
        print(f"  MISMATCH {op['name']:<28} C={sh['KH']}x{sh['KW']} max_abs_err={d}")
print(f"depthwise_conv2d_s8_pc: {n} ops from {os.path.basename(ir_dir)}, "
      f"C reference vs golden max_abs_err=0 for all: {ok}")
sys.exit(0 if ok and n else 1)
