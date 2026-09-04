#!/usr/bin/env python3
"""Export a ModelBlaster campaign model to an ExecuTorch .pte, with the SAME
architecture, SAME shapes and SAME sample input ModelBlaster uses.

This is the ET counterpart of `modelblaster/examples/<name>/<quant>/`. The
point of the exercise is a like-for-like ET-vs-MB comparison on the F2 quad
bitstream, so everything that determines cost or numerics is taken from
`modelblaster.models.<name>` rather than re-declared here:

  * `get_model()`       -- identical module, identical (pretrained) weights
  * `get_sample_input()`-- identical, deterministic input tensor(s)

Writes, per model, into --outdir:
  <tag>.pte        the serialized program
  <tag>.io.npz     input(s) (fp32, as fed to the runner), fp32 torch golden,
                   and the host ExecuTorch output for the SAME input
  <tag>.json       export report: delegate/aten node counts, undelegated op
                   histogram, pte size, host-reference errors

The undelegated-op histogram is the answer to "what does this lower to": any
node NOT inside a `call_delegate` runs as an ExecuTorch *portable* kernel,
which is plain scalar C++ compiled at the runner's -march. Only the delegated
subgraphs reach XNNPACK's RVV micro-kernels.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

# The RoSE ModelBlaster tree (models + their pretrained checkpoints).
#
# GATE -- do not simplify this to a bare sys.path.insert(0, MB_REPO). The
# ExecuTorch env carries an editable install
# (`_editable_impl_modelblaster.pth` -> <old tree>/modelblaster/src), and RoSE's
# modelblaster/ has no top-level __init__.py, so it can only ever be a
# *namespace* package. Python resolves a REGULAR package anywhere on sys.path
# in preference to a namespace portion at sys.path[0] -- so the old tree wins
# silently and you export a DIFFERENT model (measured: mlp_control, dronet and
# yolov8_nano all differ between the two trees). Bind the package root
# explicitly instead, and assert afterwards.
import types                                                        # noqa: E402
MB_REPO = os.environ.get(
    "MB_REPO", "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
sys.path.insert(0, MB_REPO)
_pkg = types.ModuleType("modelblaster")
_pkg.__path__ = [os.path.join(MB_REPO, "modelblaster")]
sys.modules["modelblaster"] = _pkg

from torch.export import export                                    # noqa: E402
from executorch.exir import to_edge, to_edge_transform_and_lower   # noqa: E402
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: E402
    XnnpackPartitioner)


# --------------------------------------------------------------------------
# model registry -- everything comes from modelblaster.models.<name>
# --------------------------------------------------------------------------
def load_mb_model(name):
    """-> (module.eval(), sample_args_tuple, calib_list)."""
    mod = __import__(f"modelblaster.models.{name}", fromlist=["*"])
    if not os.path.realpath(mod.__file__).startswith(os.path.realpath(MB_REPO)):
        raise SystemExit(
            f"ABORT: modelblaster.models.{name} resolved to {mod.__file__}, "
            f"which is OUTSIDE MB_REPO={MB_REPO}. Exporting a model from the "
            f"wrong tree makes every number in this experiment meaningless.")
    print(f"[provenance] {name} <- {mod.__file__}", flush=True)
    m = mod.get_model().eval()
    s = mod.get_sample_input()
    args = (s,) if torch.is_tensor(s) else tuple(s)

    # Calibration set for PTQ. Prefer the model's own (ViNT ships a real one
    # and explicitly warns that gaussian noise makes its scales meaningless);
    # otherwise reuse the real sample input plus small jitters of it, which
    # keeps the activation ranges in the deployment distribution -- the same
    # reasoning yolov8_nano's get_sample_input docstring gives for using a
    # real frame rather than randn.
    calib = None
    if hasattr(mod, "get_calibration_samples"):
        try:
            got = mod.get_calibration_samples(8)
            calib = [(g,) if torch.is_tensor(g) else tuple(g) for g in got]
        except Exception as e:            # checkpoint/data may be absent
            print(f"[calib] {name}: get_calibration_samples failed ({e});"
                  " falling back to jittered sample input", flush=True)
    if not calib:
        g = torch.Generator().manual_seed(7)
        calib = [args] + [
            tuple(a + 0.02 * torch.randn(a.shape, generator=g) for a in args)
            for _ in range(3)]
    return m, args, calib


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_fp32(model, args):
    ep = export(model, args)
    return to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])


def export_int8(model, args, calib):
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    try:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    except ImportError:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    try:
        from torch.export import export_for_training as _capture
    except ImportError:
        from torch._export import capture_pre_autograd_graph as _capture
    cap = _capture(model, args)
    cap = cap.module() if hasattr(cap, "module") else cap
    q = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True))
    prep = prepare_pt2e(cap, q)
    for c in calib:
        prep(*c)
    quant = convert_pt2e(prep)
    ep = export(quant, args)
    return to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])


def graph_stats(edge):
    """Delegated vs portable-aten node counts + the undelegated op histogram."""
    gm = edge.exported_program().graph_module
    deleg, aten, hist = 0, 0, {}
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        t = str(n.target)
        if "call_delegate" in t or "executorch_call_delegate" in t:
            deleg += 1
            continue
        op = str(getattr(n.target, "_op", n.target))
        # edge-dialect ops read like `aten::convolution` / `dim_order_ops::...`
        aten += 1
        hist[op] = hist.get(op, 0) + 1
    return deleg, aten, hist


# --------------------------------------------------------------------------
# host reference: run the SAME .pte through ExecuTorch's host runtime
# --------------------------------------------------------------------------
def host_reference(pte_path, args):
    from executorch.runtime import Runtime
    rt = Runtime.get()
    prog = rt.load_program(pte_path)
    meth = prog.load_method("forward")
    out = meth.execute(list(args))
    return [o.detach().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
            for o in (out if isinstance(out, (list, tuple)) else [out])]


def flat(xs):
    return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in xs])


def errs(a, b):
    a, b = flat(a), flat(b)
    n = min(a.size, b.size)
    if n == 0 or a.size != b.size:
        return {"shape_mismatch": [int(a.size), int(b.size)]}
    d = np.abs(a - b)
    den = np.maximum(np.abs(b), 1e-9)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
            "max_rel_err": float((d / den).max()), "cosine": cos,
            "numel": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", choices=["fp32", "int8"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-host-ref", action="store_true")
    a = ap.parse_args()
    tag = a.tag or f"{a.model}_{a.quant}"
    os.makedirs(a.outdir, exist_ok=True)
    pte = os.path.join(a.outdir, f"{tag}.pte")
    rep = {"tag": tag, "model": a.model, "quant": a.quant}

    t0 = time.time()
    torch.manual_seed(0)
    model, args, calib = load_mb_model(a.model)
    rep["input_shapes"] = [list(x.shape) for x in args]

    with torch.no_grad():
        golden = model(*args)
    golden = [g for g in (golden if isinstance(golden, (list, tuple)) else [golden])]
    golden_np = [g.detach().cpu().numpy() for g in golden]
    rep["output_shapes"] = [list(g.shape) for g in golden_np]

    edge = (export_fp32(model, args) if a.quant == "fp32"
            else export_int8(model, args, calib))
    deleg, aten, hist = graph_stats(edge)
    prog = edge.to_executorch()
    with open(pte, "wb") as f:
        prog.write_to_file(f)
    rep.update(delegates=deleg, undelegated_aten=aten,
               undelegated_ops=dict(sorted(hist.items(), key=lambda kv: -kv[1])),
               pte_bytes=os.path.getsize(pte), export_secs=round(time.time() - t0, 1))

    host = None
    if not a.no_host_ref:
        try:
            host = host_reference(pte, args)
            rep["host_vs_torch"] = errs(host, golden_np)
        except Exception as e:
            rep["host_ref_error"] = f"{type(e).__name__}: {e}"

    np.savez(os.path.join(a.outdir, f"{tag}.io.npz"),
             **{f"input{i}": x.detach().cpu().numpy().astype(np.float32)
                for i, x in enumerate(args)},
             **{f"golden{i}": g.astype(np.float32) for i, g in enumerate(golden_np)},
             **({f"host{i}": h.astype(np.float32) for i, h in enumerate(host)}
                if host else {}))
    with open(os.path.join(a.outdir, f"{tag}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
