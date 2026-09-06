#!/usr/bin/env python3
"""Attribute the int8 depth loss: weights vs activations, and which tensors.

The per-layer SQNR ladder (layer_error.py) shows error ACCUMULATING but cannot
say what to fix -- a tensor with bad SQNR may simply have inherited it. This
does the controlled version: take the fp32 network and re-introduce ONE
quantisation effect at a time, holding everything else in float, then measure
the depth metrics that result.

Arms:
  fp32        untouched reference
  w_only      conv weights round-tripped through their int8 grid, activations float
  a_only      every activation fake-quantised at its IR scale, weights float
  a_<group>   activations fake-quantised for one group of tensors only
  single      one tensor at a time (ranked sensitivity list)

a_only + w_only vs the true int8 number is the decomposition that says which
of the two is worth engineering effort.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fd_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--ir", required=True)
ap.add_argument("-n", type=int, default=64)
ap.add_argument("--mode", default="groups",
                choices=["groups", "single", "weights"])
ap.add_argument("--tag", default="ablation")
ap.add_argument("--per-channel-w", action="store_true")
a = ap.parse_args()

C.apply_trained_env()
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.models import fastdepth as fdm
from modelblaster.pipeline import extract_graph as EG

ir = json.load(open(os.path.join(a.ir, "graph.json")))
scales = {k: float(v["quant"]["scale"]) for k, v in ir["tensors"].items()}
kind = {o["outputs"][0]: o["op"] for o in ir["ops"]}
in_name = ir["input"]["tensor"]
rgb, gt, _ = C.load_val(a.n)


def build(quant_weights: bool, pc_dense: bool = None, pc_dw: bool = None):
    """fp32 graph, optionally with conv weights round-tripped through their
    int8 grid. `pc_dense` / `pc_dw` select per-OUTPUT-CHANNEL scales for the
    groups==1 convs and the depthwise convs independently -- that split is the
    whole question, since _apply_per_channel in extract_graph converts
    conv2d_s8 but NOT depthwise_conv2d_s8."""
    if pc_dense is None:
        pc_dense = a.per_channel_w
    if pc_dw is None:
        pc_dw = a.per_channel_w
    gm = torch.fx.symbolic_trace(fdm.get_model())
    EG._fold_conv_bn(gm)
    if quant_weights:
        for m in gm.modules():
            if isinstance(m, torch.nn.Conv2d):
                w = m.weight.data
                pc = pc_dw if m.groups != 1 else pc_dense
                if pc:
                    s = (w.abs().flatten(1).max(1).values.clamp_min(1e-8)
                         / 127.0).view(-1, 1, 1, 1)
                else:
                    s = torch.clamp(w.abs().max(), min=1e-8) / 127.0
                m.weight.data = torch.clamp(torch.round(w / s), -127, 127) * s
    return gm


class FakeQuant(torch.fx.Interpreter):
    """fp32 forward with the named node outputs pushed through their int8 grid."""
    def __init__(self, gm, names):
        super().__init__(gm)
        self.names = names

    def run_node(self, n):
        out = super().run_node(n)
        if n.name in self.names and isinstance(out, torch.Tensor):
            s = scales[n.name]
            out = torch.clamp(torch.round(out / s), -127, 127) * s
        return out


def run(gm, names, bs=16):
    rows = []
    itp = FakeQuant(gm, names) if names else None
    with torch.no_grad():
        for s in range(0, len(rgb), bs):
            xb = torch.from_numpy(rgb[s:s + bs])
            y = (itp.run(xb) if itp is not None else gm(xb)).numpy()[:, 0]
            for i in range(y.shape[0]):
                rows.append(C.per_image(y[i], gt[s + i]))
    return C.aggregate(rows)


ALL = set(scales)
enc = {t for t in ALL if t.startswith("encoder")}
dec = {t for t in ALL if t.startswith("dec")}
skip = {t for t in ALL if t.startswith("skip_proj") or t.startswith("add_1")}
dw = {t for t, k in kind.items() if k == "depthwise_conv2d_s8"}
r6 = {t for t, k in kind.items() if k == "relu6_s8"}
proj = {t for t in enc if t.endswith("_conv_2")}     # linear bottleneck 1x1
gm_f = build(False)

out = {}
base = run(gm_f, set())
out["fp32"] = base
print(C.fmt(base, "fp32"), flush=True)

if a.mode == "groups":
    arms = [("w_only_pt", None, True), ("a_only", ALL, False),
            ("a_input", {in_name}, False), ("a_encoder", enc, False),
            ("a_decoder", dec | {"head"}, False), ("a_skip", skip, False),
            ("a_depthwise", dw, False), ("a_relu6", r6, False),
            ("a_enc_proj1x1", proj, False), ("a_head", {"head"}, False),
            ("both", ALL, True)]
    for label, names, qw in arms:
        g = build(True) if qw else gm_f
        ev = run(g, names or set())
        out[label] = ev
        print(C.fmt(ev, label) + f"   dd1 {ev['d1'] - base['d1']:+.4f}", flush=True)
elif a.mode == "weights":
    arms = [("w_pt_all", False, False), ("w_pc_dense_only", True, False),
            ("w_pc_dw_only", False, True), ("w_pc_all", True, True)]
    for label, pd, pw in arms:
        ev = run(build(True, pd, pw), set())
        out[label] = ev
        print(C.fmt(ev, label) + f"   dd1 {ev['d1'] - base['d1']:+.4f}", flush=True)
    for label, pd, pw in arms:
        ev = run(build(True, pd, pw), ALL)
        out[label + "+act"] = ev
        print(C.fmt(ev, label + "+act") + f"   dd1 {ev['d1'] - base['d1']:+.4f}",
              flush=True)
else:
    order = [in_name] + [o["outputs"][0] for o in ir["ops"]]
    seen, rank = set(), []
    for t in order:
        if t in seen or t not in scales:
            continue
        seen.add(t)
        ev = run(gm_f, {t})
        rank.append((base["d1"] - ev["d1"], t, kind.get(t, "input"), ev))
    rank.sort(reverse=True)
    print(f"\n{'tensor':<28}{'op':<22}{'dd1':>9}{'d_rmse':>9}")
    for d, t, k, ev in rank[:25]:
        print(f"{t:<28}{k:<22}{-d:9.4f}{ev['rmse'] - base['rmse']:9.4f}")
    out["single"] = [{"tensor": t, "op": k, "dd1": -d,
                      "d_rmse": ev["rmse"] - base["rmse"]} for d, t, k, ev in rank]

print(C.save(a.tag, {"ir": a.ir, "n": a.n, "mode": a.mode, "arms": out}))
