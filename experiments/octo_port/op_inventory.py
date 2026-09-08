#!/usr/bin/env python
"""Op inventory for the Octo port, against BOTH ModelBlaster extractors.

    PYTHONPATH=$ZCS python op_inventory.py [--part full|backbone|score]

Produces two histograms and classifies each against the extractor that would
consume it:

  1. torch.fx level  -> pipeline/extract_graph.py      (--quant fp32/fp16/int8)
  2. aten level      -> pipeline/extract_graph_export.py (--quant int8/fp16)

`SUPPORTED_MODULES` is imported from the FX extractor so the module table
cannot drift from this report. The FX extractor's call_function / call_method
handling is an inline if/elif chain with no importable table, so those names
are listed here from a read of extract_graph.py at the cited lines and the
classification is CONFIRMED by actually running the extractor (see NOTES.md
§10 for what it raised). The export extractor's tables ARE importable and are
used directly.
"""
from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
_ZCS = HERE.parents[1] / "soc/sw/xpu-rt/zephyr-chipyard-sw"
if str(_ZCS) not in sys.path:
    sys.path.insert(0, str(_ZCS))

import torch  # noqa: E402
import torch.fx  # noqa: E402

# Tables imported from the extractors themselves.
from modelblaster.pipeline.extract_graph import SUPPORTED_MODULES  # noqa: E402
from modelblaster.pipeline.extract_graph_export import (  # noqa: E402
    _ALIAS, _FOLD, _NEW_COMPUTE, _NOOP, _SUPPORTED_COMPUTE, _SWISH, _TAIL,
)

# FX fp32 call_function targets, from extract_graph.py:4470-5278.
FX_FN_OK = {
    "relu", "flatten", "add", "adaptive_avg_pool2d", "relu6", "sigmoid",
    "softmax", "log_softmax", "log", "smooth_l1_loss", "cross_entropy",
    "scaled_dot_product_attention", "kl_div", "elu", "leaky_relu", "tanh",
    "gelu", "selu", "hardsigmoid", "softplus", "cumsum", "cumprod", "mul",
    "sum", "mean", "prod", "argmax", "argmin", "max", "min", "hardtanh",
    "matmul", "mm", "bmm", "triu", "tril", "einsum", "cat", "concat",
    "getitem", "truediv",
}
FX_METHOD_OK = {"chunk", "flip"}       # extract_graph.py:5280-5371


def fx_report(model, sample) -> None:
    gm = torch.fx.symbolic_trace(model)
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(gm).propagate(*sample)

    mods = dict(gm.named_modules())
    hist: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        if n.op == "call_module":
            m = mods[n.target]
            name = type(m).__name__
            cls = "supported" if isinstance(m, SUPPORTED_MODULES) else "REJECTED"
            hist[cls][f"nn.{name}"] += 1
        elif n.op == "call_function":
            name = getattr(n.target, "__name__", str(n.target))
            hist["supported" if name in FX_FN_OK else "REJECTED"][
                f"fn:{name}"] += 1
        elif n.op == "call_method":
            hist["supported" if n.target in FX_METHOD_OK else "REJECTED"][
                f"method:{n.target}"] += 1
        elif n.op == "get_attr":
            hist["REJECTED"]["get_attr (raise at extract_graph.py:5382)"] += 1

    print("\n=== 1. torch.fx level -> extract_graph.py (--quant fp32) ===")
    for cls in ("supported", "REJECTED"):
        if not hist[cls]:
            continue
        print(f"\n  [{cls}] {sum(hist[cls].values())} nodes")
        for k, v in hist[cls].most_common():
            print(f"    {v:5d}  {k}")
    return hist


def aten_report(model, sample) -> None:
    ep = torch.export.export(model, tuple(sample), strict=False)
    hist: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)

    def cls_of(name: str) -> str:
        if name in _SUPPORTED_COMPUTE: return "supported"
        if name in _NEW_COMPUTE: return "new"
        if name in _SWISH: return "swish"
        if name in _ALIAS: return "alias"
        if name in _FOLD: return "fold"
        if name in _NOOP: return "noop"
        if name in _TAIL: return "tail"
        return "UNKNOWN"

    for n in ep.graph_module.graph.nodes:
        if n.op in ("placeholder", "get_attr", "output"):
            continue
        if n.op == "call_function":
            name = str(n.target).split(".OverloadPacket")[0].replace("aten.", "")
        elif n.op == "call_method":
            name = f"method:{n.target}"
        else:
            name = n.op
        hist[cls_of(name)][name] += 1

    total = sum(sum(c.values()) for c in hist.values())
    print(f"\n=== 2. aten level -> extract_graph_export.py  ({total} nodes) ===")
    for cls in ("supported", "new", "fold", "alias", "noop", "tail", "swish",
                "UNKNOWN"):
        if not hist[cls]:
            continue
        print(f"\n  [{cls}] {sum(hist[cls].values())} nodes")
        for k, v in hist[cls].most_common():
            print(f"    {v:5d}  {k}")
    return hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=os.environ.get(
        "MODELBLASTER_OCTO_PART", "full"))
    a = ap.parse_args()
    os.environ["MODELBLASTER_OCTO_PART"] = a.part

    from modelblaster.models.octo_small import (  # noqa: PLC0415
        _cfg, forward_arg_names, get_model, get_sample_input)
    cfg = _cfg()
    print(f"PART={a.part}  window={cfg['window']}  primary={cfg['primary']}  "
          f"wrist={cfg['wrist']}  layers={cfg['layers']}  gn={cfg['gn']}  "
          f"attn={cfg['attn']}")
    print(f"forward({', '.join(forward_arg_names(cfg))})")
    model = get_model()
    sample = get_sample_input()

    fx_ok = True
    try:
        fx_report(model, sample)
    except Exception as e:                                  # noqa: BLE001
        fx_ok = False
        print(f"\n=== 1. torch.fx level: TRACE/PROP FAILED ===\n  "
              f"{type(e).__name__}: {e}")
    try:
        aten_report(model, sample)
    except Exception as e:                                  # noqa: BLE001
        print(f"\n=== 2. aten level: EXPORT FAILED ===\n  "
              f"{type(e).__name__}: {e}")
    return 0 if fx_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
