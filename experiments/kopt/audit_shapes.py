#!/usr/bin/env python3
"""Audit extra_shapes coverage for SHARED kernels.

verify uses collect_shapes = shapes_from_ir(model) + spec.extra_shapes.
So editing a shared kernel while building model M tests it on M's shapes
plus extra_shapes ONLY. Any shape characteristic that lives solely in
model N is invisible -- that is how the SPPF maxpool gap happened.

A gap here = a shape-key VALUE that some model uses and no extra_shape
covers. Ops used by 2+ models are the dangerous ones.
"""
import json, os, sys, collections
sys.path.insert(0, "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
from modelblaster.pipeline.generate_kernels import shapes_from_ir

MB = os.environ["MB"]
MODELS = [("dronet","int8"),("fused_full","int8"),("fused_full","fp32"),
          ("mlp_generic","int8"),("yolov8_nano","int8")]

# keys that steer control flow rather than just sizing loops
STRUCTURAL = {"KH","KW","SH","SW","PH","PW","DH","DW","IC","OC","C","N","K","M"}

per_op_model = collections.defaultdict(lambda: collections.defaultdict(list))
for name, q in MODELS:
    g = f"{MB}/examples/{name}/{q}/generated/graph.json"
    if not os.path.exists(g): continue
    ir = json.load(open(g))
    for op in sorted({o["op"] for o in ir["ops"]}):
        if op not in KERNEL_SPECS: continue
        try: sh = shapes_from_ir(ir, op)
        except Exception: continue
        if sh: per_op_model[op][f"{name}/{q}"] = sh

print("="*88)
print("extra_shapes COVERAGE AUDIT  (ops used by 2+ models are the cross-agent hazard)")
print("="*88)
gaps=[]
for op in sorted(per_op_model):
    models = per_op_model[op]
    spec = KERNEL_SPECS[op]
    extra = spec.extra_shapes or []
    shared = len(models) > 1
    # value sets
    model_vals = collections.defaultdict(lambda: collections.defaultdict(set))
    for m, shs in models.items():
        for s in shs:
            for k,v in s.items():
                if k in STRUCTURAL: model_vals[k][m].add(v)
    extra_vals = collections.defaultdict(set)
    for s in extra:
        for k,v in s.items():
            if k in STRUCTURAL: extra_vals[k].add(v)
    op_gaps=[]
    for k in sorted(model_vals):
        allv=set()
        for m in model_vals[k]: allv |= model_vals[k][m]
        missing = allv - extra_vals.get(k,set())
        if missing and extra:
            owners={}
            for v in sorted(missing):
                who=[m for m in model_vals[k] if v in model_vals[k][m]]
                owners[v]=who
            op_gaps.append((k, owners))
    tag = "SHARED" if shared else "single-model"
    print(f"\n-- {op}   [{tag}: {', '.join(sorted(models))}]   extra_shapes={len(extra)}")
    if not extra:
        print("   !! NO extra_shapes AT ALL -- verified only on whatever model is being built")
        gaps.append((op, shared, "no extra_shapes", None))
        continue
    if not op_gaps:
        print("   ok: every structural value used by a model appears in extra_shapes")
    for k, owners in op_gaps:
        s=", ".join(f"{v}({'+'.join(o)})" for v,o in list(owners.items())[:6])
        print(f"   GAP {k}: values not in any extra_shape -> {s}")
        gaps.append((op, shared, k, owners))
print("\n"+"="*88)
sh_gaps=[g for g in gaps if g[1]]
print(f"SUMMARY: {len(sh_gaps)} gap(s) on SHARED (2+ model) ops, {len(gaps)-len(sh_gaps)} on single-model ops")
