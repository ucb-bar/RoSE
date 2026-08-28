#!/usr/bin/env python3
import json, os, sys, collections
sys.path.insert(0,"/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw")
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
from modelblaster.pipeline.generate_kernels import shapes_from_ir
MB=os.environ["MB"]
MODELS=[("dronet","int8"),("fused_full","int8"),("fused_full","fp32"),
        ("mlp_generic","int8"),("yolov8_nano","int8")]
VLMAX_E32M4=32   # OC lanes per vector at e32m4, VLEN=256

def load():
    d=collections.defaultdict(dict)
    for n,q in MODELS:
        g=f"{MB}/examples/{n}/{q}/generated/graph.json"
        if not os.path.exists(g): continue
        ir=json.load(open(g))
        for op in sorted({o["op"] for o in ir["ops"]}):
            if op not in KERNEL_SPECS: continue
            try: sh=shapes_from_ir(ir,op)
            except Exception: continue
            if sh: d[op][f"{n}/{q}"]=sh
    return d

def sig_conv(s):
    return (f"K{s.get('KH')}x{s.get('KW')}", f"S{s.get('SH')}", f"P{s.get('PH')}",
            "IC1" if s.get('IC')==1 else None,
            "OCtail" if s.get('OC') and s['OC']%VLMAX_E32M4 else None,
            "OWlt4" if s.get('OW') is not None and s['OW']<4 else None)
def sig_pool(s):
    return (f"K{s.get('KH')}x{s.get('KW')}", f"S{s.get('SH')}", f"P{s.get('PH')}",
            f"D{s.get('DH')}")
def sig_lin(s):
    return ("N1" if s.get('N')==1 else None, "K1" if s.get('K')==1 else None,
            "Ktail" if s.get('K') and s['K']%VLMAX_E32M4 else None,
            "Ntail" if s.get('N') and s['N']%VLMAX_E32M4 else None)
SIG={"conv2d_s8":sig_conv,"conv2d":sig_conv,"maxpool2d_s8":sig_pool,
     "linear_s8":sig_lin,"linear_f16":sig_lin,"linear":sig_lin}

data=load()
print("="*90)
print("BRANCH-PATH COVERAGE  (only values that select a CODE PATH, not loop sizing)")
print("="*90)
rows=[]
for op in sorted(data):
    f=SIG.get(op)
    if not f: continue
    models=data[op]; spec=KERNEL_SPECS[op]; extra=spec.extra_shapes or []
    ecov=set()
    for s in extra:
        for t in f(s):
            if t: ecov.add(t)
    mcov=collections.defaultdict(set)
    for m,shs in models.items():
        for s in shs:
            for t in f(s):
                if t: mcov[t].add(m)
    missing={t:sorted(ms) for t,ms in mcov.items() if t not in ecov}
    shared=len(models)>1
    print(f"\n-- {op}  [{'SHARED' if shared else 'single'}: {', '.join(sorted(models))}]")
    print(f"   extra_shapes covers: {sorted(ecov)}")
    if not missing:
        print("   ok: no uncovered code path")
    for t,ms in sorted(missing.items()):
        crit = "CRITICAL" if shared and len(ms)<len(models) else ("note" if shared else "single")
        print(f"   [{crit}] path '{t}' used by {'+'.join(ms)} but in NO extra_shape")
        rows.append((op,t,ms,crit))
print("\n"+"="*90)
print("SHARED-OP CODE PATHS WITH NO VERIFY COVERAGE:")
for op,t,ms,c in rows:
    if c=="CRITICAL": print(f"   {op:<16} {t:<10} only in {'+'.join(ms)}")
