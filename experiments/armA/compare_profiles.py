#!/usr/bin/env python3
"""armA vs firesim_f2_rocket_saturn reference: per-op profile deltas."""
import csv, glob, collections, sys
X="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"
def load(tag,hw,m,q):
    p=glob.glob(f"{X}/gen/profile/{hw}/{tag}/{m}/{m}.{q}/*/topo_0/results.csv")
    if not p: return None
    return list(csv.DictReader(open(p[0])))
tot={}
for hw in ("gemmini_q31","V256D128_rvv"):
    for m,q in (("mlp_control","fp32"),("dronet","int8"),("yolov8_nano","int8")):
        a=load("firesim_f2_armA",hw,m,q); b=load("firesim_f2_rocket_saturn",hw,m,q)
        if a is None: print("MISSING armA",hw,m); continue
        ta=sum(float(r["mean_time"]) for r in a); tot[(hw,m)]=ta
        tb=sum(float(r["mean_time"]) for r in b) if b else float('nan')
        print(f"{hw:14s} {m:12s} armA {ta:9.4f} ms   ref {tb:9.4f} ms   x{tb/ta:5.3f}")
        if b:
            ag=collections.defaultdict(lambda:[0,0.0,0.0])
            bb={int(r['dispatch_id']):float(r['mean_time']) for r in b}
            for r in a:
                d=int(r['dispatch_id']); op=r['op']
                ag[op][0]+=1; ag[op][1]+=float(r['mean_time']); ag[op][2]+=bb.get(d,0.0)
            for op,(n,va,vb) in sorted(ag.items(),key=lambda kv:-kv[1][2]):
                if vb==0 and va==0: continue
                print(f"     {op:22s} n={n:3d}  armA {va:8.4f}  ref {vb:8.4f}  x{(vb/va) if va else 0:5.3f}")
print()
print("PROFILE TOTALS (ms):")
for k in sorted(tot): print(f"  {k[0]:14s} {k[1]:12s} {tot[k]:9.4f}")
