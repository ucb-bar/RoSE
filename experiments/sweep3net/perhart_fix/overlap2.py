#!/usr/bin/env python3
"""Cross-hart concurrent dispatch pairs, histogrammed by (opA,opB)."""
import sys, collections
def rows(path):
    hdr=None; out=[]
    for line in open(path, errors='replace'):
        line=line.strip()
        if line.startswith('entry_id,network'): hdr=line.split(','); continue
        if hdr is None: continue
        if line.startswith('==='): hdr=None; continue
        p=line.split(',')
        if len(p)!=len(hdr): continue
        try: int(p[0])
        except: continue
        out.append(dict(zip(hdr,p)))
    return out
for path in sys.argv[1:]:
    rs=rows(path); iv=[]
    for r in rs:
        try: s=int(r['actual_start_cycles']); e=int(r['actual_end_cycles']); h=int(r['worker_hart'])
        except: continue
        if h<0: continue
        iv.append((s,e,r['op'],r['name'],h))
    iv.sort()
    hist=collections.Counter(); ex=collections.defaultdict(list)
    for i in range(len(iv)):
        for j in range(i+1,len(iv)):
            if iv[j][0]>=iv[i][1]: break
            if iv[i][4]==iv[j][4]: continue
            k=tuple(sorted((iv[i][2],iv[j][2])))
            hist[k]+=1; ex[k].append((iv[i][3],iv[i][4],iv[j][3],iv[j][4]))
    print(f"=== {path} ndisp={len(iv)} total_cross_hart_overlaps={sum(hist.values())}")
    for k,c in hist.most_common():
        print(f"    {k[0]:22s} || {k[1]:22s}  x{c}   e.g. {ex[k][0][0]}@h{ex[k][0][1]} || {ex[k][0][2]}@h{ex[k][0][3]}")
