#!/usr/bin/env python3
"""Before/after max_abs_err + span for every re-measured arm."""
import os, re, glob, json
R="/scratch/dima/rose-infra/RoSE/experiments/sweep3net"
BEFORE=R+"/perhart_fix/before"
R1=R+"/perhart_fix/round1_conv_only"
ARMS=["yolov8_nano_serialP_base","yolov8_nano_gempair_base","yolov8_nano_gempair_shard","yolov8_nano_gempair_shardec",
      "dronet_serialP_base","dronet_gempair_base","dronet_gempair_shard","dronet_gempair_shardec",
      "vint_serialP_base","vint_gempair_base","vint_gempair_shard",
      "mlp_control_serialP_base","mlp_control_gempair_base"]
def err(p):
    try: t=open(p,errors='replace').read()
    except: return None
    m=re.search(r'max_abs_err=([0-9.e+-]+)', t)
    return m.group(1) if m else None
def span(p):
    try: t=open(p,errors='replace').read()
    except: return None
    hdr=None; iv=[]
    for line in t.split('\n'):
        line=line.strip()
        if line.startswith('entry_id,network'): hdr=line.split(','); continue
        if hdr is None: continue
        if line.startswith('==='): hdr=None; continue
        p2=line.split(',')
        if len(p2)!=len(hdr): continue
        try: int(p2[0])
        except: continue
        d=dict(zip(hdr,p2))
        if int(d['worker_hart'])<0: continue
        iv.append((int(d['actual_start_cycles']),int(d['actual_end_cycles'])))
    if not iv: return None
    return round((max(e for _,e in iv)-min(s for s,_ in iv))/1000.0,3)
def jid(tag):
    try:
        t=open(f"{R}/logs/fq_{tag}.log").read()
        m=re.search(r'jid=(\d+)',t); return m.group(1) if m else '?'
    except: return '?'
def jid_before(tag):
    try: return str(json.load(open(f"{R}/res_{tag}/runner.json"))['job_id'])
    except: return '?'
print(f"{'arm':34s} {'before':>14s} {'r1(conv)':>10s} {'after':>14s}   {'ms b':>8s} {'ms a':>8s}  job_before job_after")
for a in ARMS:
    b=f"{BEFORE}/{a}.uartlog"; c=f"{R}/res_{a}/uartlog"; r1=f"{R1}/{a}.uartlog"
    try: done = "FQDONE" in open(f"{R}/logs/fq_{a}.log").read()
    except: done = False
    ea, sa, ja = (err(c), span(c), jid(a)) if done else ("PENDING", "-", "-")
    print(f"{a:34s} {str(err(b)):>14s} {str(err(r1)):>10s} {str(ea):>14s}   {str(span(b)):>8s} {str(sa):>8s}  {ja:>9s}")
