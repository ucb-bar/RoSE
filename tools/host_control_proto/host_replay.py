#!/usr/bin/env python3
"""STEP 0 + Threads A/B: replay a REAL co-sim policy trace through the corrected host plant.

Instead of a synthetic (yr,fwd), replay the ACTUAL vision-model output captured from a co-sim run
(m4y_sim.log: 'fused: yr_m/fwd_m' at 10 Hz) plus the reconstructed desired_vel goal-guidance, through
the yawfix linear plant + host TinyMPC. This makes the host predictive of the real vision-in-the-loop
behavior (noisy, high-|yr| baseline). Used to:
  - STEP 0: confirm the host reproduces the co-sim (over-swing / deadlock) on the REAL trace.
  - THREAD B: sweep Q_yaw/Q_yawrate/R/u-clamp -> does the thrust plant TRACK the real continuous yr?
  - THREAD A: misalignment forward gate from desired_vel (goes to 0 on alignment, unlike |yr|).
"""
import numpy as np, re, math, argparse
PD="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/examples/problem_data/"
S="/tmp/claude-1172/-scratch-dima-rose-infra-RoSE/d4827fc8-b516-4227-b71d-79192ba241cd/scratchpad/"
GATES=np.array([[-8.05,9.0],[-8.30,13.0],[-7.75,17.0],[-8.05,21.0]]); PASS_R=1.2; BASE_SPEED=1.4
DT=0.005; UMIN0,UMAX0=-0.583,0.417; SETTLE=200; VDIV=20

def load_hpp(f):
    t=open(PD+f).read()
    b=lambda n,s: np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",re.search(n+r"\[[^\]]*\]\s*=\s*\{([^}]*)\}",t).group(1))]).reshape(s)
    Q=np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",re.search(r"Q_data\[[^\]]*\]\s*=\s*\{([^}]*)\}",t).group(1))])
    R=np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",re.search(r"R_data\[[^\]]*\]\s*=\s*\{([^}]*)\}",t).group(1))])
    rho=float(re.search(r"rho_value\s*=\s*([-\d.eE+]+)",t).group(1))
    return b("Adyn_data",(12,12)),b("Bdyn_data",(12,4)),Q,R,rho

def dare(A,B,Qv,Rv,rho):
    Q=np.diag(Qv)+rho*np.eye(12); R=np.diag(Rv)+rho*np.eye(4); P=np.diag(Qv).copy()
    for _ in range(20000):
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A); Pn=np.diag(Qv)+rho*np.eye(12)+A.T@P@(A-B@K)
        if np.max(np.abs(Pn-P))<1e-11: P=Pn;break
        P=Pn
    return K

def load_real_trace():
    yr=[];fwd=[]
    for L in open(S+"m4y_sim.log"):
        m=re.search(r"fused: yr_m=(-?\d+) fwd_m=(-?\d+)",L)
        if m: yr.append(int(m.group(1))/1000.0); fwd.append(int(m.group(2))/1000.0)
    yr=np.array(yr); fwd=np.array(fwd)
    # per-tick ZOH (each vision output holds VDIV ticks)
    yr_t=np.repeat(yr,VDIV)[:2600]; fwd_t=np.repeat(fwd,VDIV)[:2600]
    return yr_t,fwd_t

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def replay(mapping="rate", gate="none", A=None,B=None,Kinf=None, umin=UMIN0,umax=UMAX0,
           yr_max=0.8, tbg_floor=0.5, align_full=math.radians(15), align_floor=0.35, base_fwd=None):
    yr_t,fwd_t=load_real_trace(); N=len(yr_t)
    x=np.zeros(12); x[2]=2.5; x[5]=math.radians(99.0)
    px,py=-8.13,5.8; gi=0; passed=0; log=[]
    for k in range(N):
        yaw=x[5]; goal=GATES[min(gi,3)]
        if math.hypot(goal[0]-px,goal[1]-py)<PASS_R:
            if gi<3: gi+=1; passed=max(passed,gi)
            else: passed=4
            goal=GATES[min(gi,3)]
        # reconstructed desired_vel (body-frame goal guidance) + misalignment
        dx,dy=goal[0]-px,goal[1]-py; cy,sy=math.cos(-yaw),math.sin(-yaw)
        bx=cy*dx-sy*dy; by=sy*dx+cy*dy; misalign=math.atan2(by,bx)  # 0 when heading at gate
        yr_cmd = yr_t[k] if k>SETTLE else 0.0
        fwd_cmd= (base_fwd if base_fwd is not None else fwd_t[k]) if k>SETTLE else 0.0
        if fwd_cmd<0: fwd_cmd=0.0
        # forward gate
        if gate=="yr":
            a=max(0.0,1.0-abs(yr_cmd)/yr_max); fwd_cmd*= tbg_floor+(1-tbg_floor)*a*a
        elif gate=="align":
            a=max(0.0,1.0-abs(misalign)/align_full)  # 1 aligned, 0 when |misalign|>=align_full
            fwd_cmd*= align_floor+(1-align_floor)*a*a
        sp=np.zeros(12); sp[2]=2.5; sp[6]=fwd_cmd
        err=x-sp; err[0]=0; err[1]=0
        if mapping=="rate": sp[5]=x[5]; err[5]=0.0; err[11]=x[11]-yr_cmd
        else: e5=-yr_cmd*2.0; err[5]=max(-0.30,min(0.30,e5))
        u=np.clip(-Kinf@err,umin,umax)
        x=A@x+B@u
        v=x[6]; px+=v*math.cos(yaw)*DT; py+=v*math.sin(yaw)*DT
        udiff=u[0]-u[1]+u[2]-u[3]
        log.append((k*DT,math.degrees(yaw),px,py,x[2],x[11],yr_cmd,udiff,math.degrees(misalign),passed))
    return np.array(log)

def rpt(tag,L):
    t,yaw,px,py,z,wz,yr,udiff,mis,passed=L.T
    m=t>1.0
    trackerr=np.mean(np.abs(wz[m]-yr[m]))                      # yaw-rate tracking error vs real yr
    sat=100*np.mean((np.abs(udiff[m])>0.95*(UMAX0-UMIN0)/1))   # rough sat proxy
    xdrift=np.max(np.abs(px+8.0))
    print(f"{tag}: gates={int(passed.max())}  end=({px[-1]:+.1f},{py[-1]:+.1f})  max|x+8|={xdrift:.2f}m  "
          f"wz-vs-yr err={trackerr:.3f}  |udiff|>0.9max={sat:.0f}%")
    return L

if __name__=="__main__":
    A,B,Q,R,rho=load_hpp("quadrotor_yawfix_params.hpp"); K=dare(A,B,Q,R,rho)
    print("### STEP 0: replay REAL trace with the co-sim control (rate + |yr|-floor-tbg) — reproduce co-sim? ###")
    rpt("real/rate+yrtbg", replay("rate","yr",A,B,K))
    print("\n(compare to co-sim floor-tbg run: gate 1, yaw over-swings to ~60, max|x+8|~1.0)")
