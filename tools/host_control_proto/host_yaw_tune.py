#!/usr/bin/env python3
"""STEP B: fast host closed-loop yaw-loop tuning (yawfix params, GT state, gate course).

Reproduces the two co-sim failure modes and tunes them out BEFORE paying co-sim cost:
  (i)  bang-bang yaw saturation  — clamped body-relative err[5]=-yr*GAIN rails, yaw oscillates.
  (ii) slow-yaw-vs-(-x)-drift    — from the off-line 99deg spawn the drone drifts off the gate
                                    line faster than yaw converges, so it overshoots.

Plant+model = committed yawfix (Adyn, Bdyn_yawfix) — the corrected yaw mixing; GT state. A guidance
law mimics the vision policy: yr_cmd = Kp*heading_error to the current gate (a rate command the
policy would emit). Forward velocity is commanded in the HEADING direction, so YAW steers the path
(true yaw-nav, not strafe). Gates = FUSED course; spawn off-line at yaw0=99deg like seed 1000.

Two control mappings x {plain, turn-before-go fwd gating}:
  - body-relative: err[5] = clamp(-yr_cmd*GAIN, +/-YE)         (reproduces bang-bang)
  - rate:          setpoint[11] = yr_cmd (smooth)              (Stage-1-matching)
Turn-before-go: fwd_eff = fwd * max(0,cos(heading_err))^p  -> slow forward while mis-aligned.
"""
import numpy as np, re, math
BASE="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/examples/problem_data/"
t=open(BASE+"quadrotor_yawfix_params.hpp").read()
b=lambda n,s: np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",re.search(n+r"\[[^\]]*\]\s*=\s*\{([^}]*)\}",t).group(1))]).reshape(s)
A=b("Adyn_data",(12,12)); Bp=b("Bdyn_data",(12,4)); Kinf=b("Kinf_data",(4,12))
DT=0.01; UMIN,UMAX=-0.583,0.417
GATES=np.array([[-8.05,9.0],[-8.30,13.0],[-7.75,17.0],[-8.05,21.0]]); PASS_R=1.2
SPAWN=np.array([-8.13,5.8]); YAW0=math.radians(99.0)

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def run(mapping, gain=2.0, ye=0.30, kp=1.5, yr_max=0.8, base_fwd=2.0, tbg_p=0.0, secs=16.0,
        vision_div=20, yr_noise=0.15, spawn_off=(-1.4,0.0), yaw0=math.radians(130.0), seed=0):
    rng=np.random.RandomState(seed)
    x=np.zeros(12); x[2]=2.5; x[5]=yaw0
    px,py=SPAWN[0]+spawn_off[0],SPAWN[1]+spawn_off[1]              # HARDER: lateral + heading off-line
    gi=0; log=[]; passed=0; yr_cmd=0.0; fwd=0.0
    for k in range(int(secs/DT)):
        yaw=x[5]
        goal=GATES[min(gi,3)]
        if math.hypot(goal[0]-px,goal[1]-py)<PASS_R:
            if gi<3: gi+=1; passed=max(passed,gi)
            else: passed=4
            goal=GATES[min(gi,3)]
        # policy runs at 10 Hz (every vision_div ticks), ZOH between, with output noise
        if k % vision_div == 0:
            bearing=math.atan2(goal[1]-py, goal[0]-px); herr=wrap(bearing-yaw)
            yr_cmd=float(np.clip(kp*herr + rng.randn()*yr_noise, -yr_max, yr_max))
            # turn-before-go: GUEST-IMPLEMENTABLE gating by the model's OWN |yr| (large |yr| = the
            # policy wants a big turn = mis-aligned -> slow forward). No bearing-to-gate needed.
            if tbg_p>0:
                fwd=base_fwd*max(0.0, 1.0-abs(yr_cmd)/yr_max)**tbg_p
            else:
                fwd=base_fwd
        else:
            bearing=math.atan2(goal[1]-py, goal[0]-px); herr=wrap(bearing-yaw)
        sp=np.zeros(12); sp[2]=2.5; sp[6]=fwd
        err=x-sp; err[0]=0; err[1]=0
        if mapping=="bodyrel":
            e5=-yr_cmd*gain; err[5]=max(-ye,min(ye,e5))
        else:  # rate
            sp[5]=x[5]; sp[11]=yr_cmd; err[5]=0.0; err[11]=x[11]-yr_cmd
        u=np.clip(-Kinf@err,UMIN,UMAX)
        x=A@x+Bp@u
        # advance world position by body-forward speed (state[6]=body vx) along heading
        v=x[6]
        px+=v*math.cos(yaw)*DT; py+=v*math.sin(yaw)*DT
        udiff=(u[0]-u[1]+u[2]-u[3])
        log.append((k*DT,math.degrees(yaw),px,py,x[2],udiff,math.degrees(herr),passed))
    return np.array(log)

def summarize(tag,L):
    t,yaw,px,py,z,udiff,herr,passed=L.T
    m=t>0.5
    # cross-track: min distance to the gate-line x=-8 (approx) over the run
    xdrift=np.abs(px+8.0).max()
    # yaw oscillation: sign changes in d(yaw)
    dy=np.diff(yaw); osc=int(np.sum((dy[:-1]*dy[1:]<0)&(np.abs(dy[:-1])>0.05)))
    sat=100*np.mean(np.abs(udiff[m])>0.9)
    print(f"{tag}: gates={int(passed.max())}  max|x+8|={xdrift:.2f}m  yaw-dir-flips={osc}  u_yawdiff|sat|={sat:.0f}%  z-range={z.max()-z.min():.3f}")
    for i in range(0,len(t),int(2.0/DT)):
        print(f"    t={t[i]:4.1f} yaw={yaw[i]:6.1f} herr={herr[i]:+6.1f} pos=({px[i]:+.2f},{py[i]:+.2f}) gate={int(passed[i])}")

print("################ (i) BANG-BANG: body-relative, GAIN=2.0, clamp 0.30, no turn-before-go ############")
summarize("bodyrel", run("bodyrel", gain=2.0, ye=0.30, tbg_p=0.0))
print("\n################ (ii) rate command, no turn-before-go (slow-yaw vs drift) ############")
summarize("rate", run("rate", tbg_p=0.0))
print("\n################ TUNED: rate command + turn-before-go (fwd*cos^2 herr) ############")
summarize("rate+tbg", run("rate", tbg_p=2.0))
print("\n################ TUNED v2: rate + stronger turn-before-go (cos^3) + higher kp ############")
summarize("rate+tbg3", run("rate", kp=2.2, tbg_p=3.0))
