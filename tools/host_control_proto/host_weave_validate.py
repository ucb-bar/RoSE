#!/usr/bin/env python3
"""STEP 2: host-validate the FULL yaw fix on a weave (gate-like) heading profile.

Plant = committed Adyn + the PHYSICALLY-CORRECT yaw mixing (Bdyn yaw rows signs (+,-,+,-) = the
real CRAZYFLIE spin) -- this is what the real Isaac MotorThrustAction plant does. Controller uses
the BODY-RELATIVE yaw command err[5] = -yr*GAIN (re-zeroed, no absolute-yaw ref) + setpoint[6]=fwd.
Weave profile: yr flips sign every 1.5 s (a sign-changing turn sequence) while cruising fwd=1.5.

  BEFORE: Kinf from strongyaw params (WRONG yaw mixing) on the real plant  -> expect broken.
  AFTER : Kinf from yawfix   params (CORRECT yaw mixing) on the real plant -> expect tracks weave.
Require (AFTER): yaw follows the commanded turns (yaw-rate sign == yr sign), altitude held, stable.
"""
import numpy as np, re
BASE="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/examples/problem_data/"
def load(f):
    t=open(BASE+f).read()
    b=lambda n,s: np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",re.search(n+r"\[[^\]]*\]\s*=\s*\{([^}]*)\}",t).group(1))]).reshape(s)
    return b("Adyn_data",(12,12)), b("Bdyn_data",(12,4)), b("Kinf_data",(4,12))
A,B_strong,K_strong = load("quadrotor_strongyaw_params.hpp")
_,B_fix,K_fix        = load("quadrotor_yawfix_params.hpp")
B_plant = B_fix   # the physically-correct yaw mixing == what the real plant does
CTRL_DT=0.005; UMIN,UMAX=-0.583,0.417; GAIN=2.0

def weave_yr(t):   # sign-changing turn command (gate weave), rad/s
    return 0.3*(1.0 if (int(t/1.5)%2==0) else -1.0)

def run(K, fwd=1.5, secs=9.0, yaw0=0.0):
    x=np.zeros(12); x[2]=2.5; x[5]=yaw0; log=[]
    for k in range(int(secs/CTRL_DT)):
        t=k*CTRL_DT; yr=weave_yr(t)
        sp=np.zeros(12); sp[2]=2.5; sp[6]=fwd
        err=x-sp; err[0]=0; err[1]=0
        err[5]=-yr*GAIN                      # body-relative yaw command
        u=np.clip(-K@err,UMIN,UMAX)
        x=A@x+B_plant@u
        log.append((t,yr,x[5],x[11],x[2]))   # t, yr_cmd, yaw, wz, z
    return np.array(log)

def summarize(tag,L):
    t,yrc,yaw,wz,z=L.T
    # does yaw-rate follow the command sign? (fraction of samples where sign(wz)==sign(yr), after transient)
    m=t>0.3; agree=np.mean(np.sign(wz[m])==np.sign(yrc[m]))
    zrng=z.max()-z.min(); yawrng=np.degrees(yaw.max()-yaw.min())
    stable = np.all(np.abs(yaw)<10) and zrng<1.0 and np.all(np.isfinite(z))
    print(f"{tag}: wz-follows-cmd={agree*100:.0f}%  yaw swing={yawrng:.0f}deg  z-range={zrng:.3f}m  stable={stable}")
    for i in range(0,len(t),int(1.5/CTRL_DT)):
        print(f"    t={t[i]:.1f} yr_cmd={yrc[i]:+.2f} yaw={np.degrees(yaw[i]):+7.1f} wz={wz[i]:+.3f} z={z[i]:.3f}")

print("=== BEFORE: strongyaw Kinf (WRONG mixing) on real plant ===")
summarize("BEFORE", run(K_strong))
print("\n=== AFTER: yawfix Kinf (CORRECT mixing) on real plant ===")
summarize("AFTER ", run(K_fix))
