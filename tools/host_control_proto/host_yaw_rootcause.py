#!/usr/bin/env python3
"""Host-loop demonstration of the on-SoC yaw root cause + fix (pure numpy, ms).

Root cause hypothesis (from Bdyn analysis): the committed TinyMPC quadrotor model's yaw-rate
mixing (Bdyn row 11 signs [+,-,-,+], a FRONT/BACK grouping) does NOT match the real plant's
propeller-drag yaw (spin sign [+,-,+,-], a DIAGONAL grouping) -- rotors 2 & 3 are yaw-sign-
flipped between model and plant. So the MPC's yaw command produces the wrong/near-zero net yaw
on the real plant.

We isolate CONTROL from the estimator (ground-truth state) and use the plant = committed
Adyn/Bdyn (the MPC's own linearization of the real Crazyflie) but with the yaw-actuation row
replaced by the REAL diagonal-drag mixing. Three cases:
  (A) MPC(Kinf_model) on plant_model      -> self-consistent baseline (should yaw fine)
  (B) MPC(Kinf_model) on plant_REAL_yaw   -> model/plant yaw-mixing MISMATCH (reproduces co-sim: no yaw)
  (C) MPC(Kinf_FIXED)  on plant_REAL_yaw  -> Kinf re-derived for the REAL mixing (FIX: yaw works)
"""
import numpy as np, re

HPP="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/examples/problem_data/quadrotor_strongyaw_params.hpp"
def blk(t,n,s): return np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",re.search(n+r"\[[^\]]*\]\s*=\s*\{([^}]*)\}",t).group(1))]).reshape(s)
T=open(HPP).read()
A=blk(T,"Adyn_data",(12,12)); B=blk(T,"Bdyn_data",(12,4)); Kinf=blk(T,"Kinf_data",(4,12))
Qv=blk(T,"Q_data",(12,)); Rv=blk(T,"R_data",(4,)); rho=float(re.search(r"rho_value\s*=\s*([-\d.eE+]+)",T).group(1))
CTRL_DT=0.005; UMIN,UMAX=-0.583,0.417; SPIN=np.array([1.,-1.,1.,-1.])

def dare_K(A,B,Qv,Rv,rho):
    Q=np.diag(Qv)+rho*np.eye(12); R=np.diag(Rv)+rho*np.eye(4); P=np.diag(Qv).copy()
    for _ in range(20000):
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A); Pn=np.diag(Qv)+rho*np.eye(12)+A.T@P@(A-B@K)
        if np.max(np.abs(Pn-P))<1e-12: P=Pn;break
        P=Pn
    return K

# real plant B: replace the yaw-rate row (11) actuation with the diagonal propeller-drag mixing
# (signs = spin), same magnitude scale as the model's yaw row so it's an apples-to-apples swap.
B_real=B.copy()
# faithful mismatch: keep the model's per-rotor yaw magnitudes but flip the two rotors whose
# spin sign disagrees with the model (2,3): model [+,-,-,+] vs real spin [+,-,+,-].
B_real[11]=B[11]*np.array([1.,1.,-1.,-1.])
print(f"model Bdyn[11] (yaw-rate mixing): {np.round(B[11],4)}  signs {np.sign(B[11]).astype(int)}")
print(f"REAL  plant[11] (spin drag)     : {np.round(B_real[11],4)}  signs {np.sign(B_real[11]).astype(int)}")
print(f"  -> rotors 2,3 yaw sign FLIPPED between model and plant\n")
Kfix=dare_K(A,B_real,Qv,Rv,rho)

def run(Aplant,Bplant,K,yr=-0.3,fwd=1.5,yaw0=1.727,secs=6.0):
    x=np.zeros(12); x[2]=2.5; x[5]=yaw0; log=[]
    N=int(secs/CTRL_DT)
    for k in range(N):
        sp=np.zeros(12); sp[2]=2.5; sp[6]=fwd; sp[11]=yr
        err=x-sp; err[0]=0; err[1]=0; err[5]=0   # faithful RATE command (neutralize angle)
        u=np.clip(-K@err,UMIN,UMAX)
        x=Aplant@x+Bplant@u
        log.append((k*CTRL_DT,x[5],x[2],x[11]))
    return np.array(log)

for tag,(Ap,Bp,K) in {
    "(A) Kinf_model on plant_model      (self-consistent)":(A,B,Kinf),
    "(B) Kinf_model on plant_REAL_yaw   (MISMATCH=co-sim) ":(A,B_real,Kinf),
    "(C) Kinf_FIXED on plant_REAL_yaw   (FIX)             ":(A,B_real,Kfix),
}.items():
    L=run(Ap,Bp,K)
    yaw=np.degrees(L[:,1]); print(f"{tag}: yaw {yaw[0]:+.0f}->{yaw[-1]:+.0f} deg (moved {yaw[-1]-yaw[0]:+.0f}; ideal -103), z_end={L[-1,2]:.2f}, wz_end={L[-1,3]:+.3f}")
