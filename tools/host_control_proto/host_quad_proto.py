#!/usr/bin/env python3
"""FAST host-only quadrotor + TinyMPC loop to prototype yaw control (no spike, no bridge).

Reproduces the on-SoC control problem in pure numpy, ms per run, GROUND-TRUTH state (isolates the
CONTROL problem from the estimator). Plant = the EXACT MotorThrustAction / crazyflie_mpc physics:
  per-rotor force f_i = (u_i + HOVER)*MAX_THRUST_N (clamped >=0), applied along body +z at the rotor
  arm positions -> collective thrust + roll/pitch moments from the arm geometry; yaw = propeller-drag
  reaction sum(spin_i * (km/kf) * f_i). Rigid-body 6-DOF integrate at 200 Hz.

Controller = host TinyMPC: u = clamp(-Kinf @ (x - setpoint), umin, umax)  (LQR first-step = the
constrained MPC's unconstrained optimum; faithful for the yaw-authority question). Kinf/Adyn/Bdyn
loaded from the committed problem_data .hpp (same constants the guest uses).

Test: command the policy interface (yaw_rate, forward_speed) and require the drone to TURN + hold a
heading while cruising. Logs yaw(t), z(t), and the u rotor pattern.
"""
import numpy as np, re, sys, argparse

# ---- physical plant constants (from mdp_motor_thrust_action.py / crazyflie_mpc_env.py, runtime) ----
MASS   = 0.0282                      # kg (runtime print)
MAXT   = 0.58/4.0                    # N per rotor at full normalized command
HOVER  = 0.583                       # normalized hover thrust per rotor
KM_KF  = 7.94e-12/3.16e-10           # 0.0251 m, propeller drag/thrust ratio
SPIN   = np.array([+1.,-1.,+1.,-1.]) # per-rotor spin sign (CRAZYFLIE_CFG joint_vel order)
ARM    = 0.0397                      # gym-pybullet-drones CF2X arm length (m)
# CF2X X-config rotor positions (body frame), matching gym-pybullet-drones CF2X ordering.
# props at 45deg; m1..m4 around the frame. (x fwd, y left, z up)
_a = ARM/np.sqrt(2.0)
ROTOR_POS = np.array([[ _a,-_a,0],[ _a, _a,0],[-_a, _a,0],[-_a,-_a,0]])  # will validate vs Bdyn
# CF2X inertia (gym-pybullet-drones)
J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
Jinv = np.linalg.inv(J)
G = 9.81
CTRL_DT = 0.005                      # 200 Hz control
SUB = 4                              # physics substeps per control tick
DT  = CTRL_DT/SUB

def load_params(hpp):
    t=open(hpp).read()
    def blk(n,s):
        m=re.search(n+r"\[[^\]]*\]\s*=\s*\{([^}]*)\}",t)
        return np.array([float(x) for x in re.findall(r"[-+]?\d[\d.eE+-]*",m.group(1))]).reshape(s)
    return dict(A=blk("Adyn_data",(12,12)), B=blk("Bdyn_data",(12,4)),
               Kinf=blk("Kinf_data",(4,12)))

# ---------- NONLINEAR plant (exact MotorThrustAction wrench) ----------
def quat_mul(q,r):
    w0,x0,y0,z0=q; w1,x1,y1,z1=r
    return np.array([w0*w1-x0*x1-y0*y1-z0*z1, w0*x1+x0*w1+y0*z1-z0*y1,
                     w0*y1-x0*z1+y0*w1+z0*x1, w0*z1+x0*y1-y0*x1+z0*w1])
def R_from_q(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def yaw_of_q(q):
    w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

class NLPlant:
    """6-DOF rigid body driven by the exact per-rotor wrench."""
    def __init__(self, yaw0=0.0, z0=2.5):
        self.p=np.array([0,0,z0],float); self.v=np.zeros(3)
        self.q=np.array([np.cos(yaw0/2),0,0,np.sin(yaw0/2)]); self.w=np.zeros(3)  # body rates
    def step(self, u):
        f = np.clip((u+HOVER)*MAXT, 0, None)             # per-rotor force (N), body +z
        for _ in range(SUB):
            Fz=f.sum()
            # body torque: arm moments (r x F, F=[0,0,f_i]) + yaw drag reaction
            tau=np.zeros(3)
            for i in range(4):
                r=ROTOR_POS[i]; tau[0]+= r[1]*f[i]; tau[1]+= -r[0]*f[i]
                tau[2]+= SPIN[i]*KM_KF*f[i]
            Rm=R_from_q(self.q)
            acc=(Rm@np.array([0,0,Fz]))/MASS - np.array([0,0,G])
            wdot=Jinv@(tau - np.cross(self.w,J@self.w))
            self.v+=acc*DT; self.p+=self.v*DT
            self.w+=wdot*DT
            wq=np.array([0,*self.w]); self.q=self.q+0.5*quat_mul(self.q,wq)*DT
            self.q/=np.linalg.norm(self.q)
    def state12(self):
        # [x,y,z, r,p,yaw(rodrigues xyz/w), vx,vy,vz(body? world), wx,wy,wz]
        w,x,y,z=self.q; wv=w if abs(w)>1e-9 else 1e-9
        return np.array([self.p[0],self.p[1],self.p[2], x/wv,y/wv,z/wv,
                         self.v[0],self.v[1],self.v[2], self.w[0],self.w[1],self.w[2]])

# ---------- LINEAR plant (committed Adyn/Bdyn = the MPC's own model) ----------
class LinPlant:
    def __init__(self, A,B, yaw0=0.0, z0=2.5):
        self.A=A; self.B=B; self.x=np.zeros(12); self.x[2]=z0; self.x[5]=np.tan(yaw0/2)*2 if False else yaw0
        # rodrigues yaw approx small; for the linear model state[5] is the small-angle yaw
        self.x[5]=yaw0
    def step(self,u):
        self.x=self.A@self.x+self.B@u
    def state12(self): return self.x.copy()

def yaw_from_state(x, nonlinear):
    if nonlinear:  # rodrigues -> yaw (small approx: 2*atan(r_z)); use full via rebuilding quat
        rz=x[5]; return 2*np.arctan(rz)   # yaw ~ 2*atan(qz/qw)
    return x[5]

def run(mode, yr_cmd, fwd_cmd, yaw0, mapping, gain, umin, umax, P, secs=6.0, nonlinear=True):
    Kinf=P["Kinf"]
    plant = NLPlant(yaw0=yaw0) if nonlinear else LinPlant(P["A"],P["B"],yaw0=yaw0)
    N=int(secs/CTRL_DT); log=[]
    for k in range(N):
        x=plant.state12()
        sp=np.zeros(12); sp[2]=2.5; sp[6]=fwd_cmd
        err=x-sp; err[0]=0; err[1]=0
        if mapping=="rate":
            sp[11]=yr_cmd; err[11]=x[11]-yr_cmd; err[5]=0.0   # neutralize angle, faithful rate
        elif mapping=="relangle":
            err[5]=-yr_cmd*gain                                # body-relative bounded yaw offset
        elif mapping=="absangle":
            sp[5]=yaw0+yr_cmd*(k*CTRL_DT); err[5]=x[5]-sp[5]   # integrated absolute heading
        u=np.clip(-Kinf@err, umin, umax)
        plant.step(u)
        yaw=yaw_of_q(plant.q) if nonlinear else x[5]
        log.append((k*CTRL_DT, yaw, x[2], *u))
    return np.array(log)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--params", default="/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/samples/drone_control/tinympc/examples/problem_data/quadrotor_strongyaw_params.hpp")
    ap.add_argument("--mapping", default="rate", choices=["rate","relangle","absangle"])
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--yr", type=float, default=-0.3); ap.add_argument("--fwd", type=float, default=1.5)
    ap.add_argument("--yaw0", type=float, default=1.727); ap.add_argument("--linear", action="store_true")
    a=ap.parse_args()
    P=load_params(a.params)
    # validate the nonlinear plant linearization vs committed Bdyn (faithfulness check)
    print(f"params={a.params.split('/')[-1]}  mapping={a.mapping} gain={a.gain} yr={a.yr} fwd={a.fwd} yaw0={a.yaw0:.3f}rad ({np.degrees(a.yaw0):.0f}deg) plant={'LINEAR(Adyn/Bdyn)' if a.linear else 'NONLINEAR'}")
    L=run(None, a.yr, a.fwd, a.yaw0, a.mapping, a.gain, -0.583, 0.417, P, nonlinear=not a.linear)
    t=L[:,0]; yaw=np.degrees(L[:,1]); z=L[:,2]; u=L[:,3:7]
    for i in range(0,len(t),int(0.5/CTRL_DT)):
        print(f" t={t[i]:.1f} yaw={yaw[i]:+7.1f}deg z={z[i]:.3f} u=[{u[i,0]:+.2f} {u[i,1]:+.2f} {u[i,2]:+.2f} {u[i,3]:+.2f}]")
    print(f" -> yaw moved {yaw[-1]-yaw[0]:+.1f} deg over {t[-1]:.1f}s (cmd yr={a.yr} rad/s => ideal {np.degrees(a.yr*t[-1]):+.0f} deg)")
