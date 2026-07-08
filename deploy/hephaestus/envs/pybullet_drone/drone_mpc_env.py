"""RoSE gym environment for the drone_control TinyMPC co-sim loop.

This is the sim-side counterpart of the `samples/rose/drone_control` Zephyr guest
(a RoSE port of zephyr-chipyard-sw's `samples/drone_control` HIL controller). Unlike
PyBulletDroneEnv (which serves an IMU abstraction), this serves the *full linearized
quadrotor state* the TinyMPC controller expects, exactly as the physical-UART HIL
host (`scripts/pybullet_hil.py`) does:

    mpc_state = [x, y, z, r1, r2, r3, vx, vy, vz, dphi, dtheta, dpsi]   (12 x float32)

where (r1,r2,r3) are Rodrigues parameters from the body quaternion. The state is
served TARGET-RELATIVE (position offset by a hover setpoint) so the controller
regulates the drone to that setpoint. The SoC returns 4 normalized motor thrusts
(around the 0.583 hover point), which are converted to RPMs and applied — matching
`calculate_rpm()` in the HIL host.

    pybullet dynamics --full state--> RoSE bridge --> Zephyr TinyMPC controller
          ^                                                     |
          +------------ 4 normalized thrusts <-- RoSE bridge <--+

Loop wiring lives in deploy/config/config_gym_PyBulletDroneMPCEnv-v0.yaml
(reqrsp binding serves `mpc_state`; an action-latch binding applies `thrust`).
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_ROSE = os.environ.get("ROSE_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_GPD = os.path.join(_ROSE, "soc/sw/xpu-rt/zephyr-chipyard-sw/tools/gym-pybullet-drones")
if _GPD not in sys.path:
    sys.path.insert(0, _GPD)

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402

# TinyMPC hover setpoint (matches pybullet_hil.py: origin, 0.3 m above start-ish).
DEFAULT_TARGET = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Motor mixing constants (from pybullet_hil.py calculate_rpm / TinyMPC problem data).
_HOVER_THRUST = 0.583          # normalized thrust at hover (per motor)
_MAX_THRUST_N = 0.58 / 4.0     # N per motor at full normalized command


def _quat_to_rodrigues(q):
    """[qx,qy,qz,qw] -> Rodrigues params (r1,r2,r3) = q_xyz / qw (as in the HIL host)."""
    qx, qy, qz, qw = q
    if abs(qw) < 1e-9:
        qw = 1e-9 if qw >= 0 else -1e-9
    return np.array([qx / qw, qy / qw, qz / qw], dtype=np.float64)


class PyBulletDroneMPCEnv(gym.Env):
    """CtrlAviary exposed as the TinyMPC full-state HIL loop over the RoSE bridge."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, ctrl_freq=50, pyb_freq=200, target=None, **kwargs):
        # ctrl_freq matches the TinyMPC problem data (quadrotor_50hz_params_*).
        self.ctrl_freq = ctrl_freq
        self.aviary = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=False,
        )
        self.KF = float(self.aviary.KF)
        self.hover_rpm = float(self.aviary.HOVER_RPM)
        self.max_rpm = float(self.aviary.MAX_RPM)
        self.target = np.asarray(target, dtype=np.float64) if target is not None else DEFAULT_TARGET.copy()

        f32 = np.float32
        big = np.finfo(f32).max
        self.observation_space = spaces.Dict({
            # target-relative TinyMPC state the controller consumes
            "mpc_state": spaces.Box(-big, big, (12,), f32),
            # absolute pose for logging / plotting
            "pos":  spaces.Box(-big, big, (3,), f32),
            "quat": spaces.Box(-1.0, 1.0, (4,), f32),
        })
        # 4 normalized motor thrusts (controller output, ~[-0.583, 0.417])
        self.action_space = spaces.Box(-1.0, 1.0, (4,), f32)

    def _mpc_state(self, raw):
        s = np.asarray(raw[0], dtype=np.float64)  # CtrlAviary drone-0 state (20,)
        pos = s[0:3] - self.target                # regulate to the setpoint
        r = _quat_to_rodrigues(s[3:7])
        vel = s[10:13]
        angv = s[13:16]
        return np.concatenate([pos, r, vel, angv]).astype(np.float32)

    def _obs(self, raw):
        s = np.asarray(raw[0], dtype=np.float64)
        return {
            "mpc_state": self._mpc_state(raw),
            "pos": s[0:3].astype(np.float32),
            "quat": s[3:7].astype(np.float32),
        }

    def _thrusts_to_rpm(self, u):
        """Normalized TinyMPC thrusts -> per-motor RPM (matches HIL host)."""
        thrust_n = (np.asarray(u, dtype=np.float64) + _HOVER_THRUST) * _MAX_THRUST_N
        thrust_n = np.clip(thrust_n, 0.0, None)
        return np.sqrt(thrust_n / self.KF).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw, info = self.aviary.reset(seed=seed)
        return self._obs(raw), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < 4 or not np.any(a):
            rpm = np.full(4, self.hover_rpm, dtype=np.float32)   # hover until first command
        else:
            rpm = self._thrusts_to_rpm(a[:4])
        raw, rew, term, trunc, info = self.aviary.step(rpm.reshape(1, 4))
        obs = self._obs(raw)
        if os.environ.get("ROSE_ENV_DEBUG"):
            self._dbg = getattr(self, "_dbg", 0) + 1
            if self._dbg % 25 == 1:
                p = obs["pos"]
                print(f"[drone] step {self._dbg:5d} pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"z_err={p[2]-self.target[2]:+.2f} thrust={np.round(a[:4],3)}", flush=True)
        return obs, float(rew), bool(term), bool(trunc), info

    def render(self):
        return None

    def close(self):
        try:
            self.aviary.close()
        except Exception:
            pass
