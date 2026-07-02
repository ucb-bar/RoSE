"""RoSE gym environment wrapping gym-pybullet-drones (CtrlAviary).

Serves a single quadrotor's IMU-relevant state (body-frame accelerometer +
gyroscope, plus full kinematic state) as observation-dict fields the RoSE
synchronizer forwards over the bridge, and consumes 4 motor RPMs from the SoC as
the action. This is the physics backend for the RoSE drone co-sim loop:

    pybullet dynamics --IMU--> RoSE bridge --sensor.h--> Zephyr controller
          ^                                                     |
          +------------- motor RPMs <-- RoSE bridge <-----------+

gym-pybullet-drones comes from the zephyr-chipyard-sw submodule so the physics
matches the Zephyr-side drone samples. We add it to sys.path (rather than
pip-installing) since its deps (pybullet, numpy, gymnasium, scipy) are already in
the venv.
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- make gym_pybullet_drones importable from the zephyr-chipyard-sw submodule ---
_ROSE = os.environ.get("ROSE_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_GPD = os.path.join(_ROSE, "soc/sw/xpu-rt/zephyr-chipyard-sw/tools/gym-pybullet-drones")
if _GPD not in sys.path:
    sys.path.insert(0, _GPD)

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402

try:
    from scipy.spatial.transform import Rotation
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

_GRAVITY = 9.8  # m/s^2


class PyBulletDroneEnv(gym.Env):
    """Single-drone CtrlAviary exposed with IMU-friendly observation fields."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, ctrl_freq=240, pyb_freq=240, **kwargs):
        self.ctrl_freq = ctrl_freq
        self.dt = 1.0 / float(ctrl_freq)
        self.aviary = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=False,
        )
        self.hover_rpm = float(self.aviary.HOVER_RPM)
        self.max_rpm = float(self.aviary.MAX_RPM)
        self._prev_vel = np.zeros(3, dtype=np.float64)

        f32 = np.float32
        big = np.finfo(f32).max
        self.observation_space = spaces.Dict({
            "imu_accel": spaces.Box(-big, big, (3,), f32),   # body-frame specific force (m/s^2)
            "imu_gyro":  spaces.Box(-big, big, (3,), f32),   # body-frame angular velocity (rad/s)
            "state":     spaces.Box(-big, big, (20,), f32),  # full CtrlAviary state
            "pos":       spaces.Box(-big, big, (3,), f32),
            "quat":      spaces.Box(-1.0, 1.0, (4,), f32),
        })
        # 4 motor RPMs for the single drone
        self.action_space = spaces.Box(0.0, self.max_rpm, (4,), f32)

    def _imu(self, state, vel):
        """Body-frame accelerometer (specific force) + gyro from the drone state."""
        ang_v_world = np.asarray(state[13:16], dtype=np.float64)
        accel_world = (np.asarray(vel, dtype=np.float64) - self._prev_vel) / self.dt
        # accelerometer measures specific force a - g (g points down): a_world + [0,0,g]
        specific_force_world = accel_world + np.array([0.0, 0.0, _GRAVITY])
        if _HAVE_SCIPY:
            rot = Rotation.from_quat(np.asarray(state[3:7], dtype=np.float64))  # [x,y,z,w]
            gyro = rot.inv().apply(ang_v_world)
            accel = rot.inv().apply(specific_force_world)
        else:
            gyro = ang_v_world
            accel = specific_force_world
        return accel.astype(np.float32), gyro.astype(np.float32)

    def _obs(self, raw):
        state = np.asarray(raw[0], dtype=np.float64)  # drone 0, shape (20,)
        vel = state[10:13]
        accel, gyro = self._imu(state, vel)
        self._prev_vel = np.array(vel, dtype=np.float64)
        return {
            "imu_accel": accel,
            "imu_gyro": gyro,
            "state": state.astype(np.float32),
            "pos": state[0:3].astype(np.float32),
            "quat": state[3:7].astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw, info = self.aviary.reset(seed=seed)
        self._prev_vel = np.array(np.asarray(raw[0])[10:13], dtype=np.float64)
        return self._obs(raw), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        # default to hover if the SoC has not (yet) issued a valid motor command
        if a.size < 4 or not np.any(a):
            a = np.full(4, self.hover_rpm, dtype=np.float32)
        raw, rew, term, trunc, info = self.aviary.step(a[:4].reshape(1, 4))
        return self._obs(raw), float(rew), bool(term), bool(trunc), info

    def render(self):
        return None

    def close(self):
        try:
            self.aviary.close()
        except Exception:
            pass
