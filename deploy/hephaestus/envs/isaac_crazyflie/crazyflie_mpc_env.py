"""RoSE gym environment for the drone_control TinyMPC co-sim loop, on IsaacLab.

This is the **IsaacLab (Isaac Sim) counterpart** of `PyBulletDroneMPCEnv`: it flies a
Bitcraze **Crazyflie** (`cf2x`) inside NVIDIA Isaac Sim and presents the *exact same*
RoSE-facing contract, so the unchanged Zephyr guest (`samples/rose/drone_control`) and
the unchanged loop wiring drive either simulator.

It serves the full linearized quadrotor state the TinyMPC controller expects:

    mpc_state = [x, y, z, r1, r2, r3, vx, vy, vz, dphi, dtheta, dpsi]   (12 x float32)

where (r1,r2,r3) are Rodrigues parameters from the body quaternion. The state is served
TARGET-RELATIVE (position offset by a hover setpoint) so the controller regulates the
drone to that setpoint. The SoC returns **4 normalized motor thrusts** (around the 0.583
hover point) — matching TinyMPC's per-motor output — which are applied as per-rotor
z-forces at the 4 propeller bodies (`m.*_prop`), exactly like IsaacLab's own Crazyflie
demo (`scripts/demos/quadcopter_fpv.py`). Per-rotor forces reproduce collective thrust
and roll/pitch moments naturally from the arm geometry; the propeller drag reaction
(yaw) is added as a per-rotor z-torque with alternating spin sign.

    Isaac Sim dynamics --full state--> RoSE bridge --> Zephyr TinyMPC controller
          ^                                                     |
          +-------- 4 normalized thrusts (per rotor) <-- RoSE bridge <--+

Logging vs. the wire (see envs README "Logging"): `obs` carries only what the SoC needs;
the **ground truth** (absolute world-frame pose/velocity, per-rotor forces, target, and
sim time) is returned in the gym `info` dict, which is never serialized to the SoC. When
`traj_csv` (or $ROSE_TRAJ_CSV) is set the env also records that ground truth to CSV for
offline trajectory plotting. When `camera` (or $ROSE_ISAAC_CAMERA=1) is set the env adds
an offscreen third-person camera and `render()` returns its RGB frame (for video).

Unlike PyBullet, IsaacLab must boot a full Isaac Sim kit app; that happens lazily in
`__init__` (headless by default) so the synchronizer stays dependency-light unless this
env is actually `gym.make()`d. Requires the `env_isaaclab` toolchain (Isaac Sim +
isaaclab + isaaclab_assets), i.e. the same IsaacLab version pinned at
`soc/sw/xpu-rt/sims/IsaacLab`.
"""

import csv
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# TinyMPC hover setpoint (matches PyBulletDroneMPCEnv / pybullet_hil.py).
DEFAULT_TARGET = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Motor mixing constants — kept identical to PyBulletDroneMPCEnv so the *same* TinyMPC
# controller (tuned against gym-pybullet-drones CF2X) maps thrusts to physics the same way.
_HOVER_THRUST = 0.583          # normalized thrust at hover (per motor)
_MAX_THRUST_N = 0.58 / 4.0     # N per motor at full normalized command
# Propeller drag/thrust ratio (gym-pybullet-drones CF2X: KM=7.94e-12, KF=3.16e-10) used
# for the yaw reaction torque. Spin signs follow CRAZYFLIE_CFG init joint_vel
# (m1:+, m2:-, m3:+, m4:-).
_KM_OVER_KF = 7.94e-12 / 3.16e-10
_SPIN_SIGN = np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float64)

# Ground-truth CSV columns (see _traj_row).
_TRAJ_COLS = (
    ["t"]
    + ["x", "y", "z"]
    + ["qw", "qx", "qy", "qz"]
    + ["vx", "vy", "vz"]
    + ["wx", "wy", "wz"]
    + ["f0", "f1", "f2", "f3"]        # per-rotor force applied (N)
    + ["u0", "u1", "u2", "u3"]        # normalized thrust command from SoC
    + ["tx", "ty", "tz"]              # target
)


def _quat_to_rodrigues(qw, qx, qy, qz):
    """(w,x,y,z) quaternion -> Rodrigues params (r1,r2,r3) = q_xyz / qw (as in the HIL host)."""
    if abs(qw) < 1e-9:
        qw = 1e-9 if qw >= 0 else -1e-9
    return np.array([qx / qw, qy / qw, qz / qw], dtype=np.float64)


class IsaacCrazyflieMPCEnv(gym.Env):
    """IsaacLab Crazyflie exposed as the TinyMPC full-state HIL loop over the RoSE bridge."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, ctrl_freq=50, phys_freq=100, target=None,
                 device=None, headless=True, start_height=0.5, log_dir=None,
                 traj_csv=None, camera=None, camera_res=(640, 480),
                 camera_eye=(2.0, 2.0, 1.6), camera_look=(0.0, 0.0, 1.0),
                 camera_follow=True, camera_offset=(0.45, 0.45, 0.18),
                 camera_focal=35.0, frame_dir=None, **kwargs):
        # ctrl_freq matches the TinyMPC problem data (quadrotor_50hz_params_*); physics
        # runs faster and is sub-stepped (decimation) per control step.
        self.ctrl_freq = int(ctrl_freq)
        self.phys_freq = int(phys_freq)
        self.decimation = max(1, self.phys_freq // self.ctrl_freq)
        self.target = np.asarray(target, dtype=np.float64) if target is not None else DEFAULT_TARGET.copy()
        self.start_height = float(start_height)
        self._t = 0.0

        # Ground-truth trajectory CSV (offline plotting); env var lets a run enable it
        # without editing configs.
        self._traj_csv_path = traj_csv or os.environ.get("ROSE_TRAJ_CSV")
        self._traj_file = None
        self._traj_writer = None

        # Offscreen camera for video (must be decided BEFORE AppLauncher: cameras need the
        # render pipeline enabled at boot).
        if camera is None:
            camera = os.environ.get("ROSE_ISAAC_CAMERA", "") not in ("", "0", "false", "False")
        # Per-frame PNG dump (robust video capture: survives an unclean sync teardown, since
        # frames are on disk as we go). Enabling it forces the camera on.
        self._frame_dir = frame_dir or os.environ.get("ROSE_ISAAC_FRAMEDIR")
        self._frame_idx = 0
        self._camera_on = bool(camera) or bool(self._frame_dir)
        self._camera_res = tuple(camera_res)
        self._camera_eye = tuple(camera_eye)
        self._camera_look = tuple(camera_look)
        self._camera_follow = bool(camera_follow)   # chase cam that keeps the drone framed
        self._camera_offset = tuple(camera_offset)
        self._camera_focal = float(camera_focal)
        self._camera = None

        # --- 1. Boot Isaac Sim (once per process) BEFORE importing any isaaclab.sim/.assets ---
        from isaaclab.app import AppLauncher  # noqa: E402
        launcher_kwargs = {"headless": bool(headless)}
        if self._camera_on:
            launcher_kwargs["enable_cameras"] = True
        if device is not None:
            launcher_kwargs["device"] = device
        self._app_launcher = AppLauncher(**launcher_kwargs)
        self.simulation_app = self._app_launcher.app

        # --- 2. Now the sim modules are importable ---
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass
        from isaaclab_assets import CRAZYFLIE_CFG

        self._torch = torch
        self.device = device or "cuda:0"

        # Per-user log dir: the default (<tmp>/isaaclab/logs) is shared and often owned
        # by another user, which raises PermissionError on multi-user hosts.
        if log_dir is None:
            import tempfile
            log_dir = os.path.join(tempfile.gettempdir(), f"isaaclab-{os.getuid()}", "logs")
        os.makedirs(log_dir, exist_ok=True)

        cam_field = None
        if self._camera_on:
            from isaaclab.sensors import CameraCfg
            cam_field = CameraCfg(
                prim_path="{ENV_REGEX_NS}/rose_cam",
                update_period=0.0,                       # refresh every render
                height=self._camera_res[1],
                width=self._camera_res[0],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=self._camera_focal, focus_distance=400.0,
                    horizontal_aperture=20.955, clipping_range=(0.05, 100.0),
                ),
            )

        @configclass
        class _CrazyflieSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
            )
            robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if cam_field is not None:
            @configclass
            class _SceneCfg(_CrazyflieSceneCfg):
                rose_cam = cam_field
        else:
            _SceneCfg = _CrazyflieSceneCfg

        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / self.phys_freq, device=self.device, log_dir=log_dir)
        self.sim = SimulationContext(sim_cfg)
        self.scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.0))
        self.sim.reset()

        self.robot = self.scene["robot"]
        # 4 propeller bodies — per-rotor thrust is applied here (see quadcopter_fpv demo).
        self._prop_ids = self.robot.find_bodies("m.*_prop")[0]
        self._robot_mass = float(self.robot.root_physx_view.get_masses().sum())
        self._gravity = float(torch.tensor(self.sim.cfg.gravity, device=self.device).norm())
        self._hover_force_per_prop = self._robot_mass * self._gravity / 4.0
        self._sim_dt = self.sim.get_physics_dt()

        if self._camera_on:
            self._camera = self.scene["rose_cam"]
            self._update_camera_pose()
        if self._frame_dir:
            os.makedirs(self._frame_dir, exist_ok=True)

        f32 = np.float32
        big = np.finfo(f32).max
        self.observation_space = spaces.Dict({
            # target-relative TinyMPC state the controller consumes
            "mpc_state": spaces.Box(-big, big, (12,), f32),
            # absolute pose kept for convenience (harmless — only bound indices go on the wire)
            "pos":  spaces.Box(-big, big, (3,), f32),
            "quat": spaces.Box(-1.0, 1.0, (4,), f32),
        })
        # 4 normalized motor thrusts (controller output, ~[-0.583, 0.417])
        self.action_space = spaces.Box(-1.0, 1.0, (4,), f32)
        self._dbg = 0

    # --- state read: world-frame pos/vel/angvel + Rodrigues attitude (matches PyBullet env) ---
    def _read_state(self):
        d = self.robot.data
        pos = d.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        quat = d.root_quat_w[0].detach().cpu().numpy().astype(np.float64)   # (w,x,y,z)
        vel = d.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
        angv = d.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
        return pos, quat, vel, angv

    def _obs_info(self, forces_z, action_norm):
        """Return (obs for the SoC wire, info with ground truth for logging)."""
        pos, quat, vel, angv = self._read_state()
        r = _quat_to_rodrigues(quat[0], quat[1], quat[2], quat[3])
        mpc_state = np.concatenate([pos - self.target, r, vel, angv]).astype(np.float32)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)  # (x,y,z,w)
        obs = {
            "mpc_state": mpc_state,
            "pos": pos.astype(np.float32),
            "quat": quat_xyzw,
        }
        # Ground truth — NEVER serialized to the SoC; for logging / plotting / video.
        info = {
            "gt_pos": pos.astype(np.float32),           # absolute world position
            "gt_quat": quat.astype(np.float32),         # (w,x,y,z)
            "gt_vel": vel.astype(np.float32),           # world-frame linear velocity
            "gt_angvel": angv.astype(np.float32),       # world-frame angular velocity
            "rotor_force_N": np.asarray(forces_z, dtype=np.float32),
            "action_norm": np.asarray(action_norm, dtype=np.float32),
            "target": self.target.astype(np.float32),
            "sim_time": np.float32(self._t),
        }
        return obs, info, (pos, quat, vel, angv)

    def _update_camera_pose(self):
        """Aim the offscreen camera. In follow mode it chases the drone (eye = pos+offset,
        look = pos) so the Crazyflie stays framed and close; otherwise a fixed vantage."""
        if not self._camera_on or self._camera is None:
            return
        torch = self._torch
        if self._camera_follow:
            pos = self.robot.data.root_pos_w[0].detach().cpu().numpy()
            eye = pos + np.asarray(self._camera_offset, dtype=np.float64)
            look = pos
        else:
            eye = np.asarray(self._camera_eye, dtype=np.float64)
            look = np.asarray(self._camera_look, dtype=np.float64)
        eyes = torch.tensor([eye], device=self.device, dtype=torch.float32)
        targets = torch.tensor([look], device=self.device, dtype=torch.float32)
        self._camera.set_world_poses_from_view(eyes, targets)

    def _apply_wrench(self, forces_z):
        """forces_z: (4,) per-prop thrust in N -> set per-rotor z-force + yaw reaction torque."""
        torch = self._torch
        forces = torch.zeros(self.robot.num_instances, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        fz = torch.as_tensor(forces_z, dtype=forces.dtype, device=self.device)
        forces[0, :, 2] = fz
        # propeller drag reaction about body z (alternating spin sign)
        tz = torch.as_tensor(_SPIN_SIGN * _KM_OVER_KF, dtype=forces.dtype, device=self.device)
        torques[0, :, 2] = tz * fz
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=self._prop_ids,
        )

    def _thrusts_to_forces(self, u):
        """Normalized TinyMPC thrusts -> per-motor force in N (same scale as PyBullet env)."""
        thrust_n = (np.asarray(u, dtype=np.float64) + _HOVER_THRUST) * _MAX_THRUST_N
        return np.clip(thrust_n, 0.0, None)

    # --- ground-truth trajectory CSV ---
    def _open_traj(self):
        if not self._traj_csv_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._traj_csv_path)), exist_ok=True)
        self._traj_file = open(self._traj_csv_path, "w", newline="")
        self._traj_writer = csv.writer(self._traj_file)
        self._traj_writer.writerow(_TRAJ_COLS)

    def _traj_row(self, gt, forces_z, action_norm):
        if self._traj_writer is None:
            return
        pos, quat, vel, angv = gt
        row = ([self._t] + list(pos) + list(quat) + list(vel) + list(angv)
               + list(np.asarray(forces_z, dtype=float))
               + list(np.asarray(action_norm, dtype=float)) + list(self.target))
        self._traj_writer.writerow([f"{x:.6g}" for x in row])
        self._traj_file.flush()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        robot = self.robot
        joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        root_state[:, 2] = self.start_height
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        self.scene.reset()
        # settle the initial write into the data buffers
        self.scene.write_data_to_sim()
        self.scene.update(self._sim_dt)
        self._t = 0.0
        self._dbg = 0
        if self._traj_file is None:
            self._open_traj()
        forces_z = np.full(4, self._hover_force_per_prop, dtype=np.float64)
        obs, info, _ = self._obs_info(forces_z, np.zeros(4))
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < 4:
            a = np.zeros(4, dtype=np.float64)
        if not np.any(a):
            forces_z = np.full(4, self._hover_force_per_prop, dtype=np.float64)  # hover until first cmd
        else:
            forces_z = self._thrusts_to_forces(a[:4])
        # sub-step physics `decimation` times per control tick, re-asserting the wrench
        for _ in range(self.decimation):
            self._apply_wrench(forces_z)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self._sim_dt)
        if self._camera_on:
            self._update_camera_pose()        # chase the drone before rendering
            self.sim.render()
            self.scene.update(self._sim_dt)   # pull the freshly-rendered camera frame
        self._t += 1.0 / self.ctrl_freq
        obs, info, gt = self._obs_info(forces_z, a[:4])
        self._traj_row(gt, forces_z, a[:4])
        if self._frame_dir:
            frame = self.render()
            if frame is not None:
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(self._frame_dir, f"frame_{self._frame_idx:05d}.png"), frame)
                self._frame_idx += 1
        if os.environ.get("ROSE_ENV_DEBUG"):
            self._dbg += 1
            if self._dbg % 25 == 1:
                p = obs["pos"]
                print(f"[isaac-cf] step {self._dbg:5d} pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"z_err={p[2]-self.target[2]:+.2f} thrust={np.round(a[:4],3)}", flush=True)
        return obs, 0.0, False, False, info

    def render(self):
        """Return the offscreen camera's RGB frame (H,W,3 uint8), or None if no camera."""
        if not self._camera_on or self._camera is None:
            return None
        rgb = self._camera.data.output["rgb"]
        if rgb is None or rgb.shape[0] == 0:
            return None
        return rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    def close(self):
        try:
            if self._traj_file is not None:
                self._traj_file.close()
                self._traj_file = None
        except Exception:
            pass
        try:
            self.simulation_app.close()
        except Exception:
            pass
