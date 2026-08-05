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


def _euler_to_quat_wxyz(roll, pitch, yaw):
    """roll/pitch/yaw (rad, ZYX) -> unit quaternion (w,x,y,z) for root_state[:,3:7]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


class IsaacCrazyflieMPCEnv(gym.Env):
    """IsaacLab Crazyflie exposed as the TinyMPC full-state HIL loop over the RoSE bridge."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, ctrl_freq=50, phys_freq=100, target=None,
                 device=None, headless=True, start_height=0.5, log_dir=None,
                 traj_csv=None, camera=None, camera_res=(640, 480),
                 camera_eye=(2.0, 2.0, 1.6), camera_look=(0.0, 0.0, 1.0),
                 camera_follow=True, camera_offset=(0.45, 0.45, 0.18),
                 camera_focal=35.0, frame_dir=None, fpv_isaac=None, **kwargs):
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
        # Env-var camera overrides (for nicer recording angles without editing defaults):
        # ROSE_CAM_EYE / ROSE_CAM_LOOK / ROSE_CAM_OFFSET = "x,y,z"; ROSE_CAM_FOLLOW = 0/1;
        # ROSE_CAM_FOCAL = mm. A fixed elevated view (FOLLOW=0 + EYE/LOOK) reads motion best.
        def _vec3(name, default):
            v = os.environ.get(name)
            if not v:
                return default
            parts = [float(x) for x in v.replace(" ", "").split(",")]
            return tuple(parts[:3]) if len(parts) >= 3 else default
        self._camera_eye = _vec3("ROSE_CAM_EYE", self._camera_eye)
        self._camera_look = _vec3("ROSE_CAM_LOOK", self._camera_look)
        self._camera_offset = _vec3("ROSE_CAM_OFFSET", self._camera_offset)
        if os.environ.get("ROSE_CAM_FOLLOW") is not None:
            self._camera_follow = os.environ.get("ROSE_CAM_FOLLOW") not in ("0", "false", "False")
        self._camera_focal = float(os.environ.get("ROSE_CAM_FOCAL", self._camera_focal))
        self._camera = None

        # Forward-facing FPV camera (HM01B0-equivalent), body-mounted, for a real rendered
        # first-person frame served to the SoC. Opt-in via ROSE_ISAAC_FPV=1 (needs the render
        # pipeline, so it must be decided BEFORE AppLauncher). Distinct from the chase cam above
        # (which is for offscreen video). HM01B0: QVGA 320×240, 8-bit mono, 3.6 µm pixel (1/6"
        # optical format); FoV is lens-dependent (AI-deck stock lens ≈ 87° diagonal) — modeled
        # as a pinhole with horizontal FoV = ROSE_FPV_FOV (default 70°).
        if fpv_isaac is None:
            fpv_isaac = os.environ.get("ROSE_ISAAC_FPV", "") not in ("", "0", "false", "False")
        self._fpv_isaac_on = bool(fpv_isaac)
        self._fpv_isaac_res = (int(os.environ.get("ROSE_FPV_ISAAC_W", "320")),
                               int(os.environ.get("ROSE_FPV_ISAAC_H", "240")))
        self._fpv_fov = float(os.environ.get("ROSE_FPV_FOV", "70.0"))
        self._fpv_mount = _vec3("ROSE_FPV_MOUNT", (0.03, 0.0, 0.0))  # body +x, forward
        self._fpv_camera = None

        # --- 1. Boot Isaac Sim (once per process) BEFORE importing any isaaclab.sim/.assets ---
        from isaaclab.app import AppLauncher  # noqa: E402
        launcher_kwargs = {"headless": bool(headless)}
        if self._camera_on or self._fpv_isaac_on:
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

        # Forward FPV camera (HM01B0), mounted on the drone body so it moves+rotates with it.
        # Properties matched to a real HM01B0: 320×240 QVGA, 3.6 µm pixel -> physical sensor
        # width = W*3.6µm; the pinhole focal length is derived so the horizontal FoV equals the
        # modeled AI-deck lens (ROSE_FPV_FOV). The frame is rendered RGB and converted to 8-bit
        # grayscale in render_fpv_gray() (HM01B0 is monochrome).
        fpv_field = None
        if self._fpv_isaac_on:
            import math
            from isaaclab.sensors import CameraCfg
            hm_w, hm_h = self._fpv_isaac_res
            px_mm = 3.6e-3                               # HM01B0 pixel pitch (mm)
            aperture = hm_w * px_mm                      # physical sensor width (mm)
            focal = (aperture / 2.0) / math.tan(math.radians(self._fpv_fov) / 2.0)
            fpv_field = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/body/rose_fpv",   # child of the body link
                update_period=0.0,
                height=hm_h, width=hm_w,
                data_types=["rgb"],
                # look down body +x (forward), image-up = body +z, right = body -y
                offset=CameraCfg.OffsetCfg(pos=tuple(self._fpv_mount),
                                           rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=focal, focus_distance=100.0,
                    horizontal_aperture=aperture, clipping_range=(0.02, 100.0),
                ),
            )
            print("[crazyflie_env] FPV(HM01B0) %dx%d hfov=%.0f focal=%.3fmm aperture=%.3fmm"
                  % (hm_w, hm_h, self._fpv_fov, focal, aperture), flush=True)

        # Robot cfg with rigid-body SLEEP DISABLED. Defensive only: a perfectly-still hover lets
        # PhysX sleep the body after ~1 s, a plausible contributor to the systemic ~235-step
        # co-sim stall (docs/ROSE_FLIGHT_CONTROLLER_THREADING.md). NOTE: this alone does NOT
        # resolve the stall (the committed single-loop still hangs at ~236 with sleep disabled) —
        # kept as a harmless precaution while the real cause is run down via SoC (TACIT) traces.
        _robot_cfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        try:
            _rp = getattr(_robot_cfg.spawn, "rigid_props", None)
            if _rp is not None:
                _rp.sleep_threshold = 0.0
                _rp.stabilization_threshold = 0.0
        except Exception as _e:  # noqa: BLE001
            print(f"[crazyflie_env] rigid sleep-disable skipped ({_e})", flush=True)

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
            robot = _robot_cfg

        # Chain a subclass per optional camera (configclass fields are declared per class).
        _SceneCfg = _CrazyflieSceneCfg
        if cam_field is not None:
            @configclass
            class _SceneCfgCam(_SceneCfg):
                rose_cam = cam_field
            _SceneCfg = _SceneCfgCam
        if fpv_field is not None:
            @configclass
            class _SceneCfgFpv(_SceneCfg):
                rose_fpv = fpv_field
            _SceneCfg = _SceneCfgFpv

        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / self.phys_freq, device=self.device, log_dir=log_dir)
        self.sim = SimulationContext(sim_cfg)
        self.scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.0))
        # Subclass hook: spawn static environment geometry (e.g. maze/hallway walls) into
        # /World before the sim reset so they are part of the stage. Base env: no-op.
        self._spawn_walls(sim_utils)
        self.sim.reset()

        self.robot = self.scene["robot"]
        # 4 propeller bodies — per-rotor thrust is applied here (see quadcopter_fpv demo).
        self._prop_ids = self.robot.find_bodies("m.*_prop")[0]
        self._robot_mass = float(self.robot.root_physx_view.get_masses().sum())
        self._gravity = float(torch.tensor(self.sim.cfg.gravity, device=self.device).norm())
        self._hover_force_per_prop = self._robot_mass * self._gravity / 4.0
        self._sim_dt = self.sim.get_physics_dt()
        # base ("body") link for external disturbances (stress plan section 3.2)
        self._base_body_id = self.robot.find_bodies("body")[0][0]
        self._dist = self._disturbance_cfg()   # None unless ROSE_WIND_*/ROSE_GUST_*/ROSE_TORQUE_* set

        if self._camera_on:
            self._camera = self.scene["rose_cam"]
            self._update_camera_pose()
        if self._fpv_isaac_on:
            self._fpv_camera = self.scene["rose_fpv"]
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

    def _spawn_walls(self, sim_utils):
        """Hook: spawn static environment geometry into /World (maze/hallway walls). Base
        env has none. Subclasses (e.g. the multisensor nav env) override this."""
        return

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

    def _disturbance_cfg(self):
        """Parse external-disturbance config from env vars (stress plan section 3.2). Returns
        None unless something is set, so the default physics path is bit-for-bit unchanged
        (magnitude-0 gate). Steady wind (world-frame force), a timed gust, and a timed yaw
        torque impulse; wind/gust magnitudes in N, torque in N*m, times/durations in seconds."""
        def g(k, d=0.0):
            return float(os.environ.get(k, d))
        cfg = dict(
            wind_n=g("ROSE_WIND_N"), wind_dir=g("ROSE_WIND_DIR_DEG"),
            gust_n=g("ROSE_GUST_N"), gust_dir=g("ROSE_GUST_DIR_DEG"),
            gust_start=g("ROSE_GUST_START", 3.0), gust_dur=g("ROSE_GUST_DUR", 0.15),
            torque=g("ROSE_TORQUE_IMP"), torque_start=g("ROSE_TORQUE_START", 3.0),
            torque_dur=g("ROSE_TORQUE_DUR", 0.1),
        )
        if cfg["wind_n"] == 0.0 and cfg["gust_n"] == 0.0 and cfg["torque"] == 0.0:
            return None
        return cfg

    def _apply_disturbance(self):
        """Apply the configured external wrench to the BASE body (separate body_id from the
        rotor props, so the composer's per-body `set` leaves the rotor forces intact). World-
        frame wind is rotated into the body frame (the composer's forces are body-local, like
        the rotor thrust). No-op when unconfigured -> baseline unchanged."""
        if self._dist is None:
            return
        torch = self._torch
        c = self._dist
        t = self._t
        # world-frame horizontal force: steady wind + gust during its window
        import math
        fx = c["wind_n"] * math.cos(math.radians(c["wind_dir"]))
        fy = c["wind_n"] * math.sin(math.radians(c["wind_dir"]))
        if c["gust_n"] != 0.0 and c["gust_start"] <= t < c["gust_start"] + c["gust_dur"]:
            fx += c["gust_n"] * math.cos(math.radians(c["gust_dir"]))
            fy += c["gust_n"] * math.sin(math.radians(c["gust_dir"]))
        # rotate world force into the body frame (f_body = R^T f_world)
        q = self.robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)  # wxyz
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        f_body = R.T @ np.array([fx, fy, 0.0])
        tau = 0.0
        if c["torque"] != 0.0 and c["torque_start"] <= t < c["torque_start"] + c["torque_dur"]:
            tau = c["torque"]
        forces = torch.zeros(self.robot.num_instances, 1, 3, device=self.device)
        torques = torch.zeros_like(forces)
        forces[0, 0, :] = torch.as_tensor(f_body, dtype=forces.dtype, device=self.device)
        torques[0, 0, 2] = float(tau)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=[self._base_body_id],
        )

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

    def _apply_hard_ic(self, root_state):
        """Randomize the initial condition in place (stress plan section 3.1).

        No-op unless any ROSE_IC_* env var is set. Adds seeded uniform offsets to position,
        attitude (roll/pitch), linear velocity, and body rates so the estimator must converge
        from a wrong prior and the controller must recover from an off-nominal start. Seeded by
        ROSE_SCENARIO_SEED (falls back to ROSE_SENSOR_NOISE_SEED) for reproducibility.
        """
        pos_m = float(os.environ.get("ROSE_IC_POS", "0"))
        z_m = float(os.environ.get("ROSE_IC_Z", "0"))
        tilt_deg = float(os.environ.get("ROSE_IC_TILT_DEG", "0"))
        vel_ms = float(os.environ.get("ROSE_IC_VEL", "0"))
        vz_ms = float(os.environ.get("ROSE_IC_VZ", "0"))
        rate = float(os.environ.get("ROSE_IC_RATE", "0"))
        if not any((pos_m, z_m, tilt_deg, vel_ms, vz_ms, rate)):
            return
        seed = int(os.environ.get("ROSE_SCENARIO_SEED",
                                  os.environ.get("ROSE_SENSOR_NOISE_SEED", "0")))
        rng = np.random.default_rng(seed)
        torch = self._torch
        dev, dt = self.device, root_state.dtype

        def T(v):
            return torch.as_tensor(v, device=dev, dtype=dt)

        if pos_m:
            root_state[0, 0] += T(rng.uniform(-pos_m, pos_m))
            root_state[0, 1] += T(rng.uniform(-pos_m, pos_m))
        if z_m:
            root_state[0, 2] += T(rng.uniform(-z_m, z_m))
        if tilt_deg:
            roll = np.radians(rng.uniform(-tilt_deg, tilt_deg))
            pitch = np.radians(rng.uniform(-tilt_deg, tilt_deg))
            root_state[0, 3:7] = T(_euler_to_quat_wxyz(roll, pitch, 0.0))
        if vel_ms:
            root_state[0, 7] += T(rng.uniform(-vel_ms, vel_ms))
            root_state[0, 8] += T(rng.uniform(-vel_ms, vel_ms))
        if vz_ms:
            root_state[0, 9] += T(rng.uniform(-vz_ms, vz_ms))
        if rate:
            root_state[0, 10:13] += T(rng.uniform(-rate, rate, 3))

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
        self._apply_hard_ic(root_state)   # stress plan section 3.1 (no-op unless configured)
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
            self._apply_disturbance()  # external wind/gust/torque (no-op unless configured)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self._sim_dt)
        if self._camera_on or self._fpv_isaac_on:
            self._update_camera_pose()        # chase cam (no-op for the body-mounted FPV)
            self.sim.render()                 # renders all cameras (chase + forward FPV)
            self.scene.update(self._sim_dt)   # pull the freshly-rendered camera frame(s)
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

    def render_fpv_gray(self):
        """Return the forward FPV (HM01B0) camera frame as 8-bit grayscale (H,W) uint8 at the
        native HM01B0 resolution, or None if the FPV camera is off/not ready. HM01B0 is a
        monochrome sensor, so the rendered RGB is converted to luma."""
        if not self._fpv_isaac_on or self._fpv_camera is None:
            return None
        rgb = self._fpv_camera.data.output["rgb"]
        if rgb is None or rgb.shape[0] == 0:
            return None
        arr = rgb[0, ..., :3].detach().cpu().numpy().astype(np.float32)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return gray.astype(np.uint8)

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
