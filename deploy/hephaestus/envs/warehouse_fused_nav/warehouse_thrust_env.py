"""WarehouseThrustEnv — Stage-2 RoSE bridge: the FULL flight-control stack (state estimator +
TinyMPC) runs ON the SoC guest, which outputs 4 MOTOR THRUSTS. This wrapper serves the RAW
sensor suite the guest's estimator consumes and applies the guest's thrusts directly to the
drone via per-rotor wrench (MotorThrustAction), REPLACING the host-side Lee velocity tracker.

Contrast Stage-1 (WarehouseFusedNavBridgeEnv): there the host computed the velocity command's
low-level wrench (Lee tracker) and the SoC only did vision→(yaw_rate,fwd). Here the SoC is the
WHOLE controller (estimator + TinyMPC → thrusts); Isaac only integrates physics.

Wire (config_gym_WarehouseThrustEnv-v0.yaml — matches the multisensor overlay the nav guest binds):
  accel 0x12[3] | gyro 0x15[3] | flow 0x13[2] | tof 0x14[1] (down height) |
  4× multizone tof 0x30-0x33[64] (served FAR=no-wall for open-warehouse velocity nav) |
  fpv 0x11 DMA (only if the guest requests it — vision variant) |
  thrust 0x20 action_latch = 4 normalized motor thrusts (u≈0 at hover).

FREEZE seam (default): one physics step per received 0x20 thrust — a coherent
(sensors→estimator→TinyMPC→thrust→step) tuple per control tick.
Env-vars: ROSE_WH_SEED (spawn), ROSE_FREEZE, ROSE_WH_OBST, ROSE_ISAAC_CAMERA/ROSE_VIDEO_DIR/
ROSE_FPV_DIR (M4 video), ROSE_TRAJ_CSV.
"""
import argparse
import math
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_XPURT = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"
ZONES = 64
FAR_M = 4.0   # multizone ToF "no wall" distance (>= max range)
_G_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def _quat_to_matrix(qw, qx, qy, qz):
    """(w,x,y,z) unit quaternion -> body->world rotation matrix (v_world = R @ v_body). Verbatim
    from crazyflie_sensor_env._quat_to_matrix (the working stable-hover env)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _quat_to_rodrigues(qw, qx, qy, qz):
    """(w,x,y,z) -> Rodrigues (q_xyz/q_w), matching the estimator get_state attitude + mpc_env."""
    if abs(qw) < 1e-9:
        qw = 1e-9 if qw >= 0 else -1e-9
    return np.array([qx / qw, qy / qw, qz / qw], dtype=np.float64)

# --- Stage-1 fused-vision pre-quant/assembly (verbatim from WarehouseFusedNavBridgeEnv) ---
S_FRONT = 0.007683348467969519
S_TOF = 0.007874015748031496
_GROUPS = [("optical_flow", 2), ("down_tof", 1), ("baro", 2), ("quat", 4),
           ("body_rates", 3), ("desired_vel", 3)]
FRONT_H, FRONT_W = 60, 90


def _q8_np(x, s):
    return np.ascontiguousarray(np.clip(np.round(x / s), -128, 127).astype(np.int8))


class WarehouseThrustEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        import sys
        sys.path.insert(0, _XPURT)
        for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
            sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
        from isaaclab.app import AppLauncher
        self._app = AppLauncher(headless=True, enable_cameras=True).app   # kwargs form (honors cameras)

        import torch
        import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from sims.isaaclab_tasks.forest_trail import sensors as S
        from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW
        from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (
            WarehouseNavEnvCfg_PLAY_WithSensors)
        from sims.isaaclab_tasks.warehouse_nav.mdp_motor_thrust_action import MotorThrustActionCfg
        self._torch, self._S, self._GW = torch, S, GW

        self._freeze = os.environ.get("ROSE_FREEZE", "1") != "0"
        self._seed = int(os.environ.get("ROSE_WH_SEED", "1000"))
        self._prev_vel_w = None                       # finite-diff fallback for dynamic accel
        self._last_accel = np.zeros(3, np.float32)    # served accel_body (for debug logging)
        # Open-loop isolation: if set, IGNORE the guest's thrust and apply this constant u to all
        # rotors (u=0 => exact hover). Lets the estimator run on the served sensors while the
        # actuators stay safe, so a diverging estimate can't drive a runaway. Compare the guest's
        # printed estimate to the physics ground truth logged here.
        _fu = os.environ.get("ROSE_FIXED_U", "")
        self._fixed_u = float(_fu) if _fu != "" else None

        TASK_ID = "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0"
        cfg = WarehouseNavEnvCfg_PLAY_WithSensors()
        cfg.scene.num_envs = 1
        cfg.curriculum.obstacle_count.params["min_level"] = int(os.environ.get("ROSE_WH_OBST", "0"))
        cfg.episode_length_s = float(os.environ.get("ROSE_WH_EPLEN", str(cfg.episode_length_s)))
        # Step the env at 200 Hz to MATCH the guest estimator's CTRL_DT=0.005 and the co-sim
        # gym_timestep=0.005. The warehouse default (dt=0.01, decimation=2 => 50 Hz control) made
        # the on-SoC estimator integrate only 1/4 of each physics step -> its state lagged the real
        # motion 4x and TinyMPC chased a stale estimate -> divergence. control_dt = dt*decim = 0.005.
        cfg.sim.dt = float(os.environ.get("ROSE_WH_SIM_DT", "0.005"))
        cfg.decimation = int(os.environ.get("ROSE_WH_DECIM", "1"))
        # LAZY / on-demand camera render (dima 2026-08-11): the guest-consumed cam_front sensor
        # (front_camera, update_period=0.1) only refreshes at 10 Hz and is served over the bridge at
        # ~5 Hz, but IsaacLab paid a FULL RTX render EVERY 200 Hz physics step (render_interval=decim=1,
        # forced by the always-on 960x540 chase_camera at update_period=0). Measured: 30.96 ms/step
        # camera-on vs 6.06 ms physics-only -> ~24.9 ms/step (80%) was redundant render.
        # Fix: DECOUPLE the RTX pass from physics by aligning render_interval to cam_front's cadence
        # (ManagerBasedRLEnv only calls sim.render() when _sim_step_counter % render_interval == 0),
        # so physics still steps every tick but the render fires only when cam_front is due. This is
        # functionally equivalent for the guest (same frames at the same sim states) — it drops only
        # redundant renders. ROSE_RENDER_HZ=0 restores the old per-step render (render_interval=decim).
        _render_hz = float(os.environ.get("ROSE_RENDER_HZ", "10"))
        if _render_hz > 0:
            cfg.sim.render_interval = max(cfg.decimation, int(round((1.0 / _render_hz) / cfg.sim.dt)))
        else:
            cfg.sim.render_interval = cfg.decimation
        # chase_camera is a 960x540 VIDEO/visualization product NOT consumed by the guest; drop it from
        # the co-sim scene unless we're explicitly recording (ROSE_ISAAC_CAMERA=1). Its every-render
        # 960x540 pass is the single biggest render cost. When recording it comes back at render_hz rate.
        if os.environ.get("ROSE_ISAAC_CAMERA", "") != "1" and getattr(cfg.scene, "chase_camera", None) is not None:
            cfg.scene.chase_camera = None
        print(f"[WarehouseThrust] lazy_render: render_hz={_render_hz} render_interval={cfg.sim.render_interval} "
              f"chase_cam={'on' if getattr(cfg.scene, 'chase_camera', None) is not None else 'off'}", flush=True)
        # Fixed spawn altitude so it MATCHES the guest estimator's START_Z init (no huge ToF
        # innovation transient). Set together with the guest's -DSTART_Z.
        _sz = os.environ.get("ROSE_WH_SPAWN_Z", "")
        if _sz != "":
            cfg.events.reset_base.params["pose_range"]["z"] = (float(_sz), float(_sz))
        cfg.actions.velocity = MotorThrustActionCfg(asset_name="robot")   # SoC TinyMPC is the controller
        print(f"[WarehouseThrust] gym.make {TASK_ID} freeze={self._freeze} seed={self._seed} "
              f"action=MotorThrust(4)", flush=True)
        self._env = RslRlVecEnvWrapper(gym.make(TASK_ID, cfg=cfg))
        self._uenv = self._env.unwrapped
        self._uenv.sim._disable_app_control_on_stop_handle = True
        self._dev = self._uenv.device
        self._N = self._uenv.num_envs
        self._control_dt = float(cfg.sim.dt * cfg.decimation)
        self._robot = self._uenv.scene["robot"]
        self._base_id = self._robot.find_bodies("body")[0][0]   # for Isaac direct body accel
        self._origin = self._uenv.scene.env_origins
        self._gate_centers = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)
        self._pass_radius = GW.FixedGateCourseCommandCfg().success_radius
        self._K = len(self._gate_centers)
        # Vision serving (ROSE_VISION=1): also serve the fused-model inputs (front int8 / tof_cross
        # int8 / lowdim f32). The lowdim carries a host-est quat placeholder that the GUEST OVERWRITES
        # with its own EKF quat. desired_vel = host-computed goal guidance vs the gate line.
        self._vision = os.environ.get("ROSE_VISION", "") == "1"
        self.base_speed = float(os.environ.get("ROSE_WH_BASE_SPEED", "1.4"))
        self._est = None
        if self._vision:
            from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator
            self._est = StateEstimator(self._N, self._dev, control_dt=self._control_dt)
            print("[WarehouseThrust] ROSE_VISION=1: serving fused inputs (cam_front/tof_cross/lowdim)",
                  flush=True)

        f32 = np.float32
        big = np.finfo(f32).max
        self.observation_space = spaces.Dict({
            "accel": spaces.Box(-big, big, (3,), f32),
            "gyro":  spaces.Box(-big, big, (3,), f32),
            "flow":  spaces.Box(-big, big, (2,), f32),
            "tof":   spaces.Box(0.0, big, (1,), f32),
            "front": spaces.Box(0.0, big, (ZONES,), f32),
            "right": spaces.Box(0.0, big, (ZONES,), f32),
            "back":  spaces.Box(0.0, big, (ZONES,), f32),
            "left":  spaces.Box(0.0, big, (ZONES,), f32),
            "state": spaces.Box(-big, big, (12,), f32),   # GT 12-vec for the estimator-bypass test
            "cam_front": spaces.Box(-128, 127, (FRONT_H * FRONT_W,), np.int8),  # fused model front int8
            "tof_cross": spaces.Box(-128, 127, (256,), np.int8),               # fused model tof int8
            "lowdim":    spaces.Box(-big, big, (21,), f32),                    # fused model lowdim f32
        })
        # 4 normalized motor thrusts (u ~ [-0.583, 0.417]); bounded so default = 0 (hover u).
        self.action_space = spaces.Box(-2.0, 2.0, (4,), f32)

        self._goal_idx = 0
        self._gates_passed = 0
        self._tick = 0
        self._ep = 0
        self._pending_cmds = 0
        self._cmd_count = 0
        self._frozen_obs = None
        self._last_u = np.zeros(4, np.float32)

        # video (M4) — follow chase + FPV, decimated JPEGs (robust to sync os._exit)
        self._cam_on = os.environ.get("ROSE_ISAAC_CAMERA", "") == "1"
        self._video_dir = os.environ.get("ROSE_VIDEO_DIR", "")
        self._fpv_dir = os.environ.get("ROSE_FPV_DIR", "")
        self._video_decim = int(os.environ.get("ROSE_VIDEO_DECIM", "2"))
        self._vframe = 0; self._fpvframe = 0
        self._chase = self._uenv.scene["chase_camera"] if "chase_camera" in self._uenv.scene.sensors else None
        self._front = self._uenv.scene["front_camera"] if "front_camera" in self._uenv.scene.sensors else None
        if self._cam_on and self._video_dir:
            os.makedirs(self._video_dir, exist_ok=True)
        if self._cam_on and self._fpv_dir:
            os.makedirs(self._fpv_dir, exist_ok=True)
        self._traj_path = os.environ.get("ROSE_TRAJ_CSV", "")
        self._traj_f = None
        if self._traj_path:
            self._traj_f = open(self._traj_path, "w")
            self._traj_f.write("tick,ep,t,x,y,z,roll,pitch,yaw,u0,u1,u2,u3,goal,gates\n")
            self._traj_f.flush()

    def on_action_received(self):
        self._pending_cmds += 1
        self._cmd_count += 1

    # ---- pose / raw-sensor helpers ----
    def _rpy(self):
        q = self._robot.data.root_quat_w[0]
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def _xy_local(self):
        return (self._robot.data.root_pos_w[0] - self._origin[0])[:2].cpu().numpy().astype(np.float64)

    def _desired_vel(self, yaw_now):
        """Host-computed goal guidance vs the gate line (Stage-1), body-frame, * base_speed.
        Advances the gate index + emits [GATE] events (also used for M4 gates_passed)."""
        torch = self._torch
        xy = self._xy_local()
        goal_xy = self._gate_centers[min(self._goal_idx, self._K - 1)]
        if np.linalg.norm(xy - goal_xy) < self._pass_radius:
            if self._goal_idx < self._K - 1:
                self._goal_idx += 1
                print(f"[GATE] passed gate {self._goal_idx}/{self._K} at t="
                      f"{self._tick*self._control_dt:.2f}s xy=({xy[0]:+.2f},{xy[1]:+.2f})", flush=True)
                goal_xy = self._gate_centers[self._goal_idx]
            else:
                self._gates_passed = self._K
        self._gates_passed = max(self._gates_passed, self._goal_idx)
        dvec = goal_xy - xy
        cy, sy = math.cos(-yaw_now), math.sin(-yaw_now)
        bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
        nrm = max((bx * bx + by * by) ** 0.5, 1e-6)
        return torch.tensor([[bx / nrm * self.base_speed, by / nrm * self.base_speed, 0.0]],
                            device=self._dev, dtype=torch.float32).repeat(self._N, 1)

    def _sense(self, desired_vel):
        """Stage-1 sense() dict for the fused model (uses a host StateEstimator for baro/quat; the
        quat placeholder is OVERWRITTEN by the guest's EKF quat on-SoC)."""
        S, torch, u = self._S, self._torch, self._uenv
        grey = S.front_greyscale(u)
        tof_norm, _ = S.normalize_range(S.tof_stack(u), S.TOF_RANGE_MIN, S.TOF_RANGE_MAX)
        dtof = S.down_tof(u)
        dtof_norm, dtof_valid = S.normalize_range(dtof, S.DOWN_TOF_RANGE_MIN, S.DOWN_TOF_RANGE_MAX)
        flow = S.optical_flow(u); flow_valid = S.optical_flow_valid(u)
        baro = S.barometer(u, drift=self._est.step_baro_drift())
        gyro = self._robot.data.root_ang_vel_b[:, :3]
        accel = -self._robot.data.projected_gravity_b * 9.81
        filt = self._est.update(gyro, accel, baro_alt=baro[:, 1], tof_alt=dtof.squeeze(1), flow_vel=flow * 0.0)
        return {"front_grey": grey.float(), "tof_cross": tof_norm, "optical_flow": flow,
                "down_tof": dtof_norm, "baro": baro / 10.0, "quat": filt["quat"],
                "body_rates": gyro, "desired_vel": desired_vel,
                "flags": torch.cat([flow_valid, dtof_valid, torch.ones(self._N, 4, device=self._dev)], dim=1)}

    def _prequantize(self, sd):
        import torch.nn.functional as F
        fg = F.interpolate(sd["front_grey"][:1].float(), size=(FRONT_H, FRONT_W),
                           mode="bilinear", align_corners=False).detach().cpu().numpy().reshape(-1)
        front = _q8_np(fg, S_FRONT)
        tof = _q8_np(sd["tof_cross"][:1].detach().cpu().numpy().reshape(-1), S_TOF)
        parts = []
        for k, dim in _GROUPS:
            v = sd.get(k)
            parts.append(v[:1].detach().cpu().numpy().reshape(-1)[:dim].astype(np.float32)
                         if v is not None else np.zeros(dim, np.float32))
        fl = sd.get("flags")
        fl = (fl[:1].detach().cpu().numpy().reshape(-1)[:6].astype(np.float32)
              if fl is not None else np.ones(6, np.float32))
        lowdim = np.ascontiguousarray(np.concatenate(parts + [fl]).astype(np.float32))
        return {"cam_front": front, "tof_cross": tof, "lowdim": lowdim}

    def _vision_inputs(self):
        _, _, yaw = self._rpy()
        return self._prequantize(self._sense(self._desired_vel(yaw)))

    def _sense_raw(self):
        """Serve the RAW sensors in the EXACT convention the on-SoC estimator expects (mirrors the
        WORKING IsaacCrazyflieSensorEnv, where this estimator flew a stable hover):
          accel 0x12 = TRUE specific force incl. DYNAMIC term — VERBATIM from the WORKING
                       crazyflie_sensor_env (a_world = Isaac's DIRECT body_lin_acc_w, low-lag;
                       finite-diff fallback on the reset frame; accel_body = R^T·(a_world-g_world),
                       R=body->world, g_world=(0,0,-9.81)). At rest -> [0,0,+9.81]; climbing ->
                       accel_body_z increases,
          gyro  0x15 = body angular velocity (rad/s),
          flow  0x13 = body-frame horizontal velocity root_lin_vel_b[:2] (m/s),
          tof   0x14 = downward height AGL in METRES (root_pos_w-origin).z  [anchors altitude].
        Multizone walls served FAR (open warehouse → velocity nav, no position fusion)."""
        d = self._robot.data
        quat = d.root_quat_w[0].detach().cpu().numpy().astype(np.float64)   # (w,x,y,z)
        R = _quat_to_matrix(quat[0], quat[1], quat[2], quat[3])             # body->world
        vel_w = d.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
        a_isaac = d.body_lin_acc_w[0, self._base_id].detach().cpu().numpy().astype(np.float64)
        if np.linalg.norm(a_isaac) > 1e-6:          # Isaac reports ~0 on the pre-force reset frame
            a_world = a_isaac
        elif self._prev_vel_w is not None:
            a_world = (vel_w - self._prev_vel_w) / self._control_dt
        else:
            a_world = np.zeros(3)
        self._prev_vel_w = vel_w.copy()
        accel = (R.T @ (a_world - _G_WORLD)).astype(np.float32)            # specific force, body frame
        self._last_accel = accel
        gyro = d.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
        flow = d.root_lin_vel_b[0, :2].cpu().numpy().astype(np.float32)   # body-frame vel (m/s)
        h_agl = float((d.root_pos_w[0] - self._origin[0])[2].item())      # height AGL (m)
        far = np.full(ZONES, FAR_M, np.float32)
        # GT-STATE (0x16): the sim's TRUE 12-vec [x,y,z, rodrigues, vel_world, rates_body] for the
        # estimator-bypass isolating test (matches the estimator get_state layout the guest expects).
        pos = (d.root_pos_w[0] - self._origin[0]).detach().cpu().numpy().astype(np.float64)
        rod = _quat_to_rodrigues(quat[0], quat[1], quat[2], quat[3])
        rates_b = d.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float64)
        state12 = np.concatenate([pos, rod, vel_w, rates_b]).astype(np.float32)
        out = {
            "accel": np.ascontiguousarray(accel),
            "gyro":  np.ascontiguousarray(gyro),
            "flow":  np.ascontiguousarray(flow),
            "tof":   np.array([h_agl], np.float32),
            "front": far.copy(), "right": far.copy(), "back": far.copy(), "left": far.copy(),
            "state": np.ascontiguousarray(state12),
        }
        if self._vision:
            out.update(self._vision_inputs())
        return out

    def _capture_video(self):
        if not self._cam_on or (self._tick % self._video_decim) != 0:
            return
        try:
            import imageio
            if self._video_dir and self._chase is not None:
                rgb = self._chase.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3]
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(self._video_dir, f"f_{self._vframe:05d}.jpg"),
                                np.ascontiguousarray(rgb[::2, ::2]))
                self._vframe += 1
            if self._fpv_dir and self._front is not None:
                fpv = self._front.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3]
                if fpv.dtype != np.uint8:
                    fpv = (np.clip(fpv, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(self._fpv_dir, f"f_{self._fpvframe:05d}.jpg"),
                                np.ascontiguousarray(fpv))
                self._fpvframe += 1
        except Exception as e:
            if self._vframe == 0 and self._fpvframe == 0:
                print(f"[WarehouseThrust] video capture off ({e})", flush=True)
            self._cam_on = False

    def _drive_chase(self):
        if self._chase is None:
            return
        torch = self._torch
        p = self._robot.data.root_pos_w[0]; q = self._robot.data.root_quat_w[0]
        w, x, y, z = q[0], q[1], q[2], q[3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fx, fy = torch.cos(yaw), torch.sin(yaw)
        eye = torch.stack([p[0] - fx * 3.0, p[1] - fy * 3.0, p[2] + 1.6])
        tgt = torch.stack([p[0] + fx * 4.0, p[1] + fy * 4.0, p[2] - 0.1])
        self._chase.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))

    def _log_traj(self, z, roll, pitch, yaw, u):
        if self._traj_f is None:
            return
        xy = self._xy_local()
        self._traj_f.write(f"{self._tick},{self._ep},{self._tick*self._control_dt:.4f},"
                           f"{xy[0]:.4f},{xy[1]:.4f},{z:.4f},{roll:.4f},{pitch:.4f},{yaw:.4f},"
                           f"{u[0]:.4f},{u[1]:.4f},{u[2]:.4f},{u[3]:.4f},{self._goal_idx},{self._gates_passed}\n")
        if (self._tick % 10) == 0:
            self._traj_f.flush()

    def _refresh_obs(self):
        self._frozen_obs = self._sense_raw()

    def _apply_and_step(self, u):
        """One control tick: apply the guest's 4 motor thrusts (u) via MotorThrustAction, step."""
        torch = self._torch
        if self._cam_on:
            self._drive_chase()
        act = torch.as_tensor(np.asarray(u, np.float32), device=self._dev).reshape(1, 4).repeat(self._N, 1)
        self._env.step(act)
        self._tick += 1
        self._last_u = np.asarray(u, np.float32)
        z = float(self._robot.data.root_pos_w[0, 2].item())
        roll, pitch, yaw = self._rpy()
        self._capture_video()
        self._log_traj(z, roll, pitch, yaw, u)
        if (self._tick % 10) == 0:
            xy = self._xy_local()
            vb = self._robot.data.root_lin_vel_b[0].cpu().numpy()   # physics body velocity
            vz = float(self._robot.data.root_lin_vel_w[0, 2].item())
            ab = self._last_accel
            print(f"[phys] t={self._tick*self._control_dt:5.2f}s z={z:.3f} vz={vz:+.3f} "
                  f"vx={vb[0]:+.3f} vy={vb[1]:+.3f} roll={math.degrees(roll):+5.1f} pitch={math.degrees(pitch):+5.1f} "
                  f"accel_b=[{ab[0]:+.2f},{ab[1]:+.2f},{ab[2]:+.2f}] "
                  f"u=[{u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f},{u[3]:+.2f}] cmds={self._cmd_count} ticks={self._tick}",
                  flush=True)
        if z < 0.15:
            print(f"[CRASH] ground at t={self._tick*self._control_dt:.2f}s z={z:.2f}", flush=True)
        self._refresh_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._torch.manual_seed(self._seed)
        self._env.reset()
        self._prev_vel_w = None
        self._goal_idx = 0; self._gates_passed = 0; self._tick = 0; self._pending_cmds = 0
        self._refresh_obs()
        return self._frozen_obs, {}

    def step(self, action):
        u = np.asarray(action, dtype=np.float32).reshape(-1)[:4]
        if self._fixed_u is not None:      # open-loop isolation: ignore guest thrust, hold u=const
            u = np.full(4, self._fixed_u, np.float32)
        if self._freeze:
            if self._pending_cmds > 0:
                self._pending_cmds -= 1
                self._apply_and_step(u)
        else:
            self._apply_and_step(u)
        return self._frozen_obs, 0.0, False, False, {}

    def render(self):
        if self._chase is None:
            return np.zeros((16, 16, 3), np.uint8)
        rgb = self._chase.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)
