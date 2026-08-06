"""WarehouseFusedNavBridgeEnv — RoSE bridge wrapper around the PHOTOREAL warehouse-gate
IsaacLab env for the on-SoC fused-vision nav model (Milestones 1-4).

Instantiates the SAME env as sims/scripts/eval_fused_warehouse.py
(Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0), serves the real onboard
sense() suite PRE-QUANTIZED exactly as the validated host reference
(scratchpad/int8_fused_model_f16.py) over the RoSE bridge, and CLOSES THE LOOP: the guest's
(yaw_rate, forward_speed) → cmd_to_action(yr,fwd,h) (incl. altitude-hold P-loop to TARGET_H=2.0)
→ 4-ch polar velocity → inner warehouse env.step.

Control seam — FREEZE model (default, faithful to the pytorch eval):
  The async per-modality reqrsp transport means one guest inference reads front/tof/lowdim across
  several lockstep tokens. To give ONE coherent (sensors → command → step) tuple per control tick,
  the wrapper HOLDS the served sense() snapshot constant and only advances the inner physics ONCE
  per RECEIVED 0x20 command (signalled by the synchronizer via on_action_received()). Set
  ROSE_FREEZE=0 to instead free-run (advance every env.step; ZOH command; ~3-frame sensor skew).

Wire contract (matches samples/rose_fused_nav + config_gym_WarehouseFusedNavBridgeEnv-v0.yaml):
  front int8[5400] 0x11 DMA | tof int8[256] 0x41 reqrsp | lowdim f32[21] 0x42 reqrsp |
  action f32[2] 0x20 (yaw_rate, forward_speed).

Env-vars: ROSE_WH_OBST (obstacle level, 0=clean), ROSE_FREEZE (default 1), ROSE_ISAAC_CAMERA
(=1 → decimated chase-cam JPEGs to ROSE_VIDEO_DIR), ROSE_VIDEO_DIR, ROSE_TRAJ_CSV,
ROSE_WH_BASE_SPEED. Needs env_isaaclab (GPU); imported lazily by the synchronizer's gym.make().
"""
import argparse
import math
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- host-reference pre-quantization constants (scratchpad/int8_fused_model_f16.py) ---
S_FRONT = 0.007683348467969519
S_TOF = 0.007874015748031496
_GROUPS = [("optical_flow", 2), ("down_tof", 1), ("baro", 2), ("quat", 4),
           ("body_rates", 3), ("desired_vel", 3)]
FRONT_H, FRONT_W = 60, 90

_XPURT = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt"


def _q8_np(x, s):
    return np.ascontiguousarray(np.clip(np.round(x / s), -128, 127).astype(np.int8))


class WarehouseFusedNavBridgeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        import sys
        sys.path.insert(0, _XPURT)
        for _p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
            sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{_p}")
        from isaaclab.app import AppLauncher
        self._app = AppLauncher(argparse.Namespace(headless=True, enable_cameras=True)).app

        import torch
        import sims.isaaclab_tasks.warehouse_nav.config.crazyflie  # noqa: F401 (registers task)
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from sims.isaaclab_tasks.forest_trail import sensors as S
        from sims.isaaclab_tasks.forest_trail.state_estimator import StateEstimator
        from sims.isaaclab_tasks.warehouse_nav import mdp_gates as GW
        from sims.isaaclab_tasks.warehouse_nav.config.crazyflie.warehouse_nav_env_cfg import (
            WarehouseNavEnvCfg_PLAY_WithSensors,
        )
        from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg
        self._torch, self._S, self._GW = torch, S, GW

        _v = VelocityCommandActionCfg()
        self.MAX_SPEED, self.MAX_YAWRATE, self.MAX_INCL = _v.max_speed, _v.max_yawrate, _v.max_inclination
        self.TARGET_H, self.K_ALT, self.VZ_MAX = 2.0, 1.2, 0.8
        self.base_speed = float(os.environ.get("ROSE_WH_BASE_SPEED", "1.4"))
        self._freeze = os.environ.get("ROSE_FREEZE", "1") != "0"
        # Fixed known-good spawn seed. The on-spike model is ≤1 ULP of pytorch, which flies the
        # course 4/4 on seeds 1000-1003; reset()ing with the same seed reproduces that aligned
        # spawn so the co-sim threads all gates (the model is a forward-biased gate-follower —
        # a random unaligned spawn drifts). Matches eval_fused_warehouse.py's torch.manual_seed.
        self._seed = int(os.environ.get("ROSE_WH_SEED", "1000"))

        TASK_ID = "Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0"
        env_cfg = WarehouseNavEnvCfg_PLAY_WithSensors()
        env_cfg.scene.num_envs = 1
        env_cfg.curriculum.obstacle_count.params["min_level"] = int(os.environ.get("ROSE_WH_OBST", "6"))
        env_cfg.episode_length_s = float(os.environ.get("ROSE_WH_EPLEN", str(env_cfg.episode_length_s)))
        print(f"[WarehouseFusedNav] gym.make {TASK_ID}  freeze={self._freeze} "
              f"obst={env_cfg.curriculum.obstacle_count.params['min_level']}", flush=True)
        self._env = RslRlVecEnvWrapper(gym.make(TASK_ID, cfg=env_cfg))
        self._uenv = self._env.unwrapped
        self._uenv.sim._disable_app_control_on_stop_handle = True
        self._dev = self._uenv.device
        self._N = self._uenv.num_envs
        self._control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
        self._est = StateEstimator(self._N, self._dev, control_dt=self._control_dt)
        self._robot = self._uenv.scene["robot"]
        self._origin = self._uenv.scene.env_origins
        self._gate_centers = np.asarray([g[0][:2] for g in GW.FUSED_GATES], dtype=np.float64)
        self._pass_radius = GW.FixedGateCourseCommandCfg().success_radius
        self._K = len(self._gate_centers)

        self.observation_space = spaces.Dict({
            "front":  spaces.Box(-128, 127, (FRONT_H * FRONT_W,), np.int8),
            "tof":    spaces.Box(-128, 127, (256,), np.int8),
            "lowdim": spaces.Box(-np.inf, np.inf, (21,), np.float32),
        })
        self.action_space = spaces.Box(-10.0, 10.0, (2,), np.float32)

        # ---- flight state / control-tick bookkeeping ----
        self._goal_idx = 0
        self._gates_passed = 0
        self._tick = 0            # control ticks (physics steps) since reset
        self._ep = 0
        self._pending_cmds = 0    # incremented by on_action_received(); one physics tick each
        self._cmd_count = 0       # cumulative 0x20 commands received (whole run)
        self._phys_ticks = 0      # cumulative physics ticks (whole run) — must track _cmd_count
        self._last_cmd = (0.0, 0.0)
        self._frozen_obs = None
        self._done_logged = False

        # ---- video (decimated chase-cam JPEGs; robust to the sync's os._exit) + trajectory ----
        self._cam_on = os.environ.get("ROSE_ISAAC_CAMERA", "") == "1"
        self._video_dir = os.environ.get("ROSE_VIDEO_DIR", "")   # follow chase-cam frames
        self._fpv_dir = os.environ.get("ROSE_FPV_DIR", "")       # front-camera (FPV) frames
        self._video_decim = int(os.environ.get("ROSE_VIDEO_DECIM", "2"))  # 50Hz/2 = 25 fps
        self._vframe = 0
        self._fpvframe = 0
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
            self._traj_f.write("tick,ep,t,x,y,z,roll,pitch,yaw,yr_cmd,fwd_cmd,goal_idx,gates_passed\n")
            self._traj_f.flush()

    # ---- synchronizer hook: a fresh 0x20 command was received this loop iteration ----
    # Counter (not a bool) so no command is dropped if two were ever to land between steps —
    # guarantees EXACTLY one physics tick per received guest command (faithful to pytorch).
    def on_action_received(self):
        self._pending_cmds += 1
        self._cmd_count += 1

    # ---- pose helpers ----
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
        torch = self._torch
        xy = self._xy_local()
        goal_xy = self._gate_centers[min(self._goal_idx, self._K - 1)]
        if np.linalg.norm(xy - goal_xy) < self._pass_radius:   # reached the current goal gate
            if self._goal_idx < self._K - 1:
                self._goal_idx += 1
                print(f"[GATE] passed gate {self._goal_idx}/{self._K} at t="
                      f"{self._tick*self._control_dt:.2f}s xy=({xy[0]:+.2f},{xy[1]:+.2f})", flush=True)
                goal_xy = self._gate_centers[self._goal_idx]
            else:                                               # within radius of the LAST gate
                self._gates_passed = self._K
        self._gates_passed = max(self._gates_passed, self._goal_idx)
        dvec = goal_xy - xy
        cy, sy = math.cos(-yaw_now), math.sin(-yaw_now)
        bx, by = cy * dvec[0] - sy * dvec[1], sy * dvec[0] + cy * dvec[1]
        nrm = max((bx * bx + by * by) ** 0.5, 1e-6)
        return torch.tensor([[bx / nrm * self.base_speed, by / nrm * self.base_speed, 0.0]],
                            device=self._dev, dtype=torch.float32).repeat(self._N, 1)

    def _sense(self, desired_vel):
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
        return {"front": front, "tof": tof, "lowdim": lowdim}

    def _cmd_to_action(self, yr, fwd, h):
        torch = self._torch
        a0 = float(fwd) / (self.MAX_SPEED / 2.0) - 1.0
        a2 = float(yr) / self.MAX_YAWRATE
        speed = max(0.05, a0 + 1.0)
        vz_des = max(-self.VZ_MAX, min(self.VZ_MAX, self.K_ALT * (self.TARGET_H - float(h))))
        s = max(-1.0, min(1.0, vz_des / speed))
        a1 = math.asin(s) / self.MAX_INCL
        return torch.tensor([[a0, a1, a2, 0.0]], device=self._dev, dtype=torch.float32).clamp(-1.0, 1.0).repeat(self._N, 1)

    def _drive_chase(self):
        """Reposition the world-anchored chase camera to FOLLOW the drone each tick: sit behind
        (opposite heading) + above, look ahead at the drone. Mirrors eval_fused_warehouse.py's
        _drive_chase so the drone + gates stay framed (fixes the 'flies away to a speck' video)."""
        if self._chase is None:
            return
        torch = self._torch
        p = self._robot.data.root_pos_w[0]
        q = self._robot.data.root_quat_w[0]
        w, x, y, z = q[0], q[1], q[2], q[3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fx, fy = torch.cos(yaw), torch.sin(yaw)
        eye = torch.stack([p[0] - fx * 3.0, p[1] - fy * 3.0, p[2] + 1.6])
        tgt = torch.stack([p[0] + fx * 4.0, p[1] + fy * 4.0, p[2] - 0.1])
        self._chase.set_world_poses_from_view(eye.unsqueeze(0), tgt.unsqueeze(0))

    def _grab_rgb(self, cam):
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _capture_video(self):
        if not self._cam_on or (self._tick % self._video_decim) != 0:
            return
        try:
            import imageio
            if self._video_dir and self._chase is not None:
                rgb = self._grab_rgb(self._chase)[::2, ::2]     # 960x540 -> 480x270
                imageio.imwrite(os.path.join(self._video_dir, f"f_{self._vframe:05d}.jpg"), rgb)
                self._vframe += 1
            if self._fpv_dir and self._front is not None:
                fpv = self._grab_rgb(self._front)               # front-cam RGB = the FPV the model flies on
                imageio.imwrite(os.path.join(self._fpv_dir, f"f_{self._fpvframe:05d}.jpg"), fpv)
                self._fpvframe += 1
        except Exception as e:
            if self._vframe == 0 and self._fpvframe == 0:
                print(f"[WarehouseFusedNav] video capture disabled ({e})", flush=True)
            self._cam_on = False

    def _log_traj(self, z, roll, pitch, yaw, yr, fwd):
        if self._traj_f is None:
            return
        xy = self._xy_local()
        self._traj_f.write(f"{self._tick},{self._ep},{self._tick*self._control_dt:.4f},"
                           f"{xy[0]:.4f},{xy[1]:.4f},{z:.4f},{roll:.4f},{pitch:.4f},{yaw:.4f},"
                           f"{yr:.5f},{fwd:.5f},{self._goal_idx},{self._gates_passed}\n")
        if (self._tick % 10) == 0:
            self._traj_f.flush()

    def _refresh_obs(self):
        _, _, yaw = self._rpy()
        self._frozen_obs = self._prequantize(self._sense(self._desired_vel(yaw)))

    def _apply_and_step(self, yr, fwd):
        """One control tick: apply the guest command through cmd_to_action + inner step, then
        recompute the (frozen) sense snapshot and log flight state."""
        h = float(self._robot.data.root_pos_w[0, 2].item())
        if self._cam_on:
            self._drive_chase()   # aim the follow-cam BEFORE the step so its render is fresh
        _obs, _r, dones, _i = self._env.step(self._cmd_to_action(yr, fwd, h))
        self._tick += 1
        self._phys_ticks += 1
        self._last_cmd = (yr, fwd)
        z = float(self._robot.data.root_pos_w[0, 2].item())
        roll, pitch, yaw = self._rpy()
        self._capture_video()
        self._log_traj(z, roll, pitch, yaw, yr, fwd)
        if (self._tick % 10) == 0:  # ~2.5/sec at 50 Hz
            xy = self._xy_local()
            print(f"[flight] ep={self._ep} t={self._tick*self._control_dt:5.2f}s xy=({xy[0]:+.2f},{xy[1]:+.2f}) "
                  f"z={z:.2f} roll={math.degrees(roll):+5.1f} pitch={math.degrees(pitch):+5.1f} "
                  f"yr={yr:+.3f} fwd={fwd:.3f} goal={self._goal_idx}/{self._K} gates={self._gates_passed} "
                  f"[cmds={self._cmd_count} ticks={self._phys_ticks}]",  # cadence: must stay 1:1
                  flush=True)
        if self._gates_passed >= self._K:
            print(f"[GATE] *** ALL {self._K} GATES PASSED *** ep={self._ep} t={self._tick*self._control_dt:.2f}s", flush=True)
            self._finish_episode("success")
            self._reset_episode()
            return
        done = bool(dones[0].item()) if hasattr(dones, "__getitem__") else bool(dones)
        if done:
            outcome = "timeout"
            try:
                tm = self._uenv.termination_manager
                terms = {nm: bool(tm.get_term(nm)[0].item()) for nm in tm.active_terms}
                if terms.get("time_out"):
                    outcome = "timeout"
                elif any(("collision" in nm or "crash" in nm) and v for nm, v in terms.items()) or z < 0.2:
                    outcome = "crash"
            except Exception:
                pass
            self._finish_episode(outcome)
            self._reset_episode()
            return
        self._refresh_obs()

    def _finish_episode(self, outcome):
        if not self._done_logged:
            print(f"[EPISODE] ep={self._ep} outcome={outcome} gates_passed={self._gates_passed}/{self._K} "
                  f"ticks={self._tick} ({self._tick*self._control_dt:.2f}s)", flush=True)
            self._done_logged = True

    def _reset_episode(self):
        self._torch.manual_seed(self._seed)   # deterministic aligned respawn
        if hasattr(self._est, "reset"):
            self._est.reset()
        self._env.reset()
        self._ep += 1
        self._goal_idx = 0
        self._gates_passed = 0
        self._tick = 0
        self._done_logged = False
        self._pending_cmds = 0
        self._refresh_obs()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._torch.manual_seed(self._seed)   # fixed known-good aligned spawn (matches pytorch 4/4)
        if hasattr(self._est, "reset"):
            self._est.reset()
        self._env.reset()
        self._goal_idx = 0
        self._gates_passed = 0
        self._tick = 0
        self._done_logged = False
        self._pending_cmds = 0
        self._refresh_obs()
        return self._frozen_obs, {"goal_idx": self._goal_idx}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        yr, fwd = float(a[0]), float(max(0.1, min(self.MAX_SPEED, a[1])))
        if self._freeze:
            if self._pending_cmds > 0:     # a fresh 0x20 arrived → advance ONE coherent tick
                self._pending_cmds -= 1
                self._apply_and_step(yr, fwd)
            # else: hold the frozen snapshot (no physics) until the guest returns a command
        else:
            self._apply_and_step(yr, fwd)  # free-run: ZOH command, advance every env.step
        return self._frozen_obs, 0.0, False, False, {"goal_idx": self._goal_idx,
                                                     "gates_passed": self._gates_passed}

    def render(self):
        if self._chase is None:
            return np.zeros((16, 16, 3), np.uint8)
        rgb = self._chase.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)
