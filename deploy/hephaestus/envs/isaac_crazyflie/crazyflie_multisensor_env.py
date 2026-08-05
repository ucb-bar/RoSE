"""WIP — future multizone-ToF + FPV-camera Crazyflie env (NOT wired into the stress loop).

Scaffolding for the sensor set in `docs/ROSE_FUTURE_SENSORS_PLAN.md`:
  - 4× ST VL53L5CX multizone ToF (8×8, 45°×45° FoV, ≤4 m) facing FRONT/RIGHT/BACK/LEFT,
  - 1× Himax HM01B0 FPV mono camera (320×240, 8-bit).

It extends `IsaacCrazyflieSensorEnv` additively: it keeps that env's IMU/flow/downward-ToF
and *adds* the four horizontal zone grids + the FPV frame. It is deliberately **self-
contained and untested against the GPU** (the single GPU is busy with the stress plan):

  * The multizone ToF is synthesized **analytically** from ground-truth pose against a
    configurable axis-aligned "room" (+ optional obstacle boxes) — numpy only, no Isaac
    meshes, unit-testable via `python crazyflie_multisensor_env.py`. The production path
    (`build_raycaster_cfgs`) returns real `isaaclab` RayCaster configs for when a scene-
    extension hook lands (see the plan §4) — those are NOT instantiated here.
  * The FPV frame reuses the base env's camera handle as a stand-in if one exists, else it
    emits a zeroed frame with `fpv_valid=False`. A dedicated forward `CameraCfg`
    (`build_fpv_camera_cfg`) is provided for the production path.

Nothing here changes `crazyflie_sensor_env.py` / `crazyflie_mpc_env.py` / the active
configs. Registration is left commented in `register_envs.py`.
"""

import os

import numpy as np
from gymnasium import spaces

from .crazyflie_sensor_env import IsaacCrazyflieSensorEnv, _quat_to_matrix

# VL53L5CX specs (see plan §1.1)
TOF_MAX_RANGE = 4.0        # m
TOF_MIN_RANGE = 0.02       # m
TOF_FOV_DEG = 45.0         # square FoV (63° diagonal ≈ 45°×45°)
TOF_ZONES = 8              # 8×8 @ ~15 Hz (4 for 4×4 @ ~60 Hz)

# HM01B0 specs (see plan §2.1)
FPV_W, FPV_H = 320, 240    # QVGA mono
FPV_FOV_DEG = 70.0         # AI-deck stock lens ≈ 60–90° horizontal (modeled as a pinhole)
FPV_NEAR, FPV_FAR = 0.05, 8.0   # grayscale depth-shading range (m): near = bright

# Four horizontal mounts: (name, bore direction in body frame, yaw about +z in degrees).
# Body frame: +x forward, +y left, +z up.
TOF_MOUNTS = (
    ("front", (+1.0, 0.0, 0.0),   0.0),
    ("left",  (0.0, +1.0, 0.0),  90.0),
    ("back",  (-1.0, 0.0, 0.0), 180.0),
    ("right", (0.0, -1.0, 0.0), -90.0),
)


# --- Navigation environments: named sets of axis-aligned wall boxes (min_xyz, max_xyz) in
# world metres. The horizontal ToFs range against these (analytic ray-box) AND they are
# spawned as visible cuboids so co-sim videos show the environment. Selected by ROSE_MAZE.
def _maze_walls(name):
    name = (name or "").strip().lower()
    if name in ("", "none", "room"):
        return []
    if name == "hallway":
        # a corridor along +x: two long walls ~1.4 m apart (y = +-0.7), open ahead/behind.
        # Extents are env-overridable (ROSE_HALL_X0/X1) so a long corridor can be spawned for
        # a traversal that stays *through* the hall rather than reaching the open end.
        x0 = float(os.environ.get("ROSE_HALL_X0", -2.5))
        x1 = float(os.environ.get("ROSE_HALL_X1", 2.5))
        return [
            ((x0, 0.70, 0.0), (x1, 0.85, 1.8)),    # left wall
            ((x0, -0.85, 0.0), (x1, -0.70, 1.8)),  # right wall
        ]
    if name == "maze":
        # hallway + a chicane: a partial wall from the right leaving a gap on the left, then
        # a partial wall from the left -> the drone must weave.
        return [
            ((-2.5, 0.70, 0.0), (2.5, 0.85, 1.8)),    # left wall
            ((-2.5, -0.85, 0.0), (2.5, -0.70, 1.8)),  # right wall
            ((0.6, -0.70, 0.0), (0.75, 0.25, 1.8)),   # chicane 1 (from right, gap on left)
            ((1.6, -0.25, 0.0), (1.75, 0.70, 1.8)),   # chicane 2 (from left, gap on right)
        ]
    return []


def _yaw_matrix(deg):
    """Rotation about +z by `deg` degrees (body-frame yaw of a sensor mount)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _zone_ray_dirs(zones=TOF_ZONES, fov_deg=TOF_FOV_DEG):
    """(zones*zones, 3) unit ray directions in the SENSOR frame (bore = +x).

    az sweeps horizontally (about sensor +z), el vertically (about sensor +y); each spans
    ±fov/2. Row-major over (el, az) so the flattened grid matches the VL53L5CX zone order.
    """
    half = np.radians(fov_deg) / 2.0
    ang = np.linspace(-half, half, zones)
    dirs = np.empty((zones, zones, 3), dtype=np.float64)
    for i, el in enumerate(ang):          # rows: vertical
        for j, az in enumerate(ang):      # cols: horizontal
            dirs[i, j] = (np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el))
    dirs = dirs.reshape(-1, 3)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _aabb_exit_distance(o, d, box_min, box_max):
    """Distance from an INTERIOR point `o` to the AABB boundary along unit ray `d` (slab
    method). Returns +inf if degenerate. `o,d` are (N,3); box_* are (3,)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (box_min - o) * inv
        t2 = (box_max - o) * inv
        tmax = np.maximum(t1, t2)
    texit = np.min(np.where(np.isfinite(tmax), tmax, np.inf), axis=1)
    return np.where(texit > 0, texit, np.inf)


def _aabb_entry_distance(o, d, box_min, box_max):
    """Distance from `o` along unit ray `d` to ENTER an obstacle AABB, or +inf if missed.

    Slab method: per axis [near, far] = sorted (box-o)/d; entry tmin = MAX of nears, exit
    tmax = MIN of fars; hit iff tmin <= tmax and tmax >= 0. Axes parallel to a slab (d==0)
    constrain nothing when the origin is inside that slab, and force a miss when outside.
    (`o,d` are (N,3); box_* are (3,).)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (box_min - o) * inv
        t2 = (box_max - o) * inv
    near = np.minimum(t1, t2)
    far = np.maximum(t1, t2)
    parallel = (d == 0.0)
    inside = (o >= box_min) & (o <= box_max)
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), near)
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), far)
    tmin = np.max(near, axis=1)   # entry (max of per-axis nears)
    tmax = np.min(far, axis=1)    # exit  (min of per-axis fars)
    hit = (tmax >= np.maximum(tmin, 0.0))
    return np.where(hit & (tmin > 0), tmin, np.inf)


def _pixel_ray_dirs(width, height, fov_h_deg=FPV_FOV_DEG):
    """(height*width, 3) unit ray dirs in the CAMERA/body frame (bore = +x, +y left, +z up).

    Forward-facing pinhole: az sweeps horizontally (about +z), el vertically (about +y), with
    square pixels (vertical FoV scaled by height/width). Row-major over (row, col) so pixel
    (i, j) -> index i*width + j, matching an image raster. Column 0 = image left = +y (az>0),
    row 0 = image top = +z (el>0). This mirrors _zone_ray_dirs so the same ray/AABB machinery
    that synthesizes the multizone ToF also renders the FPV frame."""
    fov_h = np.radians(fov_h_deg)
    fov_v = fov_h * (float(height) / float(width))
    az = np.linspace(+fov_h / 2.0, -fov_h / 2.0, width)     # cols: left(+y) -> right(-y)
    el = np.linspace(+fov_v / 2.0, -fov_v / 2.0, height)    # rows: top(+z) -> bottom(-z)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    dirs = np.empty((height, width, 3), dtype=np.float64)
    dirs[..., 0] = np.outer(ce, ca)     # x = cos(el)cos(az)  (forward)
    dirs[..., 1] = np.outer(ce, sa)     # y = cos(el)sin(az)  (left/right)
    dirs[..., 2] = np.outer(se, np.ones_like(ca))  # z = sin(el) (up/down)
    dirs = dirs.reshape(-1, 3)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def synth_fpv_analytic(pos, quat, room_min, room_max, obstacles, dirs_body,
                       width, height, near=FPV_NEAR, far=FPV_FAR, seg_len=0.30):
    """Render a legitimate depth-shaded grayscale FPV frame analytically (numpy, no GPU).

    Casts one pinhole ray per pixel (dirs_body, camera frame) from the drone pose into the
    same axis-aligned room + obstacle AABBs the multizone ToF ranges against, takes the nearest
    hit distance, and shades it: near walls bright, far dark (an 8-bit "depth FPV"). A light
    stripe modulation keyed to the world hit point mirrors the striped corridor walls spawned in
    _spawn_walls, so the frame carries recognizable, pose-dependent structure (parallax as the
    drone moves) rather than a flat gradient. Returns uint8 (height, width)."""
    R_bw = _quat_to_matrix(quat[0], quat[1], quat[2], quat[3])       # body->world
    d_world = dirs_body @ R_bw.T
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
    o = np.broadcast_to(np.asarray(pos, np.float64), d_world.shape)
    dist = _aabb_exit_distance(o, d_world, room_min, room_max)       # room interior -> walls
    for (bmin, bmax) in obstacles:
        dist = np.minimum(dist, _aabb_entry_distance(o, d_world, bmin, bmax))
    dist = np.clip(dist, near, far)
    # depth shade: near = bright (255), far = dark (0)
    shade = 255.0 * (1.0 - (dist - near) / (far - near))
    # stripe texture from the world hit point (parity of the summed horizontal coords), so the
    # walls show the same alternating banding as the spawned cuboids and it slides with motion.
    hit = o + dist[:, None] * d_world
    band = np.floor((hit[:, 0] + hit[:, 1]) / seg_len).astype(np.int64)
    shade = shade * np.where(band % 2 == 0, 1.0, 0.82)
    frame = np.clip(shade, 0.0, 255.0).astype(np.uint8).reshape(height, width)
    return frame


class IsaacCrazyflieMultiSensorEnv(IsaacCrazyflieSensorEnv):
    """WIP: adds 4× VL53L5CX zone grids + HM01B0 FPV to the sensor env (see module docstring)."""

    def __init__(self, *args, room=None, obstacles=None, tof_zones=TOF_ZONES,
                 tof_decimation=13, fpv_size=(FPV_W, FPV_H), **kwargs):
        super().__init__(*args, **kwargs)
        # Analytic "room": interior AABB the horizontal ToFs range against. Default a
        # 4×4 m room, 2.5 m tall, centered on origin. Env-var overridable for sweeps.
        rx = float(os.environ.get("ROSE_TOF_ROOM_X", "2.0"))
        ry = float(os.environ.get("ROSE_TOF_ROOM_Y", "2.0"))
        rz = float(os.environ.get("ROSE_TOF_ROOM_Z", "2.5"))
        self._room_min = np.array(room[0] if room else (-rx, -ry, 0.0), dtype=np.float64)
        self._room_max = np.array(room[1] if room else (rx, ry, rz), dtype=np.float64)
        # Optional obstacle AABBs: list of (min_xyz, max_xyz). A named nav environment
        # (ROSE_MAZE=hallway|maze) adds wall boxes the horizontal ToFs range against; the same
        # boxes are spawned as visible cuboids in _spawn_walls so the co-sim video shows them.
        self._maze = os.environ.get("ROSE_MAZE", "")
        walls = [(np.asarray(a, np.float64), np.asarray(b, np.float64)) for a, b in _maze_walls(self._maze)]
        self._obstacles = walls + [(np.asarray(a, np.float64), np.asarray(b, np.float64))
                                   for a, b in (obstacles or [])]
        self._tof_zones = int(tof_zones)
        self._tof_ray_s = _zone_ray_dirs(self._tof_zones)     # sensor-frame rays
        self._tof_decim = max(1, int(os.environ.get("ROSE_MTOF_PERIOD", str(tof_decimation))))
        self._mtof_ctr = 0
        self._mtof_held = None
        # FPV frame size: env-overridable (ROSE_FPV_W/H) so validation can use a small,
        # fast-to-serialize frame while production keeps HM01B0 QVGA. Force W*H % 4 == 0 (the
        # synchronizer packs 4 grayscale bytes per uint32 DMA word).
        self._fpv_w = int(os.environ.get("ROSE_FPV_W", str(fpv_size[0])))
        self._fpv_h = int(os.environ.get("ROSE_FPV_H", str(fpv_size[1])))
        if (self._fpv_w * self._fpv_h) % 4 != 0:
            raise ValueError("FPV W*H must be a multiple of 4 (DMA word packing); got %dx%d"
                             % (self._fpv_w, self._fpv_h))
        self._fpv_fov = float(os.environ.get("ROSE_FPV_FOV", str(FPV_FOV_DEG)))
        # Cache the per-pixel camera-frame ray directions (pose-independent).
        self._fpv_dirs_body = _pixel_ray_dirs(self._fpv_w, self._fpv_h, self._fpv_fov)

        f32 = np.float32
        big = np.finfo(f32).max
        nz = self._tof_zones * self._tof_zones
        extra = {name: spaces.Box(0.0, TOF_MAX_RANGE, (nz,), f32)
                 for name, _, _ in TOF_MOUNTS}
        extra["fpv"] = spaces.Box(0, 255, (self._fpv_h * self._fpv_w,), np.uint8)
        # additive: keep all existing modality/pose keys, add the new ones
        self.observation_space = spaces.Dict({**self.observation_space.spaces, **extra})

    def _spawn_walls(self, sim_utils):
        """Spawn the named nav environment's walls as visible static cuboids in /World, so the
        co-sim video shows the maze/hallway. Geometry matches the analytic AABBs the ToFs range
        against (ROSE_MAZE). Read from the env var (not self._maze, which isn't set yet at the
        base-__init__ call site) so both derive from the same source."""
        walls = _maze_walls(os.environ.get("ROSE_MAZE", ""))
        if not walls:
            return
        # Split each wall into alternating-colour SEGMENTS along its long axis (a simple stripe
        # "texture") so the drone's motion relative to the corridor is obvious on camera.
        seg_len = 0.30
        stripe = [(0.66, 0.68, 0.74), (0.28, 0.31, 0.42)]   # light / dark slate stripes
        idx = 0
        for (mn, mx) in walls:
            mn = [float(v) for v in mn]
            mx = [float(v) for v in mx]
            ex, ey = mx[0] - mn[0], mx[1] - mn[1]
            axis = 0 if ex >= ey else 1                     # segment along the longer horizontal
            length = mx[axis] - mn[axis]
            nseg = max(1, int(round(length / seg_len)))
            for s in range(nseg):
                lo, hi = list(mn), list(mx)
                lo[axis] = mn[axis] + length * s / nseg
                hi[axis] = mn[axis] + length * (s + 1) / nseg
                size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
                center = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2]))
                cfg = sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=stripe[s % 2]),
                )
                cfg.func("/World/maze/wall_%d" % idx, cfg, translation=center)
                idx += 1

    # ---- analytic multizone ToF (numpy, no Isaac) --------------------------------------
    def _synth_multizone_tof(self, pos, quat):
        """Return {name: (zones*zones,) float32 distances in m} for the 4 horizontal sensors."""
        R_bw = _quat_to_matrix(quat[0], quat[1], quat[2], quat[3])   # body->world
        o = np.broadcast_to(np.asarray(pos, np.float64), (self._tof_ray_s.shape[0], 3))
        out = {}
        for name, _bore, yaw in TOF_MOUNTS:
            d_body = self._tof_ray_s @ _yaw_matrix(yaw).T            # sensor->body (rows are dirs)
            d_world = d_body @ R_bw.T                                # body->world
            d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
            dist = _aabb_exit_distance(o, d_world, self._room_min, self._room_max)
            for (bmin, bmax) in self._obstacles:
                dist = np.minimum(dist, _aabb_entry_distance(o, d_world, bmin, bmax))
            dist = np.clip(dist, TOF_MIN_RANGE, TOF_MAX_RANGE)
            out[name] = dist.astype(np.float32)
        return out

    # ---- FPV frame -----------------------------------------------------------------------
    def _synth_fpv(self, pos, quat):
        """Return (frame uint8 (H,W), valid bool) — the transport frame at (fpv_w, fpv_h).

        Two legitimate sources of forward-FPV camera sensing data:
          1. A real Isaac render from the body-mounted HM01B0 camera (ROSE_ISAAC_FPV=1):
             render_fpv_gray() returns the native-resolution 8-bit grayscale frame, resized to
             the transport size. This is a true first-person view with HM01B0 optics.
          2. Otherwise render the scene ANALYTICALLY from ground-truth pose (numpy pinhole ray
             cast against the room + obstacle AABBs) — GPU-free, the same path the multizone ToF
             uses; real pose-dependent scene data (not a placeholder). `valid` is True either way."""
        gray = self.render_fpv_gray() if getattr(self, "_fpv_isaac_on", False) else None
        if gray is not None and gray.size:
            return _resize_nn(gray, self._fpv_h, self._fpv_w), True
        frame = synth_fpv_analytic(pos, quat, self._room_min, self._room_max,
                                   self._obstacles, self._fpv_dirs_body,
                                   self._fpv_w, self._fpv_h)
        return frame, True

    def _obs_info(self, forces_z, action_norm):
        obs, info, gt = super()._obs_info(forces_z, action_norm)
        pos, quat = gt[0], gt[1]
        # low-rate: refresh the zone grids every _tof_decim control steps, hold in between
        if self._mtof_held is None or (self._mtof_ctr % self._tof_decim) == 0:
            self._mtof_held = self._synth_multizone_tof(pos, quat)
        self._mtof_ctr += 1
        for name, grid in self._mtof_held.items():
            obs[name] = grid
            info["mtof_%s" % name] = grid
        fpv, fpv_valid = self._synth_fpv(pos, quat)
        obs["fpv"] = fpv.reshape(-1)
        info["fpv_valid"] = np.bool_(fpv_valid)
        return obs, info, gt

    def reset(self, *, seed=None, options=None):
        self._mtof_ctr = 0
        self._mtof_held = None
        return super().reset(seed=seed, options=options)

    # ---- RoSE packing helpers (wire format for the future virtual drivers) --------------
    @staticmethod
    def tof_to_reqrsp_words(grid):
        """8×8 (or N×N) zone grid (m) -> flat float32 words, row-major (see plan §1.4)."""
        return np.asarray(grid, dtype=np.float32).reshape(-1)

    @staticmethod
    def fpv_to_dma_bytes(frame):
        """HM01B0 mono frame -> uint8 byte buffer for the RoSE DMA path (plan §2.3)."""
        return np.asarray(frame, dtype=np.uint8).reshape(-1)

    # ---- production IsaacLab configs (NOT instantiated here; for the scene hook) ---------
    @staticmethod
    def build_raycaster_cfgs(robot_prim="{ENV_REGEX_NS}/Robot/body",
                             mesh_prim_paths=("/World/ground",),
                             zones=TOF_ZONES, fov_deg=TOF_FOV_DEG,
                             max_range=TOF_MAX_RANGE):
        """Return {name: RayCasterCfg} for the 4 horizontal VL53L5CX (lazy isaaclab import).

        Each is a small lidar-style fan (zones×zones over fov×fov) oriented by a yaw
        quaternion, attached to the body, casting against `mesh_prim_paths` (add obstacle
        prims there). Use once the base env exposes a scene-extension hook (plan §4)."""
        from isaaclab.sensors import RayCasterCfg
        from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
        half = fov_deg / 2.0
        res = fov_deg / max(1, zones - 1)
        cfgs = {}
        for name, _bore, yaw in TOF_MOUNTS:
            a = np.radians(yaw) / 2.0
            quat = (float(np.cos(a)), 0.0, 0.0, float(np.sin(a)))   # (w,x,y,z) about +z
            cfgs[name] = RayCasterCfg(
                prim_path=robot_prim,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=quat),
                attach_yaw_only=False,
                pattern_cfg=LidarPatternCfg(
                    channels=zones,
                    vertical_fov_range=(-half, half),
                    horizontal_fov_range=(-half, half),
                    horizontal_res=res,
                ),
                max_distance=max_range,
                mesh_prim_paths=list(mesh_prim_paths),
            )
        return cfgs

    @staticmethod
    def build_fpv_camera_cfg(prim_path="{ENV_REGEX_NS}/Robot/body/hm01b0",
                             width=FPV_W, height=FPV_H, focal_length=2.0):
        """Forward-facing FPV CameraCfg (HM01B0-equivalent). rgb -> grayscale in _synth_fpv."""
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        return CameraCfg(
            prim_path=prim_path,
            update_period=0.0,
            width=width, height=height,
            data_types=["rgb"],
            # mount forward (+x) on the body; rot points the camera down its local axis
            offset=CameraCfg.OffsetCfg(pos=(0.03, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5),
                                       convention="ros"),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, focus_distance=400.0,
                horizontal_aperture=3.6, clipping_range=(0.02, 20.0),
            ),
        )


def _resize_nn(img, out_h, out_w):
    """Nearest-neighbor resize of a 2-D uint8 image (no cv2 dependency)."""
    h, w = img.shape[:2]
    ys = (np.arange(out_h) * h // out_h).clip(0, h - 1)
    xs = (np.arange(out_w) * w // out_w).clip(0, w - 1)
    return img[ys][:, xs]


# --- self-test: analytic ToF geometry, no Isaac import path exercised ---------------------
if __name__ == "__main__":
    zones = TOF_ZONES
    rays = _zone_ray_dirs(zones)
    assert rays.shape == (zones * zones, 3)
    assert abs(np.linalg.norm(rays, axis=1).max() - 1.0) < 1e-9

    # centered, level drone in a [-2,2]x[-2,2]x[0,2.5] room at z=1
    pos = np.array([0.0, 0.0, 1.0]); quat = np.array([1.0, 0.0, 0.0, 0.0])
    room_min = np.array([-2.0, -2.0, 0.0]); room_max = np.array([2.0, 2.0, 2.5])
    R = _quat_to_matrix(*quat)
    for name, _bore, yaw in TOF_MOUNTS:
        d_body = rays @ _yaw_matrix(yaw).T
        d_world = d_body @ R.T
        d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
        dist = np.clip(_aabb_exit_distance(np.broadcast_to(pos, d_world.shape),
                                           d_world, room_min, room_max),
                       TOF_MIN_RANGE, TOF_MAX_RANGE).reshape(zones, zones)
        # center zone (bore) should hit the facing wall at ~2.0 m; edge zones farther
        center = dist[zones // 2, zones // 2]
        print("%-6s center=%.3f m  min=%.3f  max=%.3f" % (name, center, dist.min(), dist.max()))
        assert 1.9 <= center <= 2.2, (name, center)
    print("OK: analytic multizone ToF geometry self-test passed (%dx%d zones)" % (zones, zones))
