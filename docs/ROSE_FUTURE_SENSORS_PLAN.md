# RoSE Future Sensors — VL53L5CX ×4 + HM01B0 (design + bring-up plan)

Design / integration plan for adding two **future** sensor modalities to the RoSE
Crazyflie/riskybird co-sim, extending the sim-to-real virtual-sensor pattern in
[`ROSE_SENSOR_ABSTRACTION.md`](ROSE_SENSOR_ABSTRACTION.md):

1. **4× ST VL53L5CX** multizone Time-of-Flight — horizontal obstacle sensing, one each
   facing **FRONT / RIGHT / BACK / LEFT**.
2. **1× Himax HM01B0** monochrome camera — forward-facing **FPV** (the Crazyflie AI-deck
   camera).

This is **future work**. It is intentionally *not* wired into the flight stress plan
([`ROSE_FLIGHT_STRESS_PLAN.md`](ROSE_FLIGHT_STRESS_PLAN.md)) — nothing here changes the
existing `IsaacCrazyflieSensorEnv-v0` loop. All new code lives in additive files (a new env
subclass + a new config) so `main`/the demo/the stress runs stay green.

---

## 0. What already exists (the pattern we extend)

- The virtual-sensor contract: the guest binds sensors through **standard Zephyr aliases**
  (`DT_ALIAS(...)`, `sensor_sample_fetch`/`sensor_channel_get`, or the video API); a **RoSE**
  build resolves those to `ucbbar,rose-*` drivers that fetch over the bridge, and a **real**
  build resolves them to the vendor driver — same app code (`ROSE_SENSOR_ABSTRACTION.md`).
- Existing downward ToF: env synth in `crazyflie_sensor_env.py:122-125` (held low-rate
  height), RoSE driver `soc/sw/zephyr-rose/drivers/sensor/rose_tof.c` (two-phase, decimated),
  binding `dts/bindings/sensor/ucbbar,rose-tof.yaml`, wired at cmd `0x14` in
  `config_gym_IsaacCrazyflieSensorEnv-v0.yaml`. **These new sensors follow the same shape.**
- **riskybird real driver present today (4× VL53L5CX already tested):**
  `zephyr-chipyard-sw@origin/riskybird-bringup`, `samples/riskybird/vl35l5cx_test/` (note the
  transposed spelling `vl35l5cx`) is a **working 4× VL53L5CX multizone bring-up**:
  `CONFIG_VL53L5CX=y`, four `compatible = "st,vl53l5cx"` nodes on `i2c0` at reprogrammed
  addresses `0x31–0x34` (aliases `tof0`–`tof3`), read via the standard
  `sensor_sample_fetch`/`sensor_channel_get(SENSOR_CHAN_DISTANCE)`. The parts boot at the
  shared `0x29`, so the sample uses an **ADS7128 I²C GPIO expander @0x17** to enable sensors
  one-at-a-time and readdress each (`vl53l5cx_readdress_bootstrap`, `VL53L5_BASE_NEW_ADDR_7BIT
  0x31`). So the 4-directional array — the hard parts (4 units, address collision, expander
  sequencing) — is **already solved in firmware**, not a fresh upgrade. (`riskybird/tof_sensor/`
  is a separate, earlier single-zone `st,vl53l0x` sample.) The RoSE virtual side just mirrors
  this: 4 aliases `tof0`–`tof3`, `st,vl53l5cx`-shaped, `SENSOR_CHAN_DISTANCE` (+ optional
  zone grid).
- The RoSE bridge has **2 reqrsp channels + 1 DMA channel** (`rose_adapter.c` header:
  `RoseTLRocketMMIOOnlyConfig`, `DMA_BUFFER(ch0)`, `rose_dma_arm/_wait/_buffer`). Small sensor
  frames use reqrsp; a **camera frame is large and must use the DMA path** (§2.3).

---

## 1. VL53L5CX ×4 (horizontal multizone ToF)

### 1.1 Datasheet specs (ST VL53L5CX, DS13754)
| Spec | Value |
|---|---|
| Zones (selectable) | **4×4 @ up to 60 Hz**, or **8×8 @ up to 15 Hz** |
| Field of view | **63° diagonal**, square (**≈45°×45°**); per-zone ≈ 45/N° (8×8 → ~5.6°, 4×4 → ~11.25°) |
| Ranging distance | ~2 cm … **400 cm** (up to 4 m; ~theoretical) |
| Resolution / units | 1 mm LSB, distance reported in **mm** per zone |
| Emitter | 940 nm invisible VCSEL, Class 1 |
| Per-zone output | distance (mm), target status (validity), reflectance, up to 4 targets/zone |
| Interface | I²C up to 1 MHz; firmware/ULD uploaded at init (~84 kB) |
| Supply / power | 3.3 V / 1.8 V IOVDD; ~few mW active |

Design choice: model **8×8 @ ~15 Hz** as the default (richest; the driver can decimate to
4×4/60 Hz). Each sensor → an **8×8 grid of distances in metres**, clipped to `[0.02, 4.0]`,
with a **no-return sentinel** (4.0 m / status-invalid) when a zone sees nothing.

### 1.2 Mounting (body frame)
Crazyflie/riskybird body frame: **+x forward, +y left, +z up**. The four sensors point
horizontally outward:

| Sensor | Bore direction (body) | Yaw about +z |
|---|---|---|
| front | +x | 0° |
| left  | +y | +90° |
| back  | −x | 180° |
| right | −y | −90° (270°) |

Each covers a 45°×45° cone about its bore; the four give ~180° horizontal coverage with
gaps at the 45° diagonals (a known VL53L5CX-array limitation — noted for planners).

### 1.3 Isaac synthesis — **ray-cast**, not camera-depth
Two options were considered:
- **RayCaster + a grid/lidar pattern** *(chosen)*: `isaaclab.sensors.RayCaster` with
  `LidarPatternCfg(horizontal_fov_range=(-22.5,22.5), vertical_fov_range=(-22.5,22.5),
  horizontal_res=45/8, channels=8)` (→ 8×8 rays) or a `GridPatternCfg`, attached to the
  robot body with `OffsetCfg(rot=<yaw quaternion>)` per direction, `max_distance=4.0`,
  casting against obstacle meshes (`mesh_prim_paths`). This is exactly how a real
  multizone ToF works (a fan of ranges) and is cheap (64 rays × 4). **Requires obstacles in
  the scene** — the current scene is ground + drone only, so horizontal rays return
  no-hit; §1.5 adds a configurable obstacle set.
- Camera depth (render `distance_to_image_plane`, downsample to 8×8): rejected — heavier
  (full render per sensor), and semantically a camera, not a ToF fan.

**Bring-up shortcut (implemented in the scaffold):** because RayCaster needs scene meshes
and we cannot boot the GPU here, the scaffold *also* provides an **analytic ray-box room
model** (`_synth_multizone_tof`) that computes the 8×8 zone distances from ground-truth
pose against a configurable axis-aligned "room" (4 walls + ceiling). This is fully
self-contained (numpy only), unit-testable without Isaac, and gives meaningful obstacle
ranges immediately; the RayCaster path (`build_raycaster_cfgs`) is the production route once
a scene-extension hook lands (§4).

### 1.4 RoSE wire + virtual driver
- **Packets** (reqrsp, multi-word, two-phase pipelined like the IMU): one command per
  sensor returning 64 float32 words (8×8) — metres, row-major:

  | Sensor | cmd | channel | words |
  |---|---|---|---|
  | tof_front | `0x30` | ch1 | 64 |
  | tof_right | `0x31` | ch1 | 64 |
  | tof_back  | `0x32` | ch2 | 64 |
  | tof_left  | `0x33` | ch2 | 64 |

  Split across the two reqrsp channels so front/back and left/right can be pipelined in
  parallel. **Low-rate**: decimate at the driver (200 Hz control / 15 Hz → ~13; or 4×4/60 Hz
  → ~3), same `tof_valid` multi-rate handling as the downward ToF.
- **Virtual driver** `ucbbar,rose-tof-zone`: mirrors `rose_tof.c` but collects `zones`
  words and exposes them. Real channel choices: expose the min-per-sensor as
  `SENSOR_CHAN_DISTANCE` (drop-in with the existing app), **plus** a private channel
  `ROSE_SENSOR_CHAN_TOF_ZONES` returning the full grid (matches how `st,vl53l5cx` exposes
  `VL53L5CX_ResultsData`). Binding props: `rose`, `rose-cmd`, `rose-channel`, `zones`
  (16/64), `decimation`, `direction` (front/right/back/left, for logging).
- **Real build:** `DT_ALIAS(tof0..3)` → `st,vl53l5cx` (Zephyr `drivers/sensor/st/vl53l5cx`,
  needs `hal_st` + the ST ULD blob; `CONFIG_VL53L5CX=y`) — **already done and tested** on
  `riskybird-bringup:samples/riskybird/vl35l5cx_test/`: four nodes at `0x31–0x34`, the shared
  `0x29` boot-address collision handled by an **ADS7128 GPIO expander @0x17** sequencing the
  sensors during `vl53l5cx_readdress_bootstrap`. So the sim-to-real story only needs the RoSE
  virtual driver to match those 4 aliases; the real firmware path exists.

### 1.5 Scene obstacles (for meaningful horizontal ranging)
Add a configurable set of static prims (walls of a box "room", or pillar obstacles) to the
scene so the horizontal ToFs and the FPV camera see structure. The scaffold's analytic
model uses the same room dimensions, so analytic and RayCaster paths agree.

---

## 2. HM01B0 (FPV monochrome camera)

### 2.1 Datasheet specs (Himax HM01B0)
| Spec | Value |
|---|---|
| Active resolution | **QVGA 320×240** (max); QQVGA 160×120 low-power mode. AI-deck commonly runs a **324×244** window |
| Colour | **Monochrome** (also Bayer variant), **8-bit** (4/6/8-bit selectable) |
| Frame rate | up to **60 fps** @ QVGA; higher at QQVGA |
| Optical format / pixel | **1/6″**, 3.6 µm pixel, rolling shutter |
| Interface | 1/4/8-bit parallel (DVP) + I²C/SCCB control (MIPI variant: HM01B0-MWA) |
| Power | ultra-low, ~**1.1 mW** QQVGA / ~2 mW QVGA |
| FoV | **lens-dependent** (not part of the sensor); AI-deck stock lens ≈ 60–90° horizontal — modeled as a configurable pinhole |
| Platform tie-in | the **Crazyflie AI-deck** camera (GAP8) — a real sim-to-real target |

Design choice: model **320×240, 8-bit grayscale, low-rate (~ configurable, e.g. 10–30 Hz)**.

### 2.2 Isaac synthesis — `Camera`
Use `isaaclab.sensors.Camera` (already used for the chase cam in `crazyflie_mpc_env.py:151-164`)
with a **body-mounted, forward-facing** `OffsetCfg` (pos ≈ (0.03, 0, 0) m on +x, rot facing
+x) and a `PinholeCameraCfg` whose `focal_length`/`horizontal_aperture` set the FoV to the
chosen lens. Render `rgb` (or `distance_to_image_plane` for a depth-FPV variant), then
**convert to 8-bit grayscale and resize to 320×240** to match HM01B0. (The scaffold reuses
the base env's camera handle as a stand-in if present, and otherwise emits a zeroed frame
with a TODO — a dedicated FPV `CameraCfg` needs the scene hook in §4.)

### 2.3 RoSE wire + virtual driver — **DMA path**
A 320×240×1 frame is **76,800 bytes** — far too large for a reqrsp packet. Use the RoSE
**DMA channel (ch0)**:
- **Guest**: `rose_dma_arm(dev, 0, nbytes)` → `rose_request(0x40)` (arm a frame) →
  `rose_dma_wait(dev, 0, timeout)` (blocks on the DMA-complete IRQ) → read
  `rose_dma_buffer(dev, 0)`. Mirrors `soc/sw/zephyr-rose/samples/dmavalidate`.
- **Env**: on cmd `0x40`, DMA the grayscale frame bytes into the guest buffer (the
  synchronizer's DMA route, same mechanism the camera-image RoSE demos use).
- **Virtual driver** `ucbbar,rose-camera`: exposes frames via the **Zephyr video API**
  (`video_dequeue`/`video_enqueue`) so a real build binds `DT_ALIAS(fpv)` to a `himax,hm01b0`
  video driver. **Note:** no upstream Zephyr HM01B0 driver exists today (user-confirmed) —
  writing the real `himax,hm01b0` video driver is a follow-up; the RoSE virtual driver +
  env synthesis can proceed and be validated first.

---

## 3. Metrics / use-cases these unlock (future)
- Horizontal ToFs → **obstacle-avoidance / wall-following** scenarios and a reactive layer
  above TinyMPC; a natural extension of the stress plan's "harder scenarios".
- FPV camera → vision-in-the-loop (the `xpu-rt/sims` DroNet pilot,
  `scripts/pilot/pilot_steering_with_camera.py`) running **on the SoC** over the RoSE DMA
  path — closing the perception→control loop in co-sim.

---

## 4. Remaining TODOs to fully wire (ordered)
1. **Scene hook (base env):** add a small, opt-in extension point to
   `crazyflie_mpc_env.py`'s scene builder so a subclass can inject extra sensors
   (RayCasters, FPV camera) and obstacle prims **without duplicating `__init__`**. Deferred
   here to avoid touching the file the stress loop uses; do it as its own change.
2. **RayCaster path:** replace the analytic room model with the `build_raycaster_cfgs()`
   sensors once the hook lands; add the obstacle set (§1.5).
3. **FPV camera:** add a dedicated forward `CameraCfg`; wire grayscale/resize in
   `_synth_fpv`.
4. **RoSE drivers:** `ucbbar,rose-tof-zone` (multizone, two-phase, decimated) + bindings;
   `ucbbar,rose-camera` (video API over DMA) + binding.
5. **Config:** new `config_gym_IsaacCrazyflieMultiSensorEnv-v0.yaml` with the `0x30..0x33`
   reqrsp packets and the `0x40` DMA camera route (a starter is included, WIP).
6. **Guest app:** consume the new sensors behind DT guards (like `HAVE_TOF`/`HAVE_FLOW`),
   keeping the shared-app sim-to-real story.
7. **Real drivers:** VL53L5CX ×4 is **already done** on riskybird
   (`samples/riskybird/vl35l5cx_test/`: `st,vl53l5cx` ×4 at `0x31–0x34` + ADS7128 expander) —
   reuse it; only the `himax,hm01b0` Zephyr video driver remains to author.

## Appendix — key file references
- Env base / sensor env (mirror): `deploy/hephaestus/envs/isaac_crazyflie/crazyflie_mpc_env.py`,
  `crazyflie_sensor_env.py`
- New scaffold (this work): `deploy/hephaestus/envs/isaac_crazyflie/crazyflie_multisensor_env.py`,
  `deploy/config/config_gym_IsaacCrazyflieMultiSensorEnv-v0.yaml` (WIP, not wired)
- RoSE ToF driver/binding to mirror: `soc/sw/zephyr-rose/drivers/sensor/rose_tof.c`,
  `dts/bindings/sensor/ucbbar,rose-tof.yaml`
- RoSE DMA path: `soc/sw/zephyr-rose/drivers/rose/rose_adapter.c`,
  `soc/sw/zephyr-rose/samples/dmavalidate/src/main.c`
- riskybird real ToF (4× VL53L5CX, tested): `zephyr-chipyard-sw@origin/riskybird-bringup:samples/riskybird/vl35l5cx_test/`
  (`st,vl53l5cx` ×4 @0x31–0x34, ADS7128 expander); earlier single-zone: `riskybird/tof_sensor/` (`st,vl53l0x`)
- IsaacLab sensors: `RayCaster`/`LidarPatternCfg`/`GridPatternCfg`, `Camera`/`PinholeCameraCfg`
  (installed `isaaclab==0.54.3`)
