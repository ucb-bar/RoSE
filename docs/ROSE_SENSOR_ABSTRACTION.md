# RoSE Sensor Abstraction: a virtual Zephyr sensor driver for co-sim / real parity

**Status:** design proposal
**Author:** (design doc — no production code)
**Date:** 2026-08-02

## 1. Motivation

Today RoSE guest software talks to the co-sim bridge with **raw packets**. The
"most real" guest, `samples/rose_flight_controller`, hard-codes the RoSE command
IDs and reqrsp channels and marshals `float32` words itself:

- `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose_flight_controller/src/main.cpp:42-46`
  defines `ROSE_CMD_IMU 0x12` on `ch2`, `ROSE_CMD_FLOW 0x13` on `ch1`,
  `ROSE_CMD_CONTROL 0x20`.
- `main.cpp:114-127` (`recv_sensor`) calls `rose_request()` + `rose_recv_reqrsp()`
  and `memcpy`s raw words into floats.
- `main.cpp:164-171` drives the loop by calling `recv_sensor(ROSE_CMD_IMU, ...)`
  and `recv_sensor(ROSE_CMD_FLOW, ...)`.

Meanwhile the **real-target** reference, `samples/flight_controller`, never sees a
packet. It uses only the standard Zephyr sensor API:

- `samples/flight_controller/src/attitude_estimator.c:16-28` —
  `DEVICE_DT_GET(DT_ALIAS(bmi088_accel))` / `DT_ALIAS(bmi088_gyro)`.
- `attitude_estimator.c:104-131` — `sensor_sample_fetch(accel_dev)` +
  `sensor_channel_get(accel_dev, SENSOR_CHAN_ACCEL_XYZ, accel)` (and
  `SENSOR_CHAN_GYRO_XYZ`).
- `samples/flight_controller/prj.conf:1-3` — `CONFIG_SENSOR=y`, `CONFIG_BMI08X=y`,
  `CONFIG_I2C=y` (the in-tree Bosch bmi08x driver family covering the BMI088).
- The same pattern in `samples/bmi088_test/src/main.c:56-92`
  (`SENSOR_CHAN_ACCEL_XYZ`, `SENSOR_CHAN_GYRO_XYZ`, `SENSOR_CHAN_DIE_TEMP`).

**Goal:** make the application (and ideally the estimator/control) code *identical*
across the real and RoSE builds — it always talks to a named Zephyr sensor device
via `sensor_sample_fetch`/`sensor_channel_get`. Only the **devicetree binding**
differs: on real hardware the alias resolves to `bosch,bmi08x-*`; in RoSE it
resolves to a **virtual driver** (`ucbbar,rose-bmi088`) that fetches the sensor's
data over the RoSE bridge.

The infrastructure to do this is *already anticipated in the codebase*. The
protocol-layer header says so explicitly:

> "Higher layers (a rose-i2c bus controller, or a rose sensor driver exposing
> sensor.h) marshal through this so the same user-facing API works in co-sim or on
> real hardware — only the bound devicetree node/bus differs."
> — `soc/sw/zephyr-rose/include/rose/rose_proto.h:9-13`

and the Kconfig help mirrors it (`soc/sw/zephyr-rose/subsys/rose/Kconfig`, the
`ROSE_PROTOCOL` entry: "Used by higher-level shims (rose-i2c bus, rose sensor
drivers)…"). This document specifies that shim.

## 2. What exists today (ground truth from the code)

### 2.1 The RoSE module (`soc/sw/zephyr-rose/`)

A proper Zephyr module (`zephyr/module.yml`: `name: rose`, `dts_root: .`,
`kconfig: Kconfig`) with three layers:

| Layer | File(s) | Role |
|---|---|---|
| Adapter driver | `drivers/rose/rose_adapter.c` (`DT_DRV_COMPAT ucbbar_roseadapter`) | Device-model API over the bridge MMIO FIFOs: `rose_tx/rose_rx/rose_tx_ready/rose_rx_ready` + camera-DMA `dma_arm/dma_wait/dma_buffer`. |
| Adapter API | `include/rose/rose.h` | `struct rose_driver_api` + inline wrappers (`rose_tx`, `rose_rx`, …). Channel map: ch1→reqrsp0, ch2→reqrsp1, ch0→DMA (`rose.h:26-51`). |
| Protocol/transport | `subsys/rose/rose_proto.c`, `include/rose/rose_proto.h` | `rose_request(dev,cmd,arg)`, `rose_recv_reqrsp(dev,ch,out,max_words)`, `rose_recv_dma(...)`. Framing: `[header(cmd echo)][num_bytes][data…]` (`rose_proto.c:26-63`). |

Binding: `dts/bindings/misc/ucbbar,RoseAdapter.yaml` (`compatible
"ucbbar,RoseAdapter"`, props `num-reqrsp-channels`, `num-dma-channels`,
`dma-base-address`).

The adapter node is provided per-app by a board overlay, e.g.
`samples/rose_flight_controller/boards/spike_riscv64.overlay:5-14`
(`rose@2000`, 2 reqrsp + 1 DMA, DMA base `0x88000000`, PLIC IRQ 3). Every
`samples/rose/*` app ships the identical overlay.

The current guest gets the adapter device with
`DEVICE_DT_GET_ONE(ucbbar_roseadapter)` (`main.cpp:61`).

### 2.2 Build flow (`soc/sim/build_zephyr_rose.sh`)

- Builds with `west build -p always -b spike_riscv64 <src> --
  -DZEPHYR_EXTRA_MODULES=$ROSE_DIR/soc/sw/zephyr-rose`
  (`build_zephyr_rose.sh:64-65`).
- App source is resolved from `samples/rose/<app>` **or** top-level
  `samples/<app>` (`build_zephyr_rose.sh:57-62`) — this is why
  `rose_flight_controller` (which lives at `samples/rose_flight_controller`)
  builds.
- Output: `soc/sim/zephyr_rose_builds/<app>/zephyr/zephyr.elf`, consumed by
  `run_spike_rose*.sh` and FireSim.

Key point: the RoSE virtual driver we add lives **inside the `zephyr-rose`
module**, so it is automatically available to any app built through this flow with
no per-app CMake changes — the app only needs the right overlay + Kconfig.

### 2.3 The estimator/control code is already sensor-shaped

`samples/rose_flight_controller/src/estimator.hpp:25-47` defines
`IStateEstimator` consuming exactly `accel[3]`, `gyro[3]`, `flow[2]`, `height`,
`tof_valid`, `dt`. That interface is agnostic to *how* the data arrived — it does
not care whether it came from a packet or a sensor device. So the estimator and
TinyMPC (`main.cpp:72-111,191-213`) are already reusable as-is; only the
acquisition front-end (`recv_sensor`) needs to change.

## 3. Real-sensor reference pattern (the thing we emulate)

**Finding from a full repo search:** there are **no `esp32` boards** anywhere in
this checkout (no board dir, overlay, `prj.conf`, or west-manifest entry), and
**no real sensor is actually bound on disk** — there is no `.overlay`/`.dts` that
instantiates a `bosch,bmi08x-*`, `st,vl53l*`, or `pmw3901` node, and the bmi08x
driver source + binding YAMLs are **not vendored here**. They live in the
`ucb-bar/zephyr` fork submodule, which is **uninitialized**:
`.gitmodules` maps `zephyr_ws/zephyr` → `https://github.com/ucb-bar/zephyr.git`
(pinned at commit `852bb170cc5655577e9cb3a673e69e2e0c3b00c5`); the directory
`soc/sw/xpu-rt/zephyr-chipyard-sw/zephyr_ws/zephyr` is empty. So the concrete
`SENSOR_DEVICE_DT_INST_DEFINE` and `bosch,bmi08x-accel`/`-gyro` bindings must be
read from that fork (or upstream Zephyr `drivers/sensor/bosch/bmi08x/` +
`dts/bindings/sensor/bosch,bmi08x-*.yaml`) — they are cited here from the standard
Zephyr driver model.

The **on-repo** reference for *how a real sensor is consumed* is therefore the two
sample apps (`bmi088_test`, `flight_controller`) plus the I2C bus helper
`samples/i2c_scanner` (`src/main.c`: `DT_NODELABEL(i2c0)` +
`i2c_write(...)` address probe — the bus a real BMI088 would hang off). Neither
sensor sample ships a `boards/*.overlay`; they assume the target board's DT already
declares the `bmi088_accel`/`bmi088_gyro` nodes and aliases (no board in this repo
does), which is the gap the RoSE overlay in §4.2 fills for the co-sim target.

The upstream Zephyr Bosch **bmi08x** driver (enabled by `CONFIG_BMI08X`,
`prj.conf:2`) registers two child sensor devices via the standard sensor model.
The device-tree contract the app depends on is only the **aliases**
`bmi088-accel` / `bmi088-gyro` (`bmi088_test/src/main.c:10-18`), each pointing at
a node whose `compatible` is `bosch,bmi08x-accel` / `bosch,bmi08x-gyro`
(BMI088 variant selected by the driver). A real overlay therefore looks like:

```dts
&i2c0 {
    status = "okay";
    bmi088_accel: bmi088-accel@18 {
        compatible = "bosch,bmi08x-accel";
        reg = <0x18>;
        /* accel-hz / accel-fs / int pins per board */
    };
    bmi088_gyro: bmi088-gyro@68 {
        compatible = "bosch,bmi08x-gyro";
        reg = <0x68>;
    };
};
/ {
    aliases {
        bmi088-accel = &bmi088_accel;
        bmi088-gyro  = &bmi088_gyro;
    };
};
```

The **application never references the compatible or the bus** — only
`DT_ALIAS(bmi088_accel)`. That stable alias is the seam we exploit.

## 4. Proposed architecture

### 4.1 Layering

```
   application (main / estimator / control)         <-- IDENTICAL both targets
        |  sensor_sample_fetch(), sensor_channel_get()
        v
   Zephyr sensor subsystem (struct sensor_driver_api)
        |
   +----+--------------------------+
   |                               |
 REAL build                     RoSE build
 bosch,bmi08x-accel/-gyro       ucbbar,rose-bmi088-accel/-gyro   <-- NEW virtual driver
   |  I2C/SPI xfer                  |  rose_request()/rose_recv_reqrsp()
   v                               v
 BMI088 silicon                 rose adapter driver (rose.h) -> RoSE bridge -> synchronizer/sim
```

We add, inside `soc/sw/zephyr-rose/drivers/sensor/`, **one virtual Zephyr sensor
driver per virtual sensor type** the drone needs:

1. `rose_bmi088` — 6-axis IMU (accel + gyro), two logical devices exposing
   `SENSOR_CHAN_ACCEL_XYZ` and `SENSOR_CHAN_GYRO_XYZ` (mirrors the real bmi08x's
   two-child structure so the aliases line up 1:1).
2. `rose_flow_deck` — optical-flow (2-axis body velocity) + downward ToF height,
   modeling the Crazyflie Flow deck v2 (`rose_flight_controller/README.rst:17-18`).

Each driver:

- declares its own `compatible` (`ucbbar,rose-bmi088-accel`,
  `ucbbar,rose-bmi088-gyro`, `ucbbar,rose-flow`, `ucbbar,rose-tof`) with a binding
  yaml under `soc/sw/zephyr-rose/dts/bindings/sensor/`;
- takes a **phandle to the rose adapter node** plus a **`rose-cmd`** (the reqrsp
  command ID) and **`rose-channel`** (which reqrsp RX channel the response lands
  on) as DT properties — this moves today's hard-coded `0x12/ch2`, `0x13/ch1`
  (`main.cpp:42-46`) into devicetree;
- implements `sensor_driver_api` (`sample_fetch` + `channel_get`) on top of
  `rose_request()` / `rose_recv_reqrsp()`;
- is registered with `SENSOR_DEVICE_DT_INST_DEFINE(...)`.

### 4.2 Devicetree strategy (the binding seam)

The app depends on **aliases only**. We keep the alias names identical
(`bmi088-accel`, `bmi088-gyro`, and new `flow`, `tof`) and swap what they point at
per build:

- **Real** overlay: alias → `bosch,bmi08x-*` node on `&i2c0/&spi0` (§3).
- **RoSE** overlay (`boards/spike_riscv64.overlay`): keep the existing `rose@2000`
  adapter node, then add virtual sensor nodes as children/siblings that reference
  it, and point the same aliases at them:

```dts
/ {
    rose0: rose@2000 { /* unchanged, from the shared overlay */
        compatible = "ucbbar,RoseAdapter";
        reg = <0x2000 0x1000>;
        num-reqrsp-channels = <2>;
        num-dma-channels = <1>;
        dma-base-address = <0x88000000>;
        status = "okay";
        /* ... */
    };

    rose_imu_accel: rose-bmi088-accel {
        compatible = "ucbbar,rose-bmi088-accel";
        rose = <&rose0>;
        rose-cmd = <0x12>;      /* ROSE_CMD_IMU  */
        rose-channel = <2>;     /* reqrsp1       */
        status = "okay";
    };
    rose_imu_gyro: rose-bmi088-gyro {
        compatible = "ucbbar,rose-bmi088-gyro";
        rose = <&rose0>;
        rose-cmd = <0x12>;      /* same packet carries accel+gyro */
        rose-channel = <2>;
        status = "okay";
    };
    rose_flow: rose-flow {
        compatible = "ucbbar,rose-flow";
        rose = <&rose0>;
        rose-cmd = <0x13>;      /* ROSE_CMD_FLOW */
        rose-channel = <1>;     /* reqrsp0       */
        status = "okay";
    };
    rose_tof: rose-tof {
        compatible = "ucbbar,rose-tof";
        rose = <&rose0>;
        rose-cmd = <0x13>;      /* shares the FLOW packet (vx,vy,h) */
        rose-channel = <1>;
        status = "okay";
    };

    aliases {
        bmi088-accel = &rose_imu_accel;
        bmi088-gyro  = &rose_imu_gyro;
        flow         = &rose_flow;
        tof          = &rose_tof;
    };
};
```

`DT_ALIAS(bmi088_accel)` resolves identically in both builds; `DEVICE_DT_GET`
returns whichever driver was compiled in. **No `#ifdef` in application code.**

### 4.3 Kconfig

Add to `soc/sw/zephyr-rose/drivers/Kconfig` (per the existing rsource layout):

- `ROSE_SENSOR_BMI088` (`depends on SENSOR && ROSE_PROTOCOL`,
  `default y if DT_HAS_UCBBAR_ROSE_BMI088_ACCEL_ENABLED`).
- `ROSE_SENSOR_FLOW` / `ROSE_SENSOR_TOF` similarly.

The RoSE build's `prj.conf` then mirrors the real one but swaps the driver family:

```conf
CONFIG_SENSOR=y            # same as real
CONFIG_ROSE=y              # bridge adapter (already in rose apps)
CONFIG_ROSE_PROTOCOL=y     # reqrsp framing
CONFIG_ROSE_SENSOR_BMI088=y
CONFIG_ROSE_SENSOR_FLOW=y
# (no CONFIG_BMI08X / CONFIG_I2C on the RoSE target)
```

Real build (`samples/flight_controller/prj.conf:1-3`) is unchanged:
`CONFIG_SENSOR=y`, `CONFIG_BMI08X=y`, `CONFIG_I2C=y`.

## 5. Code-reuse / build strategy

| Component | Real target | RoSE target | Shared? |
|---|---|---|---|
| `main` / control loop | ✓ | ✓ | **shared** — sensor API only |
| estimator (`IStateEstimator`, EKF/complementary) | ✓ | ✓ | **shared** (`estimator*.cpp/hpp`) |
| TinyMPC solver | ✓ | ✓ | **shared** |
| sensor acquisition | `bosch,bmi08x` driver | `ucbbar,rose-*` virtual driver | per-target (driver binding) |
| devicetree overlay | I2C/SPI sensor nodes | `rose@2000` + virtual sensor nodes | per-target (overlay) |
| Kconfig (`prj.conf`) | `CONFIG_BMI08X`,`CONFIG_I2C` | `CONFIG_ROSE*`,`CONFIG_ROSE_SENSOR*` | per-target |

The virtual drivers live in the `zephyr-rose` module, so
`build_zephyr_rose.sh <app>` (which already injects
`-DZEPHYR_EXTRA_MODULES=.../zephyr-rose`, `build_zephyr_rose.sh:64-65`) picks them
up automatically. No change to the build script is required. A single application
directory can serve both targets using per-board overlays
(`boards/spike_riscv64.overlay` for RoSE; `boards/<real-board>.overlay` for
hardware) — exactly the Zephyr board-overlay mechanism the samples already use.

## 6. Worked example — BMI088 IMU

### 6.1 Application snippet (identical for both targets)

This is verbatim the existing real-target code
(`attitude_estimator.c:16-28,104-131`); it needs **no changes** to run on RoSE
once the overlay above is in place:

```c
#define BMI088_ACCEL_NODE DT_ALIAS(bmi088_accel)
#define BMI088_GYRO_NODE  DT_ALIAS(bmi088_gyro)
static const struct device *accel_dev = DEVICE_DT_GET(BMI088_ACCEL_NODE);
static const struct device *gyro_dev  = DEVICE_DT_GET(BMI088_GYRO_NODE);
...
sensor_sample_fetch(accel_dev);
sensor_channel_get(accel_dev, SENSOR_CHAN_ACCEL_XYZ, accel);   /* accel[3] */
sensor_sample_fetch(gyro_dev);
sensor_channel_get(gyro_dev,  SENSOR_CHAN_GYRO_XYZ,  gyro);    /* gyro[3]  */
```

### 6.2 RoSE virtual driver sketch (`drivers/sensor/rose_bmi088.c`)

The wire packet is the one the env already produces for
`rose_flight_controller`: `ROSE_CMD_IMU 0x12` → 6 `float32` on ch2
`[ax,ay,az, gx,gy,gz]` (`main.cpp:42,164` and `README.rst:17`). The accel and
gyro logical devices share one fetch; a small cache holds the last frame so the
two `sensor_sample_fetch` calls in the app don't double-request. Sketch:

```c
#define DT_DRV_COMPAT ucbbar_rose_bmi088_accel   /* second unit for _gyro */

struct rose_imu_cfg {
    const struct device *rose;   /* DEVICE_DT_GET(DT_INST_PHANDLE(inst, rose)) */
    uint32_t cmd;                /* DT_INST_PROP(inst, rose_cmd)     -> 0x12   */
    uint8_t  channel;            /* DT_INST_PROP(inst, rose_channel) -> 2      */
    bool     is_gyro;            /* which half of the 6-word frame we expose   */
};
struct rose_imu_data { float ax,ay,az, gx,gy,gz; };

static int rose_imu_sample_fetch(const struct device *dev, enum sensor_channel chan)
{
    const struct rose_imu_cfg *cfg = dev->config;
    struct rose_imu_data *d = dev->data;
    uint32_t raw[6];

    rose_request(cfg->rose, cfg->cmd, 0);                       /* [0x12][0] */
    int n = rose_recv_reqrsp(cfg->rose, cfg->channel, raw, 6);  /* framed rx */
    if (n < 6) return -EIO;
    memcpy(&d->ax, raw, sizeof(float) * 6);                     /* float bits */
    return 0;
}

static int rose_imu_channel_get(const struct device *dev, enum sensor_channel chan,
                                struct sensor_value *val)
{
    const struct rose_imu_cfg *cfg = dev->config;
    struct rose_imu_data *d = dev->data;
    const float *v = cfg->is_gyro ? &d->gx : &d->ax;
    if ((cfg->is_gyro  && chan != SENSOR_CHAN_GYRO_XYZ) ||
        (!cfg->is_gyro && chan != SENSOR_CHAN_ACCEL_XYZ))
        return -ENOTSUP;
    for (int i = 0; i < 3; i++) sensor_value_from_double(&val[i], v[i]);
    return 0;
}

static const struct sensor_driver_api rose_imu_api = {
    .sample_fetch = rose_imu_sample_fetch,
    .channel_get  = rose_imu_channel_get,
};
/* ... SENSOR_DEVICE_DT_INST_DEFINE(inst, rose_imu_init, NULL,
       &data_##inst, &cfg_##inst, POST_KERNEL,
       CONFIG_SENSOR_INIT_PRIORITY, &rose_imu_api); ... */
```

Notes:
- `float` payload → `struct sensor_value` conversion via
  `sensor_value_from_double` (the real BMI088 driver already produces
  `sensor_value`, and the app converts back with `sensor_value_to_double`,
  `attitude_estimator.c:134-140`; round-trip is lossless enough for the
  estimator). If bit-exactness matters, expose a private
  `SENSOR_CHAN_PRIV`/`sensor_channel_get`-of-floats path instead.
- The gyro node reuses `cmd=0x12` and just slices `[3..5]` of the same frame.
  Because both call `rose_request`, the naive version issues the request twice per
  loop; §8 discusses collapsing that.

## 7. Worked example — Flow deck (optical flow + ToF)

Wire packet today: `ROSE_CMD_FLOW 0x13` → `[vx, vy, h]` on ch1
(`main.cpp:43,168`, `README.rst:18`). Two logical devices:

- `ucbbar,rose-flow` → exposes body-frame horizontal velocity. Zephyr has **no
  standard optical-flow channel**, so map to private channels
  `SENSOR_CHAN_PRIV_START + 0/1` (or the PMW3901 out-of-tree driver's convention
  if we standardize on it — see §8) and document them.
- `ucbbar,rose-tof` → exposes downward height as `SENSOR_CHAN_DISTANCE` (the same
  channel Zephyr's `st,vl53l1x`/`vl53l0x` ToF drivers use), so a real Flow deck
  build binds `DT_ALIAS(tof)` to `st,vl53l1x` with zero app change.

Multi-rate ToF: the estimator already treats ToF as **low-rate** and gates fusion
on a `tof_valid` flag (`estimator.hpp:31-38`). The virtual ToF driver should mirror
that: `sample_fetch` returns `-EAGAIN` (or sets a "stale" flag) when no fresh ToF
sample is available this step, and the app passes `tof_valid = (fetch == 0)` into
`est.update(...)`. Because flow and ToF ride in the same `0x13` packet today, the
cleanest split is to let the flow driver own the fetch and the ToF driver read the
cached `h` plus a freshness counter; alternatively give ToF its own reqrsp command
so its rate is independent of flow.

## 8. Migration path for `samples/rose_flight_controller`

The estimator, TinyMPC, and control math stay untouched. Only acquisition changes:

1. **Add the RoSE sensor drivers** to `soc/sw/zephyr-rose/drivers/sensor/` +
   bindings + Kconfig (§4).
2. **Extend the overlay** `rose_flight_controller/boards/spike_riscv64.overlay`
   with the four virtual sensor nodes + aliases (§4.2), keeping `rose@2000`.
3. **Update `prj.conf`**: add `CONFIG_SENSOR=y`, `CONFIG_ROSE_SENSOR_BMI088=y`,
   `CONFIG_ROSE_SENSOR_FLOW=y` (keep `CONFIG_ROSE`/`CONFIG_ROSE_PROTOCOL`).
4. **Replace `recv_sensor`** (`main.cpp:114-127`) and the two call sites
   (`main.cpp:164-171`) with `DEVICE_DT_GET(DT_ALIAS(...))` + `sensor_sample_fetch`
   / `sensor_channel_get`, i.e. converge on `attitude_estimator.c`'s exact usage.
   Drop the `ROSE_CMD_*`/`ROSE_*_CH` macros (`main.cpp:42-46`) — they move to DT.
5. **Control output** (`send_control`, `main.cpp:129-139`, `ROSE_CMD_CONTROL
   0x20`) stays on the raw `rose_tx` path for now — it is an *actuator*, not a
   sensor; a future "rose actuator" shim could mirror this design, but it is out of
   scope. The guest still needs `DEVICE_DT_GET_ONE(ucbbar_roseadapter)` for TX, or
   the sensor drivers expose the adapter phandle.
6. **End state:** `rose_flight_controller` and `flight_controller` share `main` +
   estimator, differing only in overlay + `prj.conf`. At that point they could be
   unified into one sample directory with two board overlays.

## 9. Open questions / risks

1. **Command-ID ↔ channel ↔ sensor mapping.** Today it is defined in **two**
   places that must agree: the guest (`main.cpp:42-46`) and the env/synchronizer
   config `deploy/config/config_gym_IsaacCrazyflieSensorEnv-v0.yaml`, which
   declares the `imu` packet (cmd `0x12`, reqrsp channel 2, 6×`float32`
   `[ax,ay,az,gx,gy,gz]`) and the `flow` packet (cmd `0x13`, reqrsp channel 1,
   `[vx,vy,h]`, "modeling a Crazyflie Flow deck v2"). Moving the guest side into
   DT props (`rose-cmd`, `rose-channel`) makes it declarative, but the **env side
   must stay in lockstep** — a mismatch silently corrupts data. Consider
   generating both from one source, alongside the existing
   `soc/sw/generated-src/rose_c_header/rose_packet.h` (referenced in
   `rose_proto.h:28`).
2. **Shared-packet sensors / double requests.** Accel+gyro share `0x12`; flow+ToF
   share `0x13`. Two `sensor_sample_fetch` calls would issue two requests unless we
   cache. Need a per-command latch (fetch-once-per-tick) or a "primary" device that
   owns the fetch. This also interacts with **framing correctness**: a spurious
   extra `rose_request` desynchronizes the reqrsp FIFO (`rose_recv_reqrsp` assumes
   one framed response per request, `rose_proto.c:37-62`).
3. **Blocking vs non-blocking fetch / latency.** `rose_rx_impl` **spins** until the
   word is valid (`rose_adapter.c:83-113`) — `sample_fetch` will busy-block the
   calling thread until the synchronizer serves the packet. Acceptable in the
   lockstep co-sim (the sim is what we're waiting on), but it means the sensor API's
   usual "cheap fetch" assumption doesn't hold; document it, and keep the fetch off
   any latency-critical path. Zephyr's async `SENSOR_TRIG`/read-decode (RTIO) path
   is a larger future option.
4. **Optical-flow channel standardization.** No in-tree Zephyr channel for optical
   flow. Choose private channels vs adopting the PMW3901 driver's convention so a
   real Flow-deck build stays app-identical. ToF maps cleanly to
   `SENSOR_CHAN_DISTANCE`.
5. **float ↔ sensor_value fidelity.** Bridge payload is `float32`; the sensor API
   uses fixed-point `sensor_value` (int + micro). Round-trip through
   `from_double`/`to_double` is fine for this estimator but not bit-exact; offer a
   private float channel if a consumer needs raw bits.
6. **Timing / rate ownership.** The control-loop rate (200 Hz, `CTRL_DT 0.005`,
   `README.rst:71-84`) currently drives requests. With sensors behind the API, the
   loop calls `sample_fetch` at its own cadence; ToF multi-rate must be handled in
   the driver (§7) so the app doesn't need to know packet-sharing details.
7. **DMA/camera sensors.** This design covers reqrsp (small, framed) sensors. A
   camera "sensor" would ride the DMA path (`rose_dma_*`, `rose_recv_dma`) and
   likely want the Zephyr `video` API rather than `sensor` — separate design.
8. **Actuator parity.** Control output stays raw (`0x20`). If sim-to-real parity
   for actuation is wanted later, a matching virtual PWM/motor driver is the analog
   of this work.

## 10. Summary of recommendation

Add virtual Zephyr **sensor** drivers to the existing `zephyr-rose` module
(`ucbbar,rose-bmi088-accel/-gyro`, `ucbbar,rose-flow`, `ucbbar,rose-tof`) that
implement `sensor_driver_api` on top of `rose_request`/`rose_recv_reqrsp`, carrying
the RoSE command-ID and reqrsp-channel as **devicetree properties**. Keep the app's
`DT_ALIAS(bmi088_accel)` etc. stable so `main`, the estimator, and TinyMPC are
byte-for-byte shared with the real `samples/flight_controller`; only the board
overlay and `prj.conf` differ per target. This is exactly the extension point the
protocol layer was written to support (`rose_proto.h:9-13`).
