# Zephyr for RoSE — Design Plan

Replace RoSE's ad-hoc baremetal application flow with a Zephyr RTOS port, exposing the
RoSE co-sim bridge/adapter through clean, layered driver abstractions instead of
hand-coded MMIO pokes.

Status: **in progress — step 1 done** (stock Zephyr boots on the RoSE SoC in metasim).
Companion docs: `ROSE_FIRESIM_MIGRATION_PLAN.md`, `INSTALL_NOTES.md`.

## Progress log

### 2026-06-30 — Step 1 DONE: stock Zephyr boots on `RoseTLRocketMMIOOnlyConfig` (metasim) ✅
- Initialized `soc/sw/xpu-rt/zephyr-chipyard-sw` (ucb-bar Zephyr-for-Chipyard workspace).
  Its `zephyr_ws/zephyr` is the ucb-bar Zephyr fork; ML samples pull many extra submodules
  (executorch/XNNPACK/…) not needed for driver bring-up.
- Reused an existing populated install (`/scratch2/dima/zephyr-chipyard-sw-fresh`, Zephyr SDK
  `1.0.0-beta1` + `zephyr` conda env) to avoid a multi-GB fresh SDK install. Built
  `west build -p always -b spike_riscv64 samples/hello_world`
  (SDK gcc 14.3.0) → `zephyr.elf` with htif `tohost`/`fromhost` symbols at `0x80008200`.
- Ran on our metasim: launched `run_sync_only.py` (the RoSE bridge driver blocks the sim until
  the synchronizer connects), then `make run-verilator … SIM_BINARY=<zephyr.elf>`. Output:
  ```
  *** Booting Zephyr OS build 4329bf61c4fe ***
  Hello World! spike_riscv64/rocketchip_virt_riscv64
  ```
- **Answers to §6 open questions:**
  - *Console:* htif works out of the box (`CONFIG_UART_HTIF=y`, `zephyr,console=&htif`) — no
    UART plumbing needed; output captured via the same spike-dasm `.out` path as baremetal.
  - *Board:* `spike_riscv64` (variant `rocketchip_virt_riscv64`) boots as-is — it's already
    rocketchip-oriented (single CPU via `MP_MAX_NUM_CPUS=1`, hard FPU, `clint@2000000`,
    `plic@c000000`, `memory@80000000`), matching the RoSE Rocket SoC.
  - *PLIC/CLINT:* functional (OS boots + ticks).
  - *RoSE bridge coexists:* the `rosebridge_t` driver connects to the synchronizer and the sim
    advances normally while Zephyr runs.
- Artifacts: build at `soc/sim/zephyr_hello_build/`, logs `soc/sim/zephyr_hello_build.log` +
  `soc/sim/run_zephyr.log`, console `…/sim/zephyr.out`.
- **Still open:** `timebase-frequency` (RoSE metasim = 1 MHz) may need a RoSE overlay for
  correct delays; and adding the `RoseAdapter` DT node + driver (steps 2–3).

### 2026-06-30 — Step 2 DONE: polling tx/rx driver + `rxvalidate` on Zephyr → RXVALIDATE PASS ✅
- Created a **RoSE-owned Zephyr module** at `soc/sw/zephyr-rose/` (keeps the upstream
  `zephyr-chipyard-sw` submodule un-patched):
  - `zephyr/module.yml` (with `settings.dts_root: .` so the binding is discovered),
    top `CMakeLists.txt`/`Kconfig`.
  - DT binding `dts/bindings/misc/ucbbar,RoseAdapter.yaml`.
  - Driver `drivers/rose/rose_adapter.c` + `Kconfig` (`CONFIG_ROSE`, gated on
    `DT_HAS_UCBBAR_ROSEADAPTER_ENABLED`), registered via `DEVICE_DT_INST_DEFINE`.
  - Public API `include/rose/rose.h`: `rose_tx` / `rose_rx` / `rose_tx_ready` /
    `rose_rx_ready` over a `struct rose_driver_api` (device-model function table).
  - Sample `samples/rxvalidate/` with a `spike_riscv64.overlay` adding the
    `rose@2000` node (`reg = <0x2000 0x1000>`, single-cell) + a `main.c` that uses only
    the driver API (no register pokes).
- Build: `west build -p always -b spike_riscv64 soc/sw/zephyr-rose/samples/rxvalidate
  -- -DZEPHYR_EXTRA_MODULES=soc/sw/zephyr-rose`.
- **Gotcha fixed:** Zephyr **lowercases** DT compatibles for C tokens —
  `DT_DRV_COMPAT`/`DEVICE_DT_GET_ONE` must use `ucbbar_roseadapter` (not
  `ucbbar_RoseAdapter`); the Kconfig `DT_HAS_..._ENABLED` symbol is uppercase. With the
  mixed-case token, `DT_INST_FOREACH_STATUS_OKAY` matched zero instances and
  `DEVICE_DT_GET_ONE` errored "no such device".
- Ran on the metasim (synchronizer route `0x11 -> ch2`):
  ```
  *** Booting Zephyr OS build 4329bf61c4fe ***
  RXVALIDATE: PASS pattern at offset 2 (buf=0x00000011,0x00000040,0xc0de0000,0xc0de0001)
  ```
  Matches the baremetal `RXVALIDATE: PASS` — the reqrsp RX path now works through a real
  Zephyr driver API instead of `rose_port.h` macros.
- Artifacts: module `soc/sw/zephyr-rose/`, build `soc/sim/zephyr_rxvalidate_build/`,
  logs `soc/sim/zephyr_rxvalidate_build*.log` + `soc/sim/run_zrx.log`.

### 2026-06-30 — Step 3 DONE: DMA + interrupt-driven completion → DMAVALIDATE PASS ✅
- Extended the driver API (`include/rose/rose.h`): `rose_dma_arm(dev, ch, nbytes)`,
  `rose_dma_wait(dev, ch, timeout)` (blocks on a `k_sem`), `rose_dma_buffer(dev, ch)`
  (returns the DMA target address from the DT `dma-base-address`).
- Driver (`drivers/rose/rose_adapter.c`): added the DMA config-counter write, a PLIC ISR,
  and per-DMA-channel semaphores. Register offsets are computed from the port topology
  exactly as the HW regmap (RoSEAdapter.scala): DMA cfg = `0x0c + (num_reqrsp+ch)*4`,
  **interrupt pending (W1C) = `0x0c + (num_reqrsp + 2*num_dma + 1)*4`** (= `0x20` for the
  2-reqrsp/1-DMA config). ISR reads pending, gives the channel's `k_sem`, writes back to
  W1C-clear (deasserts the level interrupt). IRQ wired per-instance via
  `IRQ_CONNECT(DT_INST_IRQN, DT_INST_IRQ(.., priority), ...)`, guarded by
  `DT_INST_IRQ_HAS_IDX` so the polling-only rxvalidate node (no `interrupts`) still builds.
- Overlay (`samples/dmavalidate/boards/spike_riscv64.overlay`): `rose@2000` with
  `interrupt-parent = <&plic>`, `interrupts = <3 1>` (PLIC source 3, matching the chipyard
  DTS; PLIC `#interrupt-cells = <2>` = irq, priority).
- Sample `samples/dmavalidate/`: arms DMA0, requests the frame, **blocks on the DMA-complete
  IRQ with `K_FOREVER`** (no busy-poll of the DMA status bit), then validates memory at the
  DMA target. Ran on the metasim (route `0x11 -> ch0`):
  ```
  *** Booting Zephyr OS build 4329bf61c4fe ***
  DMAVALIDATE: PASS rx[0]=0xc0de0000 rx[15]=0xc0de000f
  ```
  `K_FOREVER` means the PASS proves the **PLIC interrupt actually fired** (ISR gave the
  semaphore) — the interrupt-driven abstraction works, replacing the baremetal busy-poll of
  `ROSE_DMA_BUFFER_0`.
- Artifacts: build `soc/sim/zephyr_dmavalidate_build/`, logs `soc/sim/zephyr_dmavalidate_build.log`
  + `soc/sim/run_zdma.log`.
- **Note on timebase:** `rose_dma_wait` used `K_FOREVER` specifically to avoid the still-open
  `timebase-frequency` question (a timed `k_sem_take` would depend on the CLINT tick rate
  matching the RTL's 1 MHz mtime). A production timeout needs a RoSE board overlay setting
  `timebase-frequency = <1000000>` and verification of the tick rate.

### Both baremetal RoSE RX paths now reproduced on Zephyr through clean driver APIs:
`RXVALIDATE: PASS` (reqrsp, polling) and `DMAVALIDATE: PASS` (DMA, interrupt-driven).

### 2026-06-30 — Step 4 (core) DONE: `subsys/rose` protocol layer → PROTOVALIDATE PASS ✅
- Added the transport-neutral protocol layer `subsys/rose/` + public `include/rose/rose_proto.h`:
  - `rose_request(dev, cmd, arg)` — request framing (cmd + arg over TX).
  - `rose_recv_reqrsp(dev, ch, out, max_words)` — reads the served `[header, num_bytes,
    data...]` frame, stores up to `max_words` data words, drains the rest (keeps FIFO framed).
  - `rose_recv_dma(dev, ch, request_cmd, nbytes, &buf, timeout)` — arm → request → wait →
    return buffer pointer.
  - `rose_request_camera()` helper + the `ROSE_CMD_*` command set (mirrors rose_packet.h).
- Module wiring: `subsys/CMakeLists.txt`/`Kconfig` (`CONFIG_ROSE_PROTOCOL`), added to the
  module top-level. Gave the two `zephyr_library` calls explicit names (`rose_driver`,
  `rose_proto`) to avoid the duplicate-"rose"-dir collision.
- Proved it with `samples/protovalidate/` (uses only `rose_request_camera` +
  `rose_recv_reqrsp`, no raw tx/rx). Metasim (route `0x11 -> ch2`):
  ```
  PROTOVALIDATE: PASS n=16 buf[0]=0xc0de0000 buf[15]=0xc0de000f
  ```
  `buf[0]` is the first *data* word (not the header) and `n=16` — the protocol layer stripped
  the `[header, num_bytes]` framing that the raw `rxvalidate` had to scan past manually.
- Artifacts: `soc/sw/zephyr-rose/subsys/rose/`, build `soc/sim/zephyr_protovalidate_build/`,
  log `soc/sim/run_zproto.log`.

## Shared-API strategy (decided 2026-06-30): same user API for co-sim and real robot

Goal: an app using e.g. a BMI088 IMU should transparently use RoSE-bridge packets in co-sim
or real I2C on hardware. This is feasible via Zephyr's device model — the **shared user API is
Zephyr's generic `sensor.h`**; the deployment scenario is chosen at build time by
devicetree/board (which node/bus the sensor binds to). Confirmed upstream has `sensor.h`, the
`bosch,bmi08x` driver (I2C *and* SPI bindings), and the I2C emulation framework.

Two seams, **decision = pursue both (A is the goal, B is the stepping stone):**
- **Pattern A — transport-transparent:** a `ucbbar,rose-i2c` controller whose `i2c_transfer()`
  marshals transactions through `subsys/rose`; the **stock `bosch,bmi08x` driver runs
  unmodified**, transport chosen by the sensor's DT bus parent (`&rose_i2c` vs `&i2c0`).
  Highest fidelity; requires the RoSE sim to emulate device registers + an I2C-transaction
  packet type.
- **Pattern B — sample-transparent:** a small `ucbbar,rose-imu` driver implementing `sensor.h`
  that fetches SI accel/gyro over `subsys/rose`; the robot build binds the real `bosch,bmi08x`.
  Less work (sim serves physical quantities), per-sensor, lower fidelity. Good first proof of
  the shared `sensor.h` API.

`subsys/rose` (done) is the substrate for both.

### 2026-06-30 — Drone physics env DONE: `PyBulletDroneEnv-v0` (gym-pybullet-drones) ✅
The sim-side counterpart for the Pattern B `rose-imu` driver: a RoSE gym env that serves a
quadrotor's IMU state over the bridge.
- `deploy/hephaestus/envs/pybullet_drone/drone_env.py`: wraps `CtrlAviary` (single CF2X drone)
  from the **zephyr-chipyard-sw `tools/gym-pybullet-drones` submodule** (same dynamics as the
  Zephyr drone samples; added to `sys.path` — no pip install of the package). Observation dict:
  `imu_accel` (body-frame specific force, scipy rotation), `imu_gyro` (body-frame angular
  velocity), `state` (full 20-vec), `pos`, `quat` — all `float32`. Action = 4 motor RPMs
  (defaults to hover when the SoC hasn't issued a command).
- Registered `PyBulletDroneEnv-v0` in `register_envs.py`; config
  `deploy/config/config_gym_PyBulletDroneEnv-v0.yaml` maps `imu_accel -> cmd 0x12/ch2`,
  `imu_gyro -> cmd 0x13/ch1`.
- New venv deps (in `requirements.txt`): `pybullet`, `transforms3d`, `setuptools<81`
  (BaseAviary imports `pkg_resources`, dropped in setuptools >= 81).
- Standalone smoke test (`deploy/hephaestus/test_drone_env.py`) → **DRONE_ENV_SMOKE: PASS**:
  hovering gives `accel=[0,0,9.8]`, `gyro=[0,0,0]`, drone rests at z=0.11 m; `imu_accel`
  serializes to bridge words `[0x0,0x0,0x411ccccc]` (= float32 9.8), confirming the
  float-bits-over-uint32 path the SoC `rose-imu` driver will consume.

### 2026-07-01 — Organized `samples/rose/` test suite + on-FPGA validation ✅
Consolidated the ad-hoc validate apps into an organized, self-checking suite under
`soc/sw/xpu-rt/zephyr-chipyard-sw/samples/rose/` (the Zephyr workspace's sample home; the
`rose` driver module stays in `soc/sw/zephyr-rose`):
- `common/rose_check.h` — shared validator: checks every word vs the known PatternEnv ramp,
  emits a greppable `ROSE <name>: PASS/FAIL ...` marker; holds the test cmd->channel map.
- `reqrsp/` — low-level RX (`rose_tx`/`rose_rx`, framed read, cmd 0x12 -> ch2).
- `dma/` — interrupt-driven camera DMA (`rose_dma_arm`/`rose_dma_wait`, cmd 0x11 -> ch0).
- `protocol/` — high-level transport (`rose_request`/`rose_recv_reqrsp`, framing stripped).
- `selftest/` — combined DMA + reqrsp in one binary (summary marker) for one-shot FPGA runs.
- `README.rst` — index + build/run.
- **Unified 2-route synchronizer config** (`config_gym_PatternEnv-v0.yaml`): `0x11 -> ch0 DMA`,
  `0x12 -> ch2 reqrsp` — one config drives every sample.

Build gotcha: the small `dma` image failed to link with
`R_RISCV_GPREL_I relocation truncated against z_interrupt_stacks` (interrupt stack outside the
+/-2KB GP window; layout-sensitive — `selftest` was large enough to avoid it). `CONFIG_RISCV_GP=n`
does NOT help (it's a `choice` member, silently ignored). Fix: `CONFIG_EXCEPTION_STACK_TRACE=n`
in `dma/prj.conf` (drops the stacktrace helper that GP-references the far symbol). All 4 build.

**On-FPGA validation (U250, 1.14.0 bitstream, `+partitioned=1`):** ran `selftest` ->
```
ROSE dma:      PASS (16 words, [0]=0xc0de0000 [15]=0xc0de000f)
ROSE reqrsp:   PASS (16 words, [0]=0xc0de0000 [15]=0xc0de000f)
ROSE selftest: dma=PASS reqrsp=PASS => PASS
```
Bridge trace: Zephyr boots -> `Pushing cmd 11` (DMA) -> `Pushing cmd 12` (reqrsp) -> both PASS.
So both the interrupt-driven DMA path and the reqrsp/protocol path are validated through the
Zephyr driver stack on real hardware, in one run. (Build via
`soc/sim/zephyr_rose_builds/`; FPGA workload `deploy/workloads/rose-selftest*`.)

### Next: step 4b — Pattern B `ucbbar,rose-imu` sensor driver (implements `sensor.h`) over
`subsys/rose`, now with `PyBulletDroneEnv-v0` as the real IMU source (SoC sends cmd 0x12/0x13,
receives float accel/gyro). Then step 5 — a proper RoSE board (timebase + RoseAdapter baked in),
west-manifest integration, and eventually Pattern A (`rose-i2c` so the stock BMI088 driver runs
unmodified) with a sim-side register model.

---

## 1. Motivation

Today every RoSE SoC program (`soc/sw/rose-images/airsim-packettest/*.c`, etc.) is baremetal
and ad-hoc:

- Direct register pokes via generated macros in `soc/sw/generated-src/rose_c_header/rose_port.h`
  (`reg_write32(ROSE_TX_DATA_ADDR, w)`, `reg_read32(ROSE_RX_DATA_2)`, …).
- **Busy-polling** `ROSE_STATUS_ADDR` bits for flow control — including for DMA completion,
  even though the hardware raises a real interrupt for it (see §2).
- No threading, no buffering, no error handling.
- The RoSE packet protocol (framing, the budget/bigstep synchronizer handshake, camera-request
  commands) is **re-implemented in each app**.
- Bespoke build: `riscv64-unknown-elf-gcc … -specs=htif_nano.specs -T tests/htif.ld …`.

The goal: drivers + a transport subsystem with real APIs, so an app reads like
`rose_request_camera(); rose_recv_frame(); process();` with no knowledge of register layout.

---

## 2. Why Zephyr fits (the hardware already self-describes)

The RoSE adapter (`RoseAdapterTL`, `soc/src/main/scala/RoSEAdapter.scala`) is already a
device-tree-described, interrupt-capable TileLink MMIO peripheral. The chipyard-generated DTS
for `RoseTLRocketMMIOOnlyConfig` contains a node that maps **directly** onto Zephyr's
devicetree-driven driver model:

```dts
RoseAdapter@2000 {
    compatible = "ucbbar,RoseAdapter";
    interrupt-parent = <&L15>;        // the PLIC
    interrupts = <3>;                 // DMA-complete IRQ line
    reg = <0x0 0x2000 0x0 0x1000>;    // control MMIO window, 4 KB
    reg-names = "control";
};
```

- Defined in HW as `new SimpleDevice("RoseAdapter", Seq("ucbbar,RoseAdapter"))`.
- Interrupt source: `nInterrupts = #DMA ports`, an `IntSourceNode` with a `w1ToClear` pending
  register, **wired to the PLIC** via `ibus.fromSync := roseAdapterTL.intnode`.
- Zephyr binds drivers by `compatible`, derives base/size from `reg`, and wires the IRQ from
  `interrupts`/`interrupt-parent` — so the plumbing is essentially free.

### Register map (from `rose_port.h`, control window @ 0x2000)

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x2000 | STATUS | RO | bit0 TX_ENQ_READY, bit1 RX_DEQ_VALID(ch2), bit2 RX_DEQ_VALID(ch1), bit3 DMA_BUFFER(ch0) |
| 0x2008 | TX_DATA | WO | enqueue a TX word (valid set on write) |
| 0x200c | RX_DATA_1 | RO | dequeue reqrsp0 (ch1); read sets ready |
| 0x2010 | RX_DATA_2 | RO | dequeue reqrsp1 (ch2); read sets ready |
| 0x2014 | DMA_CONFIG_COUNTER_0 | WO | DMA buffer size (bytes) for ch0 |
| 0x2018 | DMA_CURR_COUNTER_0 | RO | DMA progress for ch0 |
| (intr) | pending | W1C | DMA-complete interrupt pending bits |

Topology: `ROSE_PORT_COUNT = 3`, width 32. Ports = 1 DMA (ch0 → memory @ `0x88000000`) +
2 reqrsp (ch1, ch2). The DMA target `0x88000000` is **not** in the adapter node — it is a fixed
location in FASED DRAM (see §6 open questions).

---

## 3. Proposed layered architecture

Four layers, each replacing a chunk of today's ad-hocery.

### Layer 1 — DT binding
`dts/bindings/misc/ucbbar,RoseAdapter.yaml`: describe the node Zephyr already receives from
chipyard — `reg`, `interrupts` — plus RoSE-specific properties:
`num-reqrsp-channels`, `num-dma-channels`, `dma-base-address` (0x88000000), `port-width` (32).
Consumed via `DEVICE_DT_GET` / `DT_INST_*`.

### Layer 2 — low-level device driver
`drivers/misc/rose/rose_adapter.c`, exposing a device-model API (`struct rose_driver_api`):

| Current ad-hoc baremetal | Zephyr driver API |
|---|---|
| `while(!ROSE_TX_ENQ_READY); reg_write32(TX_DATA, w)` | `rose_tx(dev, w)` |
| `while(!ROSE_RX_DEQ_VALID_2); reg_read32(RX_DATA_2)` | `rose_rx(dev, ch, &w, timeout)` |
| `reg_write32(DMA_COUNTER, n); poll ROSE_DMA_BUFFER_0` | `rose_dma_arm(dev, ch, buf, n)` + `rose_dma_wait(dev, ch, timeout)` |
| (DMA-complete interrupt unused) | **ISR clears W1C pending → gives `k_sem`** so threads block instead of busy-polling |
| `rose_port.h` magic addresses | `DT_INST_REG_ADDR`, `DT_INST_IRQN`, devicetree-derived |

Key upgrade: **interrupt-driven DMA completion**. The ISR services the pending register and
signals a semaphore; a perception thread `k_sem_take()`s and sleeps rather than spinning on
`ROSE_DMA_BUFFER_0`. Enables genuine multitasking (e.g. control thread + perception thread).

### Layer 3 — RoSE transport/protocol subsystem
`subsys/rose/`: encapsulate what every app currently duplicates —
- packet framing (control vs data, `cmd < 0x80`; data layout `[cmd, big_step, budget, num_bytes, data]`),
- the **budget/bigstep synchronizer handshake**,
- channel ↔ port routing,
- high-level ops: `rose_request_camera(dev, CAM_LEFT)`, `rose_recv_frame(dev, ch_or_dma, buf, len)`.

This is where RoSE stops being "poke these registers in this order" and becomes an API.

### Layer 4 — applications / tests
Zephyr apps and `ztest` cases. The validation tests collapse to:

```c
const struct device *rose = DEVICE_DT_GET_ONE(ucbbar_RoseAdapter);
rose_dma_arm(rose, 0, buf, sizeof(buf));
rose_request_camera(rose, CAM_LEFT);
rose_dma_wait(rose, 0, K_MSEC(100));   /* sleeps on the DMA-complete IRQ */
/* validate buf against expected pattern */
```

---

## 4. Build / run integration

- Replace the bespoke gcc/htif invocation with `west build` + a **Zephyr board** for the RoSE
  SoC (likely a variant/overlay over whatever `zephyr-chipyard-sw` provides). The board `.dts`
  is (or includes) the chipyard-generated DTS, so RoseAdapter, PLIC, CLINT, and UART all come
  from there.
- Run flow is **unchanged**: Zephyr emits an ELF that FireSim loads via htif/fesvr —
  `make run-verilator … SIM_BINARY=<zephyr.elf>` still works; the same FPGA bitstream runs it.

---

## 5. Incremental migration path (each step verifiable in metasim)

1. **Boot stock Zephyr** on `RoseTLRocketMMIOOnlyConfig` (init `soc/sw/xpu-rt/zephyr-chipyard-sw`,
   get hello-world over the console). De-risks board/console/PLIC.
2. **Polling driver** (tx/rx only) + binding → port `rxvalidate` as a Zephyr app → reproduce
   `RXVALIDATE: PASS`.
3. **Add DMA + IRQ path** → port `dmavalidate` → reproduce `DMAVALIDATE: PASS`, now
   interrupt-driven.
4. **Build `subsys/rose`** (protocol layer) → port a real perception/control app.
5. **Deprecate** the `airsim-packettest` baremetal flow.

Reference results to reproduce (current baremetal, chipyard 1.14.0 metasim):
`DMAVALIDATE: PASS rx[0]=0xc0de0000 rx[15]=0xc0de000f`,
`RXVALIDATE: PASS pattern at offset 2`.

---

## 6. Open questions to resolve first

- **Console:** does `zephyr-chipyard-sw` provide an HTIF console, or do we target the SoC UART
  (this config has a UART bridge)? Determines step-1 plumbing.
- **Board reuse:** does `zephyr-chipyard-sw` already ship a Rocket board matching this config
  (HART count, CLINT/PLIC base, timebase freq), or must we author one? (Submodule not yet
  initialized — first thing to inspect.)
- **DMA target memory:** `0x88000000` is a fixed target in FASED DRAM, not in the adapter node.
  Zephyr must describe it as a reserved/`zephyr,memory-region` (or the driver uses the fixed
  address with explicit cache management) so the allocator doesn't collide and reads are coherent.
- **PLIC setup** for IRQ 3 (priority/threshold/enable) — handled by Zephyr's RISC-V PLIC driver
  if the board enables it.
- **Multi-DMA / multi-channel scaling:** `nInterrupts = #DMA ports`; the binding/driver should
  generalize beyond the single DMA channel of `RoseTLRocketMMIOOnlyConfig`.

---

## 7. Source-of-truth references

- HW adapter + DTS node + IRQ wiring: `soc/src/main/scala/RoSEAdapter.scala`
  (`SimpleDevice("RoseAdapter", Seq("ucbbar,RoseAdapter"))`, `IntSourceNode`,
  `ibus.fromSync := roseAdapterTL.intnode`).
- Generated register map: `soc/sw/generated-src/rose_c_header/rose_port.h`.
- Generated DTS: `soc/sim/chipyard/sims/firesim-staging/generated-src/firechip.chip.FireSim.RoseTLRocketMMIOOnlyConfig/*.dts`.
- Zephyr port submodule: `soc/sw/xpu-rt/zephyr-chipyard-sw` (nested in XPU-RT; not yet initialized).
- Current baremetal apps to replace: `soc/sw/rose-images/airsim-packettest/*.c`.

---

## 8. Recommended first move

Initialize `soc/sw/xpu-rt/zephyr-chipyard-sw`, scope its existing board/driver/console support,
and boot stock Zephyr on `RoseTLRocketMMIOOnlyConfig` in metasim (step 1). That answers most of
§6 before any driver code is written.
