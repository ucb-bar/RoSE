# Gate-nav co-sim: spike reproduce → faithful FPGA port

The on-SoC flight stack (**fused vision + EKF + TinyMPC → 4 motor thrusts**, all vectorized
RVV) flying the IsaacLab warehouse **gate course**. This is the config that produces the
`docs/media/framefix_gatenav/*_3gate_*.mp4` videos: **3/4 gates on seed 1000**.

**The working tier today is SPIKE** (functional RISC-V ISS `rose_spike_sim` + Isaac). This doc
is the full spike reproduce, plus how to port it faithfully to the **FPGA** (firesim1 U250,
Saturn RVV) so the *only* thing that changes is the compute backend.

---

## 1. What runs where

```
IsaacLab warehouse (photoreal gates, cameras, rigid-body dynamics)  ── GPU (garden)
        ▲ sensors (cam 0x11 DMA, tof_cross 0x41, lowdim 0x42)  │ thrusts (0x20)
        │                        gym_synchronizer (:10001)      ▼
   on-SoC flight stack  =  rose_fused_mpc guest ELF, RVV-vectorized
        └── compute backend: SPIKE (rose_spike_sim ISS)  OR  FPGA (Saturn+RoSE bitstream)
```

The guest is **identical** for both tiers (same source, same flags, same RVV) — only the board
target and the harness that executes it differ.

---

## 2. Reproduce on SPIKE (the working 3/4 config)

### 2a. Prereqs
- `env_isaaclab` conda env + IsaacLab (external; `ISAAC_PY=/scratch2/dima/miniforge3/envs/env_isaaclab/bin/python`).
- `rose_spike_sim` built: `soc/src/main/cc/rose_spike/build.sh → soc/sim/rose_spike_sim`.
- Guest ELF built (below).

### 2b. Build the guest (vectorized rose_fused_mpc)
```bash
source experiments/rose_nav_cosim/env.sh          # sets ROSE_DIR, ROSE_MODULE
experiments/rose_nav_cosim/build_guest.sh
```
`build_guest.sh` compiles Zephyr sample **`rose_fused_mpc`** for board **`spike_riscv64`** with:
- `-DRISCV_VECTOR=1 -DNAV_MODE=1 -DROSE_FUSED_NAV=1`  (RVV on, full on-SoC nav)
- `-DMODEL_DIR=model/rvv_f16`  (the winning hybrid: int8 encoders `conv2d_s8_pc` + fp16 tail `linear_f16`/`lstm_f16`)
- `-DCTRL_ITERS=5000 -DSETTLE_ITERS=200 -DFUSED_VISION_DIV=10` (20 Hz vision)
- `-DSTART_Z=2.0 -DTARGET_Z=2.0 -DYAW_CMD_GAIN=0.5` (gate-centerline altitude, coordinated turn)
- `-DEXTRA_CONF_FILE=rvv_nav.conf`  → `CONFIG_RISCV_ISA_EXT_V=y`, `CONFIG_RISCV_VECTOR_MAX_LEN=512`

**RVV parity check** (should be non-zero and equal to the FPGA guest — see §4):
```bash
riscv64-unknown-elf-objdump -d build_guest/zephyr/zephyr.elf | grep -coiE '\b(vsetvli|vfmacc|vle32)\b'
riscv64-unknown-elf-readelf -A build_guest/zephyr/zephyr.elf | grep -o 'v1p0\|zvfh1p0\|zvl128b1p0'
```

### 2c. Run it
```bash
MODE=full ISAAC_PY=<env_isaaclab py> bash experiments/rose_nav_cosim/run_cosim.sh
```
`run_cosim.sh` starts the synchronizer, waits for `listening on`, then launches the spike bridge.
**The exact env it sets** (everything else is left at gym_synchronizer defaults — that matters, §3):

| Var | Value | Note |
|---|---|---|
| `ROSE_GYM_ENV` | `WarehouseThrustEnv-v0` | Stage-2: guest→4 motor thrusts (0x20) via MotorThrustAction |
| `ROSE_VISION` | `1` | serve fused inputs {cam_front, tof_cross, lowdim} |
| `ROSE_WH_OBST` | `0` | clean course |
| `ROSE_FREEZE` | `1` | **FREEZE seam: 1 env.step per received thrust** |
| `ROSE_WH_SEED` | `1000` | deterministic gate layout |
| `ROSE_ISAAC_CAMERA` | `1` | keep the 960×540 chase camera for video (+`ROSE_VIDEO_DIR`/`ROSE_FPV_DIR`/`ROSE_VIDEO_DECIM=2`) |
| `ROSE_WH_EPLEN` | `60` | episode length **seconds** |
| `ROSE_MAX_SIM_TIME` | `3000` (full) | grant-time cap, s — generous (see §3) |
| `ROSE_SYNC_WATCHDOG_S` | `300` | stall watchdog |

Spike bridge (`run_spike_rose_lockstep.sh <elf> 1`):
```
ROSE_ISA=rv64gcv_zicntr_zihpm_zfh_zvfh   rose_spike_sim -p 1 \
  --rose-base=0x2000 --rose-irq=3 --rose-dma-base=0x90000000 \
  --rose-nreqrsp=2 --rose-ndma=1 --rose-port=10001 <elf>
```

### 2d. Expected result
`[GATE] passed gate 1/4 @t≈3.3s`, `2/4 @t≈6.6s`, `3/4 @t≈10s`; drone cruises ~1.4 m/s at z≈2.0 m.
Artifacts in `run_out/` (`sync.log`, `chase_frames/`, `fpv_frames/`). Grep `\[GATE\]` in `sync.log`.

---

## 3. Env parameters — and how they map to the FPGA

The config yaml (`deploy/config/config_gym_WarehouseThrustEnv-v0.yaml`) sets `gym_timestep: 0.005`
(200 Hz physics) and **no** `firesim_step`/`firesim_freq`, so the synchronizer defaults apply:

| Parameter | Spike value | Where | Keep on FPGA? |
|---|---|---|---|
| `gym_timestep` (physics dt) | **0.005 s** (200 Hz) | config yaml | **identical** |
| `firesim_freq` (modeled) | **1e9** (1 GHz) | default | **identical** |
| `firesim_step` (cycles/grant) | **10000** (default on spike) | default | FPGA: use a **coarser** value (e.g. 5e6) for throughput — see below |
| FREEZE seam | 1 physics step / thrust | `ROSE_FREEZE=1` | **identical** |
| Guest app + RVV | rose_fused_mpc, 252 RVV insns | build_guest | **identical** (rebuilt for Saturn board) |
| `ROSE_MAX_SIM_TIME` | 3000 (grant-time cap) | env | **SCALE UP** (key gotcha) |

**The one parameter that does NOT port 1:1 is `ROSE_MAX_SIM_TIME`**, and here's the physics:

Under FREEZE, physics advances **per thrust**, not per grant. The guest emits one thrust per
control iteration, which costs `guest_compute_cycles` of SoC time. On the FPGA that's the *real*
Saturn-RVV latency (≫ spike's functional cost). `ROSE_MAX_SIM_TIME` caps **grant-time**
(`count × firesim_step / firesim_freq`), so the flight (physics) time you actually get is:

```
physics_time = MAX_SIM_TIME × (firesim_freq × gym_timestep) / guest_compute_cycles_per_iter
             = MAX_SIM_TIME × (5e6 cycles/physics-step) / guest_compute_cycles_per_iter
```
`firesim_step` **cancels out** — it only trades grant granularity (throughput) for barrier count,
not the physics/grant ratio. So on the FPGA you must raise `MAX_SIM_TIME` by roughly
`guest_compute_cycles / 5e6`.

- Spike (functional, ~few grants/step): `MAX_SIM_TIME=3000` is far more than the ~10 s of physics
  needed → binds on `EPLEN=60` instead.
- FPGA (Saturn RVV, `guest_compute_cycles > 45e6` inferred): to reach ~12 s of physics (3 gates),
  `MAX_SIM_TIME ≈ 120–150`. **My earlier FPGA flight used `MAX_SIM_TIME=30` → only ~2–3 s of
  physics → 0 gates.** That was the whole bug — not the app, not RVV, not drone speed.

---

## 4. Faithful FPGA port (firesim1 U250, Saturn RVV)

**Identical to spike** (verified): the guest app, the RVV code (252 insns, ISA `v1p0 zvfh1p0
zvl128b`, byte-for-byte kernels), the env (`WarehouseThrustEnv-v0`, `ROSE_VISION=1`,
`ROSE_WH_OBST=0`, `ROSE_FREEZE=1`, `ROSE_WH_SEED=1000`), `gym_timestep=0.005`, `firesim_freq=1e9`,
the FREEZE seam, and the wire contract (0x11 DMA / 0x41 / 0x42 / 0x20).

**Changes (compute backend only):**
1. Rebuild `rose_fused_mpc` for the **Saturn/FireSim board** (same flags) instead of
   `spike_riscv64`; stage as the `rose-nav.json` guest. *(Already present & RVV-identical.)*
2. Execute via `firesim infrasetup/runworkload` on the Saturn+RoSE bitstream instead of
   `rose_spike_sim`. Bridge/sync wiring per `rose-firesim1-run-farm`.
3. `firesim_step`: pick 5e6 for throughput (coarse grants amortize the barrier; cancels out of
   the physics ratio).
4. **`ROSE_MAX_SIM_TIME ≈ 150`** (not 30) — the only real retune, per §3. Expect ~45 min wall.

**Success = the same 3/4 gates on seed 1000, on real silicon.** Everything above the compute
backend is held constant, so a discrepancy would be a genuine FPGA/Saturn-vs-ISS finding rather
than a config artifact.

---

## 5. FPGA port attempt result (2026-08-12) — the real blocker is bridge reqrsp delivery

Ran the faithful port on firesim1 (verified-identical guest/RVV/env; `MAX_SIM_TIME=200`, chase
camera). **It flew correctly for the first ~60 control iterations** — drone settling at 2.0 m,
all sensors served, physics advancing, identical to spike's early trajectory. Then it **deadlocked**:

- At settle iter 60 the guest requested `0x14` (raw ToF). The synchronizer **dequeued the request**
  (`Dequeued … cmd: 0x14, num_bytes: 0000`) but the **response never reached the guest** — it sat
  WFI-frozen on `cmd 14` for ~9.5 min (uartlog mtime frozen) while grants spun to 6680.
- The traj CSV (physics steps) stuck at 70 while `Stepping simulation` (grant count) climbed
  freely — **grant-iters advance regardless of physics**, which is why earlier FPGA "flights"
  looked like progress when the physics was frozen. **Always instrument physics via `ROSE_TRAJ_CSV`,
  not the grant-iter count.**

**Diagnosis:** an **FPGA RoSE-bridge reqrsp-delivery stall on `0x14`** under the full per-iter
sensor load (`0x11/0x41/0x42/0x20/0x12/0x15/0x13/0x14`). Spike's functional bridge (`rose_spike_sim`)
delivers all of them; the FPGA bridge (`rosebridge.cc` on Saturn) does not. This is **the** blocker
for on-hardware gate-nav — not config, model, RVV, `firesim_step`, or `MAX_SIM_TIME` (all verified
correct). `net_read_full` fixed the host-side framing (one layer); the remaining bug is the
**FPGA-side reqrsp *response delivery*** (the `0x14`/tof path).

**Next target:** `soc/src/main/cc/rosebridge.cc` reqrsp response path — how a served sensor value
is returned to the guest over MMIO, and why `0x14` intermittently isn't delivered after ~60 iters
of correct delivery. Artifact of the correct pre-stall hover: `scratchpad/.../fpga_settle_chase.mp4`.

### 5a. Root-cause narrowing (2026-08-12, instrumented flight)

Ruled out, each with evidence, by escalating tests:
- **Host framing** — `deploy/hephaestus/tests/test_socket_thread_stress.py` green; `net_read_full` clean.
- **The metasim** — its DMA path is broken/unvalidated in Verilator (`RoseTL*MMIOOnlyConfig` hangs
  on the *first* DMA); a dead end for this bug. (`experiments/rose_bridge_stress` documents this.)
- **Isolated DMA+reqrsp delivery, any payload size** — the `reqrsp_stress` guest on the FPGA passes
  **300 iters** of small *and* 5400-byte (camera-sized) DMA+reqrsp (MMIO datapath). So the bridge's
  isolated delivery is fine.
- **Sync-side serve** — an instrumented flight (`ROSE_SERVE_DEBUG=1` in `gym_synchronizer.py` logs
  each reqrsp serve + txqueue depth) shows **`txq 0->0` for every reqrsp** — the sync serves and
  drains each response; the send path never backs up.

**What the instrumented flight caught at the stall:** the guest's last uart line is `Pushing cmd 42`
(requesting `lowdim`, ch2) with **no subsequent nav telemetry** — so it is WFI on the **0x42
*response***, not computing. The sync served+sent 0x42 (`txq 0->0`), but it **never reached the guest**.
The per-iter sequence at the freeze is **large DMA `0x11` → ch1 reqrsp `0x41` → ch2 reqrsp `0x42`**;
the `reqrsp_stress` that PASSED went DMA→ch2 directly. So the remaining trigger is a **channel-crossing
reqrsp immediately after a large DMA** (0x11→0x41(ch1)→0x42(ch2)), intermittent (~5-6 vision ticks).

**So the bug is the bridge/RTL reqrsp *delivery to the guest*** (rosebridge.cc `fsim_txdata`→MMIO
`rxfifo`→arbiter→channel demux, or the RTL arbiter/rxfifo state after a large DMA + channel switch) —
NOT host framing, NOT the sync, NOT isolated single-channel delivery. **Repro recipe:** instrumented
flight above; next, extend `reqrsp_stress` to `dma(0x11 large) → reqrsp(ch1) → reqrsp(ch2)` to get it
in the fast non-Isaac harness, then instrument the bridge's per-channel delivery.

## 6. Delivery FIXED (2026-08-13) — held-valid handshake; and the NEW blocker: yaw divergence

**The delivery bug is fixed and validated.** Root cause was narrowed to a **per-word drop at the
MMIO→rxfifo enqueue**: `in_valid` was a 1-cycle `Pulsify` into the rxfifo AsyncQueue (gated deq clock);
a missed pulse dropped ~1/100k words, starving the arbiter in `sLoad`. Fix (`RoSEBridgeModule.scala`):
a **held-valid handshake** — `in_valid_held` register SET on the MMIO write, CLEARED on skid `enq.fire`;
driver `send()` polls `in_valid_pending` until clear. Diagnostic `in_enq_count` (rxfifo enq.fire count)
lets the `[ARB]` heartbeat print `enq` vs `tx`. **Validated on bitstream `2026-08-13--11-05-05` (30 MHz):
flew 2020 iters / 91,053 words with `enq==tx` throughout (zero drops)** vs prior max 45–488. Details +
tooling in `experiments/rose_arb_deadlock/` and memory `rose-fpga-arbiter-deadlock`.

**With delivery fixed, forward-nav runs for the first time — and reveals a NEW, distinct blocker:**
the drone **flies but misses the gates** (`gates=0` on seed 1000). Matched traj CSVs vs spike show yaw
is **bit-exact during settle** and diverges to a **constant ~0.5 rad (~30°) the moment vision engages**
(~tick 210), so the drone flies ~30° off the +y corridor (diagonal −x drift, ~0.35 m/s vs spike 1.2).
The controller/thrusts are correct; the heading reference is biased. Leading cause: the fused-vision
fp16 tail (`lstm_f16`) on Saturn **native Zvfh hardware fp16** vs spike's fp16 **emulation** — the
"genuine Saturn-vs-ISS finding" §4 anticipated. **This retires the delivery hypothesis of §5** and moves
the investigation to on-Saturn vs on-spike numerical fidelity of the vision inference. Memory:
`rose-fpga-nav-yaw-divergence`.
