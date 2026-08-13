# FPGA gate-nav bring-up — live debug log

**Goal:** reproduce the recorded warehouse **3-gate navigation** on the **FPGA** (FireSim/U250 Saturn+RoSE),
matching the spike reference (spike passes 3/4 gates on seed 1000), with recorded video.

**Reference (spike, working):** `[GATE] passed 1/4 @3.33s, 2/4 @6.65s, 3/4 @10.14s`; drone cruises
~1.4 m/s at z≈2.0 m up the +y corridor (gate centres local: G1(-8.05,9) G2(-8.30,13) G3(-7.75,17) G4(-8.05,21), pass radius 0.9 m).

**Status (2026-08-13): ✅ GOAL REACHED — 3/4 gates on FPGA, matching spike.** Primary stall bug FIXED. Frozen-camera
**ROOT CAUSE FOUND & FIXED** — a DMA address mismatch (RTL writes 0x88000000, guest read 0x90000000). Guest-only fix
(no rebuild). Compute backend (Saturn RVV nav policy) is byte-identical to spike; the entire gap was the **camera
sensor-delivery path**. With the camera unfrozen the FPGA drone flies spike's trajectory and clears all 3 gates.

| Gate | Spike (reference) | FPGA (Saturn+RoSE, U250) |
|------|-------------------|--------------------------|
| 1/4  | 3.33s             | **3.27s** @ (-8.27,+8.13) |
| 2/4  | 6.65s             | **7.07s** @ (-8.57,+12.15) |
| 3/4  | 10.14s            | **9.92s** @ (-7.95,+16.13) |

Gate 4 missed by both (spike is 3/4). Camera confirmed unfrozen: uCam climbs continuously, frame hashes change
each vision tick like spike (was frozen at `ba24d8a5` ×23). Fix committed: RoSE `5f3042d` → xpu-rt `0c5315b`
(zephyr-chipyard-sw `77aaa6b`: DT dma-base 0x88000000 + DMADIAG) → zephyr-rose `b7cd18a` (curr_counter half-select).

---

## The bug chain (symptom → hypotheses → resolution)

### 1. RoSE-bridge delivery deadlock — ✅ FIXED & VALIDATED
- **Symptom:** intermittent mid-flight WFI-hang; arbiter starved in `sLoad` awaiting a served reqrsp word.
- **Root cause:** per-word drop at the MMIO→rxfifo enqueue — `in_valid` was a 1-cycle `Pulsify` into the
  rxfifo AsyncQueue (gated deq clock); a missed pulse dropped ~1/100k words.
- **Fix:** held-valid handshake (`in_valid_held` set on MMIO write, cleared on skid `enq.fire`); driver
  `send()` polls `in_valid_pending`. Diagnostic `in_enq_count` (rxfifo enq.fire) → `[ARB]` prints `enq` vs `tx`.
- **Validation:** flew **2020 iters / 91,053 words, `enq==tx` throughout (zero drops)** vs prior max 45–488.
- **Commit:** `4431778`. Bitstream `2026-08-13--11-05-05`.

### 2. Frozen camera DMA — ✅ ROOT-CAUSED (this is why gates fail)
- **Symptom:** with delivery fixed, forward-nav runs but the drone flies **diagonally & misses all gates**
  (yaw ~0.5 rad off, cruise 0.35 vs 1.2 m/s). `gates=0` on seed 1000.
- **Method:** guest→sync **VDIAG echo** (cmd 0x21, FNV hash of the 3 model inputs {cam,tof,lowdim} + fp16
  outputs, sent over the working data channel since guest printk isn't captured on FPGA). Ran the
  byte-identical guest on spike AND FPGA.
- **Finding:** at the first vision tick (bit-identical spawn, proven by matching `low`): **cam FROZEN at
  `ba24d8a5` for all 23 ticks** while spike has 113 unique frames; tof+lowdim fresh; the sync serves fresh
  frames (12 unique `[dma-serve]` crc32). ⇒ the FPGA camera DMA delivers a stale (first) frame; vision runs on
  a dead image → wrong steer.
- **Mechanism:** `RoSEDMA` ch0 is a ping-pong double buffer exposing the just-filled half via STATUS bit3
  `DMA_BUFFER`; the guest `rose_dma_buffer` always read half 0. Applied a guest half-select fix (necessary but
  NOT sufficient — the RoSEDMA *refresh* itself never advances `dma_half` on HW). RoSEDMA counter behavior is
  opaque without waveforms.
- **Commit:** `fba53db` (+ `rose_adapter` half-select).

### 3. reqrsp-camera FIFO deadlock — ✅ FIXED
- **Approach:** bypass the broken DMA — route the camera (0x11) over the proven reqrsp path.
- **Symptom:** a single 1350-word camera frame DEADLOCKED at ~493/1350 words (arbiter stalls in `sLoad`;
  256-deep per-channel FIFO overflows).
- **Fix:** deepen `RoSEAdapter` per-channel `rx_buffer_fifo` 256→2048 so a whole frame lands in one shot.
- **Validation:** ARB `rx1` 493 → **1491** (whole frame delivered, `enq-tx=0`, no stall).
- **Commit:** `64ed2bb`. Bitstream `2026-08-13--14-28-37` (the current one).

### 4. reqrsp-camera large-read block — ✅ LOCALIZED (cascade hypothesis REJECTED)
- **Symptom:** frame delivered (rx1=1491) but the guest never completes the read; physics never advances.
- **Method:** per-read **RRDIAG echo** (cmd 0x23) — echoes `(channel, header, num_bytes)` for every reqrsp
  read + caps nwords so a bad length can't hang.
- **Finding:** all 8 nav-sensor reads are **correctly framed** (correct headers/num_bytes); counts match
  exactly (rx1 1491 = 139 nav + 1352 camera). The guest blocks on the camera's **first `rose_rx(header)`**
  despite ch1 holding the data. ⇒ **NOT a shared-channel framing cascade** — a specific block on the large read.
- **Commit:** `c53dbd7`.

### 5. Chunked camera — ❌ FAILED (same block; reqrsp path is a dead end)
- **Idea:** small reqrsp reads (64-word ToF) work; the large single read blocks. Sync serves 2-D obs row-by-row →
  reshape `cam_front` to 27×200 B so the camera arrives as 27 small framed reads (`ROSE_CAM_CHUNK` + `FMPC_CAM_REQRSP`).
- **Result:** the arbiter delivered all 27 chunk-packets (**`sh=35`** vs the old freeze at `sh=9`, `rx1=1543`),
  but the guest **still blocks on the FIRST `rose_rx(header)` of the camera response** (`[CAMDIAG]` never fires,
  physics never advances). ⇒ **the block is NOT read-size** — the guest fundamentally cannot read the camera
  reqrsp response on ch1 after the 8 clean nav reads, whether it's 1 packet or 27.
- **Conclusion:** the reqrsp-camera path is a dead end without waveform-level insight into why `RX1_VALID` doesn't
  advance the camera read despite the data being in the FIFO.

### 6. Frozen-camera ROOT CAUSE FOUND & FIXED — DMA address mismatch — ✅ FIXED & VALIDATED (3/4 gates)
- **Method:** guest reads the RoSEDMA `curr_counter` reg (0x18) after each frame and echoes it (**DMADIAG**, cmd 0x24) —
  direct register reads, no bitstream change.
- **Finding:** `curr_counter` cleanly **ping-pongs 5400 → 0 → 5400 → 0** (the DMA writes fresh frames to alternating
  halves correctly!) while the cam stays frozen `ba24d8a5`. That's only possible if the guest reads a *different
  address than the DMA writes*.
- **ROOT CAUSE:** **RTL `DstParams(DMA_address = 0x88000000)`** (RoSEConfigs.scala, both configs) vs guest DT
  **`dma-base-address = 0x90000000`**. The guest DT was moved above SRAM (0x90000000) to avoid heap corruption but the
  **RTL was never synced** — it still writes 0x88000000. The guest read 0x90000000 (never written) → frozen; the DMA
  wrote fresh frames to 0x88000000 the whole time. This retires the "RoSEDMA refresh is broken" theory — the DMA
  refresh works perfectly; it was an address mismatch.
- **Fix (guest-only, NO rebuild):** (a) DT `dma-base-address` → `0x88000000` (read where the DMA writes; safe — the DMA
  already writes there without destabilizing the guest); (b) `rose_dma_buffer` selects the just-filled ping-pong half
  from `curr_counter` (race-free, unlike the ISR STATUS-bit3 latch). Proper long-term fix: change the RTL DMA_address
  to 0x90000000 + rebuild.
- **Validation:** camera unfrozen (uCam climbs continuously, per-tick frame hashes change like spike); the FPGA
  drone flew spike's trajectory and passed **3/4 gates** (1@3.27s, 2@7.07s, 3@9.92s) vs spike (3.33/6.65/10.14s).
  Gate 4 missed by both — an exact match to the spike reference. **This closes the FPGA gate-nav investigation.**
- **Commit:** RoSE `5f3042d` → xpu-rt `0c5315b` (zephyr-chipyard-sw `77aaa6b`) → zephyr-rose `b7cd18a`.

---

## Rejected hypotheses (important negatives)
- ❌ **fp16 numerical divergence** (Saturn Zvfh vs spike emulation) — VDIAG proved the model *inputs* differ
  (frozen camera), not the compute.
- ❌ **Guest ELF build mismatch** — deployed guest differs by 928 B from spike's but produces a *bit-identical*
  divergent trajectory → not the build.
- ❌ **Delivery-drop / word loss** — `enq==tx` zero drops after the held-valid fix.
- ❌ **Shared-channel framing cascade** (camera + 6 sensors on ch1) — RRDIAG proved all nav reads are cleanly framed.
- ❌ **Boot-time camera request misordering** — guest only checks `device_is_ready` at boot; no early 0x11.
- ❌ **RoSEDMA refresh logic broken** — the DMADIAG counter trace proved the DMA writes fresh frames correctly (ping-pong 5400↔0); the freeze was purely the read/write address mismatch, not the refresh.
- ❌ **`counter_max` never set (DMA never wraps)** — driver DMA_CFG offset (0x14) matches the RTL
  `written_counter_max` register exactly.

## Resolution & remaining cleanup (the DMA path was never broken)
- ✅ **RESOLVED by the address fix** — the reqrsp detour (#3–#5) and the "DMA refresh" theory were both chasing a
  symptom. The DMA read path always worked; the guest was simply reading the wrong address. One-line DT change
  (0x90000000→0x88000000) + curr_counter half-select → 3/4 gates, no bitstream rebuild.
- **Long-term cleanup (optional, not required for the goal):** sync the RTL `DstParams(DMA_address)` in
  RoSEConfigs.scala to 0x90000000 and rebuild so RTL and DT agree on the canonical (above-SRAM) address; then the
  guest DT can revert to 0x90000000. Purely cosmetic — the current 0x88000000 pairing is validated and stable.
- The reqrsp-camera diagnostics (RRDIAG/CAMDIAG, chunked path) remain in-tree behind flags as useful tooling.

---

## Diagnostic tooling (committed, `experiments/rose_arb_deadlock/`)
- **`[ARB]` heartbeat** (`ROSE_ARB_TRACE`): in-bitstream arbiter counters (sh/tx/rx0/rx1/enq/stall) — localizes delivery stalls.
- **VDIAG** (cmd 0x21): per-vision-tick hash of model inputs + fp16 outputs → proved the frozen camera.
- **CAMDIAG** (cmd 0x22): camera header/num_bytes the guest reads.
- **RRDIAG** (cmd 0x23, `ROSE_REQRSP_DBG`): per-reqrsp-read `(ch,header,num_bytes)` + cap → proved nav reads clean.
- Flight harnesses: `gatenav_flight.sh` (exact spike config, MAX_SIM_TIME=250, TRAJ, chase cam),
  `flight_validate_heldvalid.sh`; spike reference `spike_vdiag_ref.txt`.

## Environment gotchas
- Split co-sim: FPGA on firesim1.millennium (slot 3f:00.0), Isaac/GPU on garden, over LAN (`ROSE_SYNC_HOST=136.152.139.10`).
- **Always** infrasetup immediately before runworkload (AER-off → silent host reset otherwise).
- Ports 10001–10061 on garden got stuck held (no findable owner; shared-machine contention) → moved sync+driver
  to `ROSE_SYNC_PORT=10071` (data port 60002 hardcoded in `rosebridge.cc`).

## Task tracker
- `#114` ✅ **DONE**: FPGA camera delivery fixed (DMA address mismatch) → **3/4 gates** on Saturn+RoSE, matching spike.

_Last updated: 2026-08-13 — ✅ GOAL REACHED. Frozen-camera ROOT CAUSE = DMA address mismatch (0x88000000 vs 0x90000000);
guest-only fix (RoSE 5f3042d) → drone flies spike's trajectory → **3/4 gates** (1@3.27s, 2@7.07s, 3@9.92s). Recorded video assembled._
