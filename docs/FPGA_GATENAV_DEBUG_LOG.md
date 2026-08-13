# FPGA gate-nav bring-up — live debug log

**Goal:** reproduce the recorded warehouse **3-gate navigation** on the **FPGA** (FireSim/U250 Saturn+RoSE),
matching the spike reference (spike passes 3/4 gates on seed 1000), with recorded video.

**Reference (spike, working):** `[GATE] passed 1/4 @3.33s, 2/4 @6.65s, 3/4 @10.14s`; drone cruises
~1.4 m/s at z≈2.0 m up the +y corridor (gate centres local: G1(-8.05,9) G2(-8.30,13) G3(-7.75,17) G4(-8.05,21), pass radius 0.9 m).

**Status (2026-08-13):** primary stall bug FIXED; gate reproduction blocked by a *chain* of FPGA camera-delivery
bugs — 3 fixed/root-caused, currently testing a 4th fix (chunked camera). Compute backend (Saturn RVV nav
policy) is byte-identical to spike; the entire gap is the **camera sensor-delivery path**.

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

### 6. Pivot back to the DMA path — ⏭ THE REMAINING VIABLE ROUTE
- **Key asymmetry:** on the **DMA** path the guest **reads the frame fine** (physics *advanced* in the frozen-camera
  runs — the guest read `dma_base`, ran the model, sent thrusts); only the **refresh** is broken (stale frame).
  On the **reqrsp** path the guest can't read the response at all. So the DMA path is much closer to working.
- **Fix:** make the `RoSEDMA` deliver a fresh frame each request — reset the write counter to 0 on each arm
  (DMA_CFG write) so every frame overwrites `dma_base` (robust to whatever the ping-pong counter does), OR
  single-buffer (wrap at `counter_max`). Needs a targeted RTL change + a 3rd bitstream rebuild (~2 hr).
  RoSEDMA counter timing is opaque from logs — this is the point where FireSim metasim waveforms would help.

---

## Rejected hypotheses (important negatives)
- ❌ **fp16 numerical divergence** (Saturn Zvfh vs spike emulation) — VDIAG proved the model *inputs* differ
  (frozen camera), not the compute.
- ❌ **Guest ELF build mismatch** — deployed guest differs by 928 B from spike's but produces a *bit-identical*
  divergent trajectory → not the build.
- ❌ **Delivery-drop / word loss** — `enq==tx` zero drops after the held-valid fix.
- ❌ **Shared-channel framing cascade** (camera + 6 sensors on ch1) — RRDIAG proved all nav reads are cleanly framed.
- ❌ **Boot-time camera request misordering** — guest only checks `device_is_ready` at boot; no early 0x11.
- ❌ **`counter_max` never set (DMA never wraps)** — driver DMA_CFG offset (0x14) matches the RTL
  `written_counter_max` register exactly.

## The path forward (reqrsp exhausted → DMA refresh)
- **PRIMARY: RoSEDMA refresh fix** — the guest already reads DMA frames correctly; only the refresh is stale.
  Reset the DMA write counter on each arm so every frame overwrites `dma_base`. Targeted RTL + 3rd rebuild (~2 hr).
  Best confidence, since the DMA read path is proven to work.
- Fallback: dedicated reqrsp channel (parametric `DstParams`) — but the reqrsp *read* itself blocks, so this is
  lower-confidence than the DMA fix.

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
- `#114` (open): fix FPGA camera delivery → pass gates. Candidate: chunked camera (testing), else RoSEDMA refresh / dedicated channel.

_Last updated: 2026-08-13 — chunked camera FAILED (same block, reqrsp dead end); pivoting to the RoSEDMA refresh fix (needs 3rd rebuild)._
