# RoSE FPGA gate-nav stall — arbiter-delivery deadlock: diagnostics + fix tooling

Dedicated space for the tooling/test infra used to root-cause and fix the intermittent
FPGA gate-nav mid-flight stall ("guest WFI-hangs on a served reqrsp"). This is the
diagnostic harness + captured traces + the reproducible flight/build scripts.

## Root cause (confirmed)

The stall is an **intermittent single-word drop in the host→FPGA per-word MMIO path**.
A reqrsp payload word (~1 in 100k) never reaches the FPGA rxfifo, so the arbiter waits
forever in `sLoad` for it while the sync keeps granting tokens and the host driver shows
its queues empty (`brxq=0 txd=0`). It is NOT the arbiter itself, the channel-FIFO depth,
`can_advance`, framing, routing, the host RX thread, or the sync.

### How it was pinned

1. **`ROSE_ARB_TRACE` driver heartbeat** (added to `soc/src/main/cc/rosebridge.cc`
   `tick()`): reads the in-bitstream `genROReg` arbiter counters every N ticks and prints
   `[ARB] sh=.. tx=.. rx0=.. rx1=.. ch2~=.. bf=.. cyc=.. budg=.. last=0x.. nb=.. brxq=.. txd=.. irx=.. iadv=.. lrx=..`.
   Forwarded to the remote driver via `runtime_config.py` (same path as `ROSE_SYNC_HOST`).
   Driver-only — no bitstream rebuild to read counters already in the bitstream.
2. At a live hang: `sh/tx/rx*` all FROZEN while `cyc` advances + `budg` oscillates
   (grants flowing); `brxq=0 txd=0` (host delivered everything to the FPGA rxfifo);
   `last=0x14` (or 0x42 — varies by run: whichever response's word is dropped).
3. **3 new arbiter stall counters** (`arb_counter_idle_rxstall` / `idle_advstall` /
   `load_rxstall`, added to both bundle defs + both arbiters + genROReg + the driver
   heartbeat) were ALL FROZEN at the hang (none climbing) => the arbiter is not stalling
   on `rx.ready` (b) or `can_advance` (a) — it is STARVED, stuck in `sLoad` awaiting a
   payload word that never arrives. That localizes the loss upstream of the arbiter, in
   the per-word MMIO rxfifo enqueue.

`analyze_arb.py <uartlog>` parses a captured trace and classifies the flatline.

## The fix — DONE & VALIDATED (2026-08-13)

The fix is at the rxfifo enqueue: `in_valid` was a 1-cycle `Pulsify`
(`RoSEBridgeModule.scala` ~L482); a missed pulse / AsyncQueue CDC edge dropped a word.
Replaced with a **held-valid handshake** — `in_valid_held` register SET on the MMIO write,
CLEARED on skid `enq.fire`, so the word is held on the enq port until provably accepted
(no missable pulse); driver `send()` polls `in_valid_pending` until clear. A diagnostic
`in_enq_count` counter (rxfifo `enq.fire` count, genROReg'd) lets the `[ARB]` heartbeat
print `enq` vs `tx`.

**VALIDATION (bitstream `2026-08-13--11-05-05`, 30 MHz):** flew the instrumented gate-nav to
**iter 2020 / 91,053 words with `enq==tx` the entire flight (`enq-tx=0` → ZERO drops)**,
where every prior config hung at 45–488 iters. The one transient `enq-tx=10` at iter 620
drained to 0 next heartbeat (normal AsyncQueue pipeline occupancy, not a drop). The stall
bug is dead. Trace: `traces/heldvalid_zerodrops_91kwords.txt`.

## POST-FIX finding: gate-nav yaw divergence → ROOT-CAUSED to a FROZEN camera (NOT fp16)

With delivery fixed, the flight ran its FORWARD-NAV phase for the first time (it always
deadlocked at settle before). Initial result: the drone **flew but missed the gates** — it
tracked diagonally instead of up the +y corridor, `gates=0` on seed 1000 where spike passes 3/4.

The first hypothesis was a fused-vision fp16-tail numerical divergence (Saturn Zvfh HW fp16 vs
spike fp16 emulation). **That was WRONG.** A guest→sync **VDIAG echo** (per-vision-tick FNV hash
of the three model inputs {cam,tof,lowdim} + fp16 outputs, sent as an unknown cmd 0x21 over the
working data channel since guest printk isn't captured on FPGA) proved the model *inputs* differ,
not the compute: the **camera was FROZEN** on its first frame (`ba24d8a5` for all 23 ticks) while
spike had 113 unique frames; tof+lowdim were fresh. Vision ran on a dead image → wrong steer.

**ROOT CAUSE (confirmed via the DMADIAG curr_counter probe, cmd 0x24):** a DMA **address
mismatch**. The FPGA RTL `RoSEDMA` writes camera frames to `DMA_address=0x88000000`
(`RoSEConfigs.scala DstParams`) but the guest DT `dma-base-address` was `0x90000000` (matching
spike `--rose-dma-base`), never synced to the RTL. The guest read `0x90000000` (never written) →
frozen frame forever, while the DMA wrote fresh frames to `0x88000000` the whole time (proven by
`curr_counter` cleanly ping-ponging 5400↔0). The DMA refresh was never broken; only the read
address was wrong.

**FIX (guest-only, no bitstream rebuild):** (a) DT `dma-base-address` → `0x88000000` (read where
the DMA writes); (b) `rose_dma_buffer` selects the just-filled ping-pong half from `curr_counter`
(race-free vs the ISR STATUS-bit3 latch). Commits: RoSE `5f3042d` → xpu-rt `0c5315b`
(zephyr-chipyard-sw `77aaa6b`) → zephyr-rose `b7cd18a`. Long-term cleanup: sync the RTL
`DMA_address` to `0x90000000` + rebuild so RTL/DT agree (cosmetic; current pairing is validated).

## ✅ GOAL REACHED — full 3-gate navigation reproduced on FPGA with recorded video

With the camera unfrozen, the FPGA drone flies **spike's trajectory** up the warehouse corridor
and clears **3/4 gates** — an exact match to the spike reference (both miss gate 4):

| Gate | Spike (reference) | FPGA (Saturn+RoSE, firesim1 U250) |
|------|-------------------|-----------------------------------|
| 1/4  | 3.33s | **3.27s** @ (-8.27,+8.13) |
| 2/4  | 6.65s | **7.07s** @ (-8.57,+12.15) |
| 3/4  | 10.14s | **9.92s** @ (-7.95,+16.13) |

Camera confirmed unfrozen: `uCam` climbs continuously, per-tick frame hashes change like spike.
The entire nav stack — camera over DMA, VL53L5CX multizone ToF + IMU/flow over reqrsp, the
fused-vision RVV policy on **real Saturn fabric**, in lockstep with IsaacLab on garden over LAN —
navigates autonomously gate-to-gate.

**Recorded video:** `docs/media/fpga_gatenav_3gate.mp4` — chase cam + the drone **FPV camera**
inset (the actual policy vision input), 960×540, 63 s. Gate-crossing stills committed under
`stills/fpga_gate{1,2,3}.jpg`. A chase-only variant is reproducible via `make_gatenav_video.sh`.

**Full experimental setup** (every host, config, env var, command): `docs/FPGA_GATENAV_REPRODUCE.md`.
**Sim-throughput characterization** (~85 % FPGA duty, ~15 % co-sim seam): `docs/FPGA_GATENAV_THROUGHPUT.md`
+ `analyze_throughput.py` (over `traces/fpga_gatenav_3gate_heartbeat.log`).

Superseded attempts (kept for the record): deeper per-channel rx FIFO (8→256, wrong
layer — the stall counters proved the arbiter wasn't the problem); DMA-RX datapath (the
driver was refactored to select the datapath at runtime via `ROSE_DMA_RX=1`, and that
works, but the existing DMA bitstreams do NOT boot the guest — the DMA RTL is unvalidated).

## Scripts

- `analyze_arb.py` — classify an `[ARB]` trace (which delivery stage flatlined).
- `rebuild_driver_mmio.sh` / `rebuild_driver_dma.sh` — driver-only recompile for the
  MMIO / DMA Saturn config (reuses generated-src; ~minutes; no bitstream rebuild).
- `build_bitstream.sh` — full `firesim buildbitstream` for the fix (elaboration then Vivado).
- `flight_arbtrace.sh` — full WarehouseThrustEnv gate-nav co-sim on the split
  (firesim1 U250 FPGA + garden Isaac) with `ROSE_ARB_TRACE` + chase-cam video.
- `flight_validate.sh` — shorter run (past the ~frame-104 hang point) to validate a fix.
- `flight_validate_heldvalid.sh` — the delivery-FIX validation flight (`ROSE_ARB_TRACE`,
  reads `enq` vs `tx`); this is the run that proved zero drops over 91k words.
- `flight_dma.sh` — same but `ROSE_DMA_RX=1` (runtime DMA datapath).
- `gatenav_flight.sh` — the gate-nav DELIVERABLE flight on the fixed bitstream: exact spike
  3/4-gate config (seed 1000, FREEZE, vision) + `ROSE_MAX_SIM_TIME=250` + `ROSE_TRAJ_CSV`
  (REAL physics; grant-iters != physics under FREEZE) + chase-cam. This is what surfaced the
  yaw divergence.
- `make_gatenav_video.sh` — assemble the chase-cam JPEGs into an mp4.

## Traces

- `traces/hang_60mhz_mmio_run2.txt` — 60 MHz MMIO bitstream, hung frame ~104 on 0x42.
- `traces/hang_30mhz_deeperfifo.txt` — 30 MHz deeper-FIFO bitstream, hung frame ~50 on
  0x14 with all 3 stall counters frozen (the decisive "arbiter starved" evidence).
- `traces/heldvalid_zerodrops_91kwords.txt` — the delivery-FIX validation: `enq==tx` over
  2020 iters / 91k words (zero drops). See `traces/_heldvalid_README.txt`.
- `traces/postfix_yaw_divergence.txt` + `traces/fpga_gatenav_gn_traj.csv` — the post-fix
  gate-nav yaw divergence (FPGA diagonal flight vs spike straight; ~0.5 rad once vision engages).

See also the memory note `rose-fpga-arbiter-deadlock.md` and the in-tree instrumentation
in `soc/src/main/{cc/rosebridge.cc,cc/rosebridge.h,scala/RoSEBridgeModule.scala,scala/RoSEBridge.scala,scala/RoSEIO.scala,scala/RoSEBridgePort.scala,scala/RoSEAdapter.scala}`.
