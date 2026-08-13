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

## POST-FIX finding: gate-nav yaw divergence (delivery is no longer the blocker)

With delivery fixed, the flight runs its FORWARD-NAV phase for the first time (it always
deadlocked at settle before). Result: the drone **flies but misses the gates** — it tracks
diagonally instead of up the +y corridor, `gates=0` on seed 1000 where spike passes 3/4.
Matched traj CSVs (`traces/fpga_gatenav_gn_traj.csv` vs spike
`experiments/rose_nav_cosim/run_out/traj.csv`) show yaw is **bit-exact to spike during
settle** and diverges to a **constant ~0.5 rad (~30°) only once vision engages (~tick 210)**
— see `traces/postfix_yaw_divergence.txt`. Leading cause: the fused-vision fp16 tail
(`lstm_f16`) on Saturn **native Zvfh hardware fp16** vs spike's fp16 **emulation**. This is a
distinct, numerical issue — see memory `rose-fpga-nav-yaw-divergence.md`.

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
