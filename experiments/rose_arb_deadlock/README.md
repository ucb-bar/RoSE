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

## The fix (in progress)

The robust fix is at the rxfifo enqueue: `in_valid` is a 1-cycle `Pulsify`
(`RoSEBridgeModule.scala` ~L482); a missed pulse / AsyncQueue CDC edge drops a word.
Replace it with a **held-valid handshake** (set on MMIO write, clear on `enq.fire`) +
have the driver `send()` poll an enqueue-status readback before the next word — this
guarantees no drop. Needs a bitstream rebuild (~1–1.5 hr at 30 MHz).

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
- `flight_dma.sh` — same but `ROSE_DMA_RX=1` (runtime DMA datapath).

## Traces

- `traces/hang_60mhz_mmio_run2.txt` — 60 MHz MMIO bitstream, hung frame ~104 on 0x42.
- `traces/hang_30mhz_deeperfifo.txt` — 30 MHz deeper-FIFO bitstream, hung frame ~50 on
  0x14 with all 3 stall counters frozen (the decisive "arbiter starved" evidence).

See also the memory note `rose-fpga-arbiter-deadlock.md` and the in-tree instrumentation
in `soc/src/main/{cc/rosebridge.cc,cc/rosebridge.h,scala/RoSEBridgeModule.scala,scala/RoSEBridge.scala,scala/RoSEIO.scala,scala/RoSEBridgePort.scala,scala/RoSEAdapter.scala}`.
