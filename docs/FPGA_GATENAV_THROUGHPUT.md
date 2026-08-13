# FPGA gate-nav — sim-throughput characterization (successful 3-gate run)

Measured on the recorded seed-1000 flight (`Saturn+RoSE @ 30 MHz` on firesim1 U250 ⟷ IsaacLab
on garden over LAN). Reproduce with `python3 experiments/rose_arb_deadlock/analyze_throughput.py`
(parses the committed wall-clock heartbeat log + physics trajectory).

## Headline

| Metric | Value |
|--------|-------|
| **FPGA-productive duty** | **85.3 %** (effective 25.6 MHz of the 30 MHz fabric) |
| **Co-sim sync overhead** | **14.7 %** (28.7 ms sync stall per grant) |
| Grant-iter throughput | 5.12 grants/s |
| Physics-step throughput | 1.34 steps/s (744 ms/physics-step) |
| Physics real-time factor | 0.0067× (1 physics-second per ~149 wall-seconds) |
| Grants per physics step | 3.81 (FREEZE + the 12-sensor per-step exchange) |
| Wall to gate 1 / 2 / 3 | 9.1 / 18.2 / 25.6 min |

Productive window: physics 0.1 s → 12.6 s (2500 steps, 9520 grants) over 1860 s wall,
excluding boot (~90 s to `listening`+`infrasetup`) and the post-flight tail plateau.

## How the numbers are derived

The co-sim is lockstep: each **grant** advances the FPGA `ROSE_FIRESIM_STEP = 5e6` target
cycles, then the bridge blocks until IsaacLab returns the next sensor frame + thrust. With the
target modeled at `ROSE_FIRESIM_FREQ = 1 GHz`, one grant = **5 ms of target time = 5 ms of
physics** dispatched, and FireSim maps 1 target cycle → 1 host FPGA cycle (single-clock target).

- **FPGA compute per grant** = 5e6 cycles ÷ 30 MHz = **166.7 ms** (unavoidable, this is the SoC
  actually running on fabric).
- **Measured wall per grant** = 1860 s ÷ 9520 = **195.4 ms**.
- ⇒ **sync stall per grant** = 195.4 − 166.7 = **28.7 ms** — the co-sim seam: IsaacLab
  `env.step` (physics + FPV render every other grant) + bridge (de)serialization + the LAN
  round-trip to garden. That is the entire non-FPGA cost, and it is **~15 %** of wall time.

Equivalently: 9520 grants × 5 ms = 47.6 s of target time simulated in 1860 s wall →
effective host clock 47.6 s × 1 GHz ÷ 1860 s = **25.6 MHz**, i.e. **85 %** of the 30 MHz fabric
is doing useful target work; the rest is the seam.

## Reading it

- **The FPGA, not the seam, is the bottleneck.** At 30 MHz the SoC needs 167 ms to advance one
  5 ms co-sim step; the whole distributed co-sim (physics, GPU render, LAN to another machine)
  adds only 29 ms on top. The split FPGA↔GPU topology is *not* what makes this slow — the low
  fabric clock is. A 60 MHz build would roughly halve wall-to-gate at the same ~15 % seam.
- **Why 3.81 grants per physics step (not 1):** under `ROSE_FREEZE` physics advances once per
  thrust, but each physics step drives a full sensor exchange — the 12-packet suite (camera
  DMA + 11 reqrsp sensor reads across ch1/ch2) plus the guest's compute — spanning several
  bridge grants before the next thrust is emitted.
- **Real-time factor 0.0067×** is expected for cycle-accurate FPGA co-sim at 30 MHz with a
  1 GHz target model (a 33× target:host clock ratio floor, further divided by grants/step).

## Caveats

- Wall timing is from the harness 32 s heartbeat (`traces/fpga_gatenav_3gate_heartbeat.log`);
  the per-grant stall is an average, not a per-exchange distribution (the in-bitstream `[ARB]`
  heartbeat gives cycle-level detail if a finer breakdown is needed).
- This is the **30 MHz** MMIO-camera-over-DMA build. The 60 MHz DMA sibling (task #105) would
  be the apples-to-apples speed artifact; not built for this run.
- Duty/overhead assume the 1:1 target↔host cycle mapping FireSim uses for this single-clock
  Vitis U250 target; the 30 MHz figure is the timing-closed fabric clock (WNS +0.060 ns).
