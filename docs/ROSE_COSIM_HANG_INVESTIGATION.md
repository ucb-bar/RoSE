# Co-sim "~236-step" hang — investigation & SoC-side ruling

Tracks the intermittent hang where the RoSE lockstep co-sim freezes (Zephyr guest at
~0% CPU, synchronizer pinned ~113%), historically observed around step ~220–236 on
clean/still hovers. This note records what was ruled out, the evidence, and the
instrumentation added to catch it next time.

## TL;DR

- **The SoC is ruled out as the deadlock source.** The lockstep barrier releases on
  guest **mtime**, and mtime keeps advancing even while the guest sits in **WFI** — so a
  fully-idle guest still completes its grant. The device-level trace confirms the SoC
  streams clean `grant → step → ack` cycles throughout. (Details below.)
- **The hang is intermittent and host/scene-correlated, not a deterministic step.** On a
  freshly-cleaned host it did **not** reproduce: a clean-hover `SensorEnv` run reached
  **685** steps and the `MultiSensorEnv`+striped-walls scenario — which hung at ~220
  twice earlier the same day — reached **605+**, both healthy. Earlier hangs coincided
  with a host loaded with orphaned Isaac/GPU processes (later killed).
- **TACIT tracing is validated and available** for SoC-side instrumentation (below).
- A **synchronizer stall watchdog** was added so the next occurrence self-attributes
  (Isaac `env.step` vs the SoC-ack busy-spin) instead of hanging silently.

## Why the SoC (WFI) is NOT the deadlock

The lockstep loop lives in `soc/src/main/cc/rose_spike/rose_spike_sim.cc`
(`rose_sim_t::idle()`): on each grant it sets `budget_target = mtime_now() + budget/
INSNS_PER_RTC_TICK`, calls `step(INTERLEAVE)`, and acks once `mtime_now() >=
budget_target`. `mtime_now()` is the guest CLINT `time`.

The natural suspicion is: *"if the guest executes WFI, it retires nothing, mtime freezes,
the grant never acks, the synchronizer spins forever."* That is **not** how this Spike
build behaves. In the vendored `riscv-isa-sim`:

- `sim_t::step` (`riscv/sim.cc`) advances `current_step += steps` by the **scheduled**
  quantum and ticks the clint by `INTERLEAVE / INSNS_PER_RTC_TICK` **regardless of how
  many instructions actually retired**.
- The WFI catch (`riscv/execute.cc`, `wait_for_interrupt_t`) retires exactly **one**
  instruction per `processor_t::step` call and returns to the outer loop.

So an idle guest still advances mtime (~`INTERLEAVE/INSNS_PER_RTC_TICK` per `idle()`
call), `budget_target` is reached, and the grant is acked. Empirically, the device
debug log (`ROSE_SPIKE_DEBUG=1`) shows unbroken `grant received → step start → ack step`
cycles across hundreds of steps on healthy runs. **The guest being idle/in-WFI does not
stall the barrier.**

> Correction: an earlier note (`ROSE_FLIGHT_CONTROLLER_THREADING.md`) speculated PhysX
> rigid-body sleep and/or a WFI/mtime freeze. The WFI/mtime part is disproven here. A
> defensive `sleep_threshold=0` is committed (`crazyflie_mpc_env.py`) and does help the
> simple-scene case, but it is not, by itself, "the fix".

## Where the hang actually points

With the SoC acking normally, a synchronizer pinned at ~113% CPU is spinning in one of
its own unbounded busy-waits (`gym_synchronizer.py`): `check_token_exhaustion()` /
`get_firesim_cycles()` (waiting on a SoC ack) — or it is blocked inside Isaac's
`env.step()` (physics/render). Since the SoC ack path is healthy, the remaining suspect
is the **Isaac/PhysX side** (consistent with the still-hover + loaded-host correlation).
The stall watchdog (below) will confirm the exact frame the next time it fires.

## Stall watchdog (added)

`gym_synchronizer.py` starts a daemon thread (`_start_stall_watchdog`) that watches
`self.count`. If a step stops advancing for `ROSE_SYNC_WATCHDOG_S` seconds (default 45,
`0` disables), it prints the frozen step and calls
`faulthandler.dump_traceback(all_threads=True)`. The stuck frame pins the cause:

- a frame in `env.step` / isaac / physx  → **Isaac/PhysX side** (SoC idle & fine);
- a frame in `check_token_exhaustion` / `get_firesim_cycles` → **waiting on a SoC ack**.

Diagnostic only — it never alters control flow, so healthy runs are unaffected (verified:
a clean run stepped normally with zero watchdog firings).

## TACIT / L-Trace — validated SoC tracing

Flow (documented in `zephyr-chipyard-sw/samples/tacit/TACIT_TRACING.md`, `origin/dev`):

    build (spike_riscv64, CONFIG_STARTUP_TACIT=y)
      -> spike --trace=l          # l_trace spike, emits tacit.out/.log/.debug
      -> ltrace-decoder --to-txt  # -> control-flow trace.txt

Verified end-to-end here (hello-world sample): boots, `Hello World!`, exits clean,
**541,930 instructions traced** (`tacit.out` 258 KB) → decoder emits **793,201** lines /
251,269 packets, decoded PCs matching the encoder ground-truth `tacit.debug` (reset
vector `0x800001fe` → … → `End`). Toolchain: l_trace spike at
`/scratch2/dima/chipyard-fsim/toolchains/riscv-tools/riscv-isa-sim/build/spike`,
decoder at `/scratch2/dima/chipyard-fsim/software/tacit_decoder` (`misc_decoders`).

### Instruction-level SoC tracing *inside* the lockstep (both wired)

`spike --trace=l` runs standalone (no RoSE bridge), so it can't drive a bridge-dependent
control loop. Two in-lockstep tracers are now built into the harness:

1. **Spike commit-log** — `rose_spike_sim` (the normal harness). Set
   `ROSE_SPIKE_COMMITLOG=<path>` (+ `ROSE_SPIKE_COMMITLOG_START/END` step window) to write
   Spike's per-instruction commit log (PC, insn, reg/mem writes) for the boot hart during
   the live co-sim. Each step is ~5M instructions (~350 MB), so use a 1-step window near
   the point of interest. Verified: on a clean hover the window shows the SoC busy-polling
   the RoSE STATUS reg (`0x2000`) — the reqrsp sensor-wait spin (SoC executing, not hung).

2. **TACIT / L-Trace** — `rose_spike_trace` (`build.sh trace`), the same harness linked
   against the l_trace spike, which auto-registers the `0x3000000` encoder MMIO. Run a
   `CONFIG_STARTUP_TACIT` guest under it (sync supplies grants) and it emits
   `tacit.out`/`.debug` in the CWD, decodable by `ltrace-decoder`. Verified end-to-end in
   the lockstep (hello-world guest: 541,930 instrs → 793,201-line decoded trace).
   `ROSE_SPIKE_MAX_STEPS=N` cleanly bounds either capture (flushes on stop).

Tracing the *flight controller* this way needs it rebuilt with `CONFIG_STARTUP_TACIT=y`
against a zephyr tree that has the TACIT SoC driver (present in the -fresh/`dev` tree, not
the xpu-rt sample tree) — the harness side is done.

## Reproduction / evidence commands

- Device-level SoC trace in the lockstep: run the spike side with `ROSE_SPIKE_DEBUG=1`
  (`run_spike_rose_lockstep.sh`) → `grant/step/ack` log.
- Clean-hover repro (did not hang, 685 steps): `SensorEnv-v0` + `rose_flight_controller`.
- Walls repro (did not hang, 605 steps): `MultiSensorEnv-v0` `ROSE_MAZE=hallway`.
- Watchdog: default-on; lower `ROSE_SYNC_WATCHDOG_S` to tighten, set `0` to disable.
