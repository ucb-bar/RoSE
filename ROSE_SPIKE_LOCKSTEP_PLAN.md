# RoSE Spike lockstep — moving synchronization into the step loop

Design plan for replacing the current passive `--extlib` device with a mechanism
that gives a **true two-sided cycle barrier**, works on **multicore** targets, and
is **independent of wall-clock speed**. Companion to `ROSE_SPIKE_BRIDGE_PLAN.md`
(which covers the functional bridge that already passes reqrsp/dma/protocol/selftest).

## STATUS: IMPLEMENTED (2026-07-01) — Option B, all samples PASS

Chose **Option B** (control-inverted harness + minimal patch). Delivered:
- `riscv/sim.h`: `step()` moved to `protected` (2-line patch, captured in
  `soc/src/main/cc/rose_spike/rose_spike_sim_stepaccess.patch` for `setup.sh`).
- `soc/src/main/cc/rose_spike/rose_spike_sim.cc`: `rose_bridge_ctrl` (protocol +
  regmap + DMA/IRQ), thin `rose_mmio_device_t`, and `rose_sim_t : sim_t` overriding
  `idle()` to run the §3 loop; standalone `main()` builds cfg/mems and runs.
- `soc/src/main/cc/rose_spike/build.sh` builds both flavors (`plugin`, `harness`);
  harness statically links `libriscv/libfesvr/libsoftfloat/libdisasm/libfdt` →
  `soc/sim/rose_spike_sim`. Run via `soc/sim/run_spike_rose_lockstep.sh [elf] [nprocs]`.

**Results** (live synchronizer, PatternEnv):
- `reqrsp`, `dma` (interrupt-driven, through WFI), `protocol`, `selftest` → all PASS.
- **Deterministic budget, zero overshoot:** `mtime` steps exactly 0→10000→20000→…
  (10000 ticks = `step_size`/`INSNS_PER_RTC_TICK` = 1e6/100) per grant. Contrast the
  `--extlib` model's wall-clock-dependent ~9% overshoot.
- **Multicore:** `-p 2` round-robins both harts; budget stays exactly 10000/step and
  `reqrsp` PASSes — the case the per-hart `mcycle` scheme mismeasured. Gating on the
  global CLINT `mtime` (`get_core(0)->get_state()->time->read()`) is hart-count-invariant.
- **Wall-clock independence:** `idle()` returns without stepping when no grant is
  pending, so Spike cannot advance past the granted budget regardless of physics
  wall-clock speed.

The `--extlib` plugin (`librose_spike.so`) is retained as the fast single-core
functional tier. Remaining bridge-level gaps (multi-DMA channels, DMA bounds, `CS_RESET`
now handled in the harness controller, bandwidth) tracked in `ROSE_SPIKE_BRIDGE_PLAN.md`.

---

## 1. Why the current `--extlib` model is not real lockstep

The plugin (`rose_spike_device.cc`) acks `CS_GRANT_TOKEN` only after hart 0's
`mcycle` advances by `step_size`. Three structural problems:

1. **Single-hart cycle count.** `mcycle` is *per hart* (bumped individually in
   `execute.cc`); there is no global cycle. Spike round-robins harts
   (`sim_t::step`: `INTERLEAVE` instrs/hart, one device `tick()` per full round).
   Gating on `get_core(0)->mcycle` mismeasures global time as soon as there is >1
   hart or any hart sits in WFI. The only *shared* timebase is the CLINT's
   `mtime`.
2. **No barrier on the Spike side.** Only the synchronizer hard-blocks
   (`check_token_exhaustion` spin). Spike itself is never stopped by the bridge —
   it free-runs and the device merely *observes* `mcycle`. Between our ack and the
   synchronizer actually serving the response, Spike keeps executing.
3. **Wall-clock dependence.** Correctness today relies on the *guest driver*
   spinning on `STATUS` / WFI (a software barrier) to not out-run the world. If
   physics is slower than Spike in wall-clock and the guest does not poll, Spike
   runs ahead. There is no simulator-level guarantee.

A passive `abstract_device_t` (called *from* the loop via `load`/`store`/`tick`)
cannot own pacing. The fix must live where the stepping happens.

## 2. Spike embedding surface (verified)

- `sim_t::run()` (public) → `htif_t::run()` (public) — the outer loop. It
  **repeatedly calls the virtual `idle()`** and, between calls, services HTIF
  tohost/fromhost (console, syscall proxy, program exit).
- `htif_t::idle()` is `virtual void idle() {}` (htif.h:69); `sim_t::idle()`
  overrides it (sim.h:153) and calls the **private** `sim_t::step(INTERLEAVE)`.
- `sim_t::step` and `sim_t::idle` are **private**; a subclass cannot call them
  explicitly.
- `sim_t::get_core(i)` is **public**; `processor_t::step(n)` is **public**.
  `sim->addr_to_mem()` and `get_intctrl()->set_interrupt_level()` are already used.
- CLINT `mtime` is the shared timebase; `clint_t::tick(rtc_ticks)` bumps it and
  drives `MTIP`. The clint is one of the internal `devices` (private vector).
- **Precedent:** `testchipip/.../cospike_impl.cc` embeds `sim_t` directly, builds
  its own `cfg_t`/mems/clint, adds them via `sim->add_device`, and drives stepping
  with `sim->get_core(hartid)->step(1)` — **no Spike source modification**. It does
  not call `sim->run()`; it is driven externally.
- Prebuilt libs to link a custom binary: `build/libriscv.a`, `libfesvr.a`,
  `libsoftfloat.a`, `libdisasm.a`, `libfdt.a` (+ `libspike_main.a`).

## 3. The lockstep loop (target semantics)

```
per physics step (one CS_GRANT_TOKEN):
  1. BLOCK until a grant arrives            # Spike does nothing until physics grants
  2. advance the WHOLE machine by step_size # all harts, deterministic, no wall-clock race
  3. drain guest TX (requests issued this step)
  4. serve responses + deliver DMA/IRQ
  5. send CS_RSP_STALL (ack)                # physics unblocks, computes next step
```

Properties: bidirectional hard barrier (neither side free-runs); deterministic
budget (no ~9% overshoot); multicore-correct (paces the *machine*, gated on the
global `mtime`, not one hart); wall-clock-independent.

## 4. Architecture options

### Option A — Blocking `tick()` inside the existing `--extlib` plugin  *(cheapest)*
Make the device's `tick()` a hard barrier: after each round it decrements a global
budget (rounds × `INTERLEAVE`, or `mtime` delta); when the budget is spent it sends
the ack and **busy-waits inside `tick()`** (pumping the socket) until the next
grant. Because `tick()` is called from `sim_t::step`, blocking it freezes the whole
loop — all harts, `mtime`, everything.
- **Pros:** no harness, no patch; fixes all three problems; multicore-correct if
  gated on `mtime`/round-count; stays a drop-in `--extlib` library.
- **Cons:** busy-waits a host thread inside a device callback (same style as the
  synchronizer's own spin); barrier granularity = one round-robin round
  (`INTERLEAVE`, ~5000 instrs) rather than an exact instruction; HTIF console is
  deferred while blocked (harmless — the guest is frozen).

### Option B — Control-inverted harness: subclass `sim_t`, override `idle()`  *(recommended)*
A new binary `rose_spike_sim` links the spike libs, builds the sim exactly as
`spike.cc` does (reusing `--extlib`/`--device` parsing so the RoSE device still
registers), then runs `sim->run()`. `class rose_sim_t : public sim_t` **overrides
`idle()`** to implement §3: each `idle()` invocation advances one budget-sized
chunk and, at step boundaries, does the RoSE exchange + ack + grant-wait.
- Reuse of `sim_t::step` (which ticks the clint/`mtime` and round-robins harts) is
  the clean path, but `step()` is **private** → needs a **1–2 line Spike patch**
  (make `step()` and/or the device list `protected`), applied via `setup.sh` like
  RoSE's other tree injections.
- **Pros:** owns the loop the way the user asked; deterministic; reuses all of
  `sim_t` (mmu, dtb, mtime, syscall proxy, console, program load); lockstep logic
  is small and localized in the override.
- **Cons:** requires a minimal Spike source patch; new build target.

### Option C — Standalone cospike-style harness, zero patch  *(most code)*
Like Option B but do **not** subclass/reuse the private `step()`. Instead
replicate the setup (own `cfg`/mems/**own clint**/devices, à la `cospike_impl.cc`)
and drive `get_core(i)->step(n)` yourself, manually ticking the clint and devices
per round, and reproducing HTIF program-load/console/exit.
- **Pros:** zero Spike diff.
- **Cons:** ~200–400 lines duplicating spike setup + HTIF glue; must keep the clint
  and mtime cadence correct by hand; highest risk of subtle divergence from stock.

## 5. Recommended mechanism (Option B, detailed)

### Ownership move
The socket client + protocol handshake + budget accounting move **out of the
device** and **into `rose_sim_t`** (the controller that owns the loop). The RoSE
MMIO surface stays as a thin `abstract_device_t` that just forwards:
- `store(TX_DATA)` → append to controller's TX packet assembler.
- `load(STATUS/RX/DMA_CURR/INT_PEND)` → read controller state (RX FIFOs, dma_done).
- `store(INT_PEND W1C)` → controller clears pending + deasserts IRQ.
The controller keeps `rose_sync_client`, routes, RX FIFOs, DMA/IRQ delivery
(unchanged logic from today), and the new budget state.

### `idle()` override (single- and multi-core)
```cpp
void rose_sim_t::idle() override {
  if (done()) return;
  if (!ctrl.grant_active) {
    // Step 1: hard-block for the next grant (pump socket; poll, don't spin hot).
    if (!ctrl.wait_for_grant())   // returns false only on shutdown
      { htif_stop(); return; }
    ctrl.budget_target = mtime_now() + cycles_to_mtime(step_size);
  }
  // Step 2: advance one quantum toward the budget. Reuse sim_t::step so the
  // clint/mtime and all harts advance correctly (needs the protected-step patch).
  step(INTERLEAVE);                       // round-robins every hart; ticks devices
  if (mtime_now() >= ctrl.budget_target) {
    ctrl.drain_tx_and_serve();            // Steps 3-4: requests -> responses/DMA/IRQ
    ctrl.send_ack();                      // Step 5: CS_RSP_STALL
    ctrl.grant_active = false;
  }
}
```
- **Budget is gated on `mtime`** (global, shared by all harts) → multicore-correct.
  `cycles_to_mtime()` converts the `step_size` cycle budget using the CLINT freq so
  one physics period maps to a fixed `mtime` delta regardless of hart count/IPC.
- **Deterministic:** the number of instructions per step depends only on `mtime`
  cadence (fixed by `INTERLEAVE`/`INSNS_PER_RTC_TICK`), never on wall-clock.
- **Data ordering:** responses/DMA are delivered in `drain_tx_and_serve` *before*
  the ack releases physics and *before* the next step runs, so the guest observes
  fresh sensor data at the start of the next step.

### Grant-wait without a hot spin
`wait_for_grant()` polls the socket with a short blocking `recv` timeout (or
`poll(2)`), delivering any served data that arrives while waiting, and returns when
`CS_GRANT_TOKEN` is seen. This is a genuine block (no forward progress), not a
busy-loop.

## 6. Multicore specifics
- `step(N)` advances all harts through the round-robin; gating on `mtime` means
  "the machine advanced one physics period," independent of per-hart IPC or WFI.
- IRQ delivery already targets the PLIC (`set_interrupt_level`), which fans out to
  the correct hart context — unchanged.
- DMA writes via `addr_to_mem` are visible to all harts (shared memory) — unchanged.
- No per-hart cycle bookkeeping anywhere; `mcycle` is no longer used for pacing.

## 7. Packaging / build
- New source: `soc/src/main/cc/rose_spike/rose_spike_sim.cc` (harness `main` +
  `rose_sim_t`), reusing `rose_sync_client.{h,cc}` and the existing regmap/DMA/IRQ
  logic (refactored into a controller class shared with, or replacing, the device).
- New build target producing `soc/sim/rose_spike_sim`, linking `libriscv.a
  libfesvr.a libsoftfloat.a libdisasm.a libfdt.a` (mirror `spike.cc`'s link line;
  reuse `/tmp/build_rose_spike.sh` flags: `-std=c++2a -D_GNU_SOURCE -include
  sys/syscall.h`).
- Spike patch (Option B): `sim.h` `private:`→`protected:` boundary so `step()` (and
  the device list, if needed) is reachable. Apply from `setup.sh` as a guarded
  patch against the pinned spike commit, alongside RoSE's existing injections.
- `run_spike_rose.sh` gains a mode to launch `rose_spike_sim` instead of stock
  `spike --extlib=...`. Keep the `--extlib` path for the fast functional tier.

## 8. Migration & validation
1. Build `rose_spike_sim`; boot Zephyr, confirm HTIF console/exit still work.
2. Re-run `reqrsp` / `dma` / `protocol` / `selftest` — expect identical PASSes.
3. **Lockstep assertions:** log `(mtime, physics_step)` each boundary; verify a
   fixed `mtime`-delta per grant (no overshoot) and that a compute-only guest (no
   RoSE I/O) still advances physics in proportion to cycles.
4. **Multicore:** build a 2-hart Zephyr (or bare tests) config; verify the barrier
   holds with both harts busy and with one hart in WFI.
5. **Wall-clock independence:** artificially slow the physics env; confirm Spike
   blocks (no drift), unlike the `--extlib` model.
6. Keep the `--extlib` library as the single-core functional tier; document
   `rose_spike_sim` as the lockstep tier.

## 9. Open decisions (need input)
- **Spike patch vs zero-diff:** accept the minimal `sim.h` patch (Option B, clean +
  small) or keep Spike pristine and pay the extra harness code (Option C)?
- **Or take Option A first?** Blocking-`tick()` gets multicore + two-sided barrier
  today with almost no code and no patch, at the cost of an `INTERLEAVE`-granularity
  busy-wait. Could ship A now and pursue B for precision/determinism later.
- **Budget unit:** gate on `mtime` (recommended, global) vs summed per-hart cycles.
