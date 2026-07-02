# RoSE bridge on Spike — Design Plan

Add a RoSE co-simulation bridge as a **dynamically-loaded Spike device plugin**, so the same
Zephyr apps (`samples/rose`) and the same Python synchronizer / gym envs drive a *functional*
Spike run — a fast inner-loop co-sim tier alongside FireSim metasim and FPGA.

Status: **prototype loads + boots + connects; stalls at the budget handshake (next step).**
Companion: `ZEPHYR_RoSE_PLAN.md`, `INSTALL_NOTES.md`.

## Prototype status (2026-07-01)

Files (in the RoSE repo, reused by FireSim later):
- `soc/src/main/cc/rose_sync_client.{h,cc}` — standalone synchronizer-protocol client
  (socket + wire framing + `CS_*`), with **lazy non-blocking connect** (`ensure_connected()`)
  so a Spike device can't stall setup before the synchronizer is up.
- `soc/src/main/cc/rose_spike/rose_spike_device.cc` — `rose_bridge_t : abstract_device_t`
  (regmap load/store, TX packetization, RX-FIFO + DMA delivery, `simif::addr_to_mem` DMA,
  `intctrl->set_interrupt_level` IRQ, `REGISTER_DEVICE` + `parse_from_fdt`).
- Build: `/tmp/build_rose_spike.sh` -> `soc/sim/librose_spike.so` (g++ `-std=c++2a -fPIC -shared`
  vs the `riscv-isa-sim` headers; needs `-D_GNU_SOURCE -include sys/syscall.h` to dodge a
  conda-gcc-13 `SYS_futex`/atomic_wait quirk; `addr_to_mem` is accessed via the `simif_t` base
  since it's private on `sim_t`).
- Run harness: `soc/sim/run_spike_rose.sh [elf]`.

**Achieved:**
- `.so` compiles and **dlopens cleanly** into Spike (ABI-compatible); `--device=rose_bridge`
  is **registered** (a bogus name errors "not found in loaded extlibs"; `rose_bridge` doesn't).
- End-to-end bounded run (`samples/rose/reqrsp` + live synchronizer): the device
  **connects** (`[rose_sync] connected to localhost:10001`), Zephyr **boots on Spike**
  (`*** Booting Zephyr OS ... ***`), and the synchronizer reports `bridge connected — starting
  synchronizer loop`. The RoseAdapter at `0x2000` did NOT collide with Spike's memory map
  (no access exception) — resolving open question §8.

**COMPLETE — all four `samples/rose` apps PASS on Spike** against the live synchronizer:

| sample     | result on Spike                                             |
|------------|------------------------------------------------------------|
| `reqrsp`   | `ROSE reqrsp: PASS (16 words, [0]=0xc0de0000 [15]=0xc0de000f)` |
| `dma`      | `ROSE dma: PASS (16 words, ...)` (interrupt-driven)        |
| `protocol` | `ROSE protocol: PASS (16 words, ...)`                       |
| `selftest` | `ROSE selftest: dma=PASS reqrsp=PASS => PASS`               |

Two bugs were fixed to get here (both in `rose_spike_device.cc`):

1. **Token handshake stall** (reqrsp): the synchronizer's `check_token_exhaustion()` blocks after
   each `CS_GRANT_TOKEN` waiting for a control packet (`cmd > 0x80`) in its `sync_rxqueue`. The
   prototype `pump()` ignored `CS_GRANT_TOKEN`, so the loop never advanced and no data was served.
   Fixed: `pump()` acks each grant with `CS_RSP_STALL` (functional model — the granted step is
   consumed immediately). This unblocked the reqrsp serve -> first `PASS`.

2. **Missing `INT_PEND` register** (dma): the DMA/IRQ host path worked (data written to
   `0x88000000` via `addr_to_mem`, `set_interrupt_level(3,1)` delivered the PLIC IRQ, ISR fired),
   but `load()` had **no case for `int_pend_off()` (0x20)** so the ISR read `pend=0`, gave no
   semaphore, and cleared nothing (interrupt storm; `rose_dma_wait(K_FOREVER)` hung). Fixed:
   `load()` returns the DMA-done pending bitmask at `int_pend_off()`; the ISR's W1C then clears it
   and `set_irq(false)` deasserts. This unblocked the interrupt-driven DMA path -> `dma PASS`.

Verified along the way: the RoseAdapter at `0x2000` doesn't collide with Spike's memory map
(no access exception, resolving open question §8); Spike's PLIC (`PLIC_BASE 0x0c000000`) matches
Zephyr's `spike_riscv64` DT plic node, and `--extlib` devices are ticked (`sim.cc` `for (dev :
devices) dev->tick()`), so `pump()` runs during guest WFI. A `ROSE_SPIKE_DEBUG=1` env var enables
per-event tracing (route/deliver/dma-write/set_irq/int_pend) in the `.so`.

Also flagged during bring-up: `connect_blocking()` in the device ctor hung Spike setup when the
synchronizer wasn't up — fixed via the lazy-connect rework above.

**Cycle-lockstep (implemented).** Initially grants were acked immediately, which preserved
transaction ordering but left the guest's compute rate *decoupled* from physics time: the
synchronizer grants one token per physics step and blocks in `check_token_exhaustion()` until the
bridge acks, but an instant ack meant the guest ran only ~one Spike interleave quantum per step
instead of the configured `firesim_step` cycles. Fixed by gating the ack on a cycle budget:

- `CS_DEFINE_STEP` now captures `step_size` (was ignored).
- On `CS_GRANT_TOKEN`, record `grant_target = mcycle + step_size` instead of acking; `mcycle`
  (`sim->get_core(0)->get_state()->mcycle`, IPC=1 in functional Spike) is the cycle cost model.
- `pump()` (called from `tick()`/MMIO, incl. during WFI) emits `CS_RSP_STALL` only once
  `mcycle >= grant_target`. `CS_REQ_CYCLES` now reports cycles retired since the grant.

Result: **physics advances one step per `step_size` retired guest cycles** — a true functional
lockstep. Verified: `define_step size=1000000` → each grant completes exactly ~1e6 cycles later;
DMA works through WFI (guest idles the full budget, then the frame is delivered at the step
boundary → IRQ → `PASS`); `reqrsp`, `dma`, `selftest` all still PASS. Two caveats: (1) grant-to-
grant deltas overshoot ~9% because the guest free-runs during the synchronizer's Python
`env.step()` between grants — bounded, and the "≥ step_size cycles/step" guarantee holds; (2) this
is functional lockstep (correct compute/physics *rate*), **not** RTL cycle-accuracy — no
pipeline/memory-latency model. Bandwidth (`CS_CFG_BW`) is still not modeled.

**Remaining (optional, not blocking):** multi-DMA-channel support (delivery/status/int-pend
hardcode ch0), DMA armed-size bounds checking, `CS_RESET` state reset, and bandwidth modeling.
None are exercised by the current single-camera + reqrsp workloads.

---

## 1. Goal & motivation

Today RoSE co-sim runs only on FireSim (Verilator metasim, slow) or FPGA (fast, needs the
board). Spike (the RISC-V ISA sim) is already used by `zephyr-chipyard-sw` (`spike_riscv64`
board) and is a fast *functional* simulator. A RoSE bridge device for Spike gives a third tier:

    Spike (functional, seconds)  ->  FireSim metasim (Verilator, timing-ish)  ->  FPGA (cycle-accurate)

all sharing ONE synchronizer + ONE set of Zephyr apps/drivers. Spike becomes the fast
inner loop for driver / app / algorithm development against the real physics sim.

## 2. Why it fits Spike cleanly (verified API)

`riscv-isa-sim` (`riscv/abstract_device.h`) already exposes exactly what's needed:

```cpp
class abstract_device_t {
  virtual bool load(reg_t addr, size_t len, uint8_t* bytes) = 0;   // MMIO read
  virtual bool store(reg_t addr, size_t len, const uint8_t*) = 0;  // MMIO write
  virtual reg_t size() = 0;
  virtual void tick(reg_t rtc_ticks) {}                            // periodic hook
};
class device_factory_t {
  virtual abstract_device_t* parse_from_fdt(const void* fdt, const sim_t* sim,
                                            reg_t* base, const std::vector<std::string>& sargs) const;
};
#define REGISTER_DEVICE(name, parse, generate)  // -> mmio_device_map_t
```
plus `spike --extlib=<lib.so>` (dlopen `RTLD_NOW|RTLD_GLOBAL`) and `--device=<name>,<args>` —
i.e. the dynamically-loaded module mechanism the design calls for.

**One devicetree node drives both worlds.** `parse_from_fdt` matches by FDT `compatible`, so the
same `ucbbar,RoseAdapter@2000` node the Zephyr driver binds to also tells Spike where to attach
the device (reg base, interrupts, `dma-base-address`). The `samples/rose` apps, the `rose`
Zephyr driver, and the DTS overlay run **unmodified**.

## 3. Two-MMIO insight (from the FireSim driver)

The FireSim path has TWO interfaces, and the Spike device fuses them:

1. **SoC-side RoseAdapter** (`0x2000`, `rose_port.h`): STATUS / TX_DATA / RX_DATA_n /
   DMA_CFG / DMA_CURR / int-pending. This is what the Zephyr driver touches.
2. **Host-side GoldenGate bridge** (`ROSEBRIDGEMODULE_struct`: `in_bits/valid/ready`,
   `in_budget_*`, `in_bigstep_*`, `cycle_budget`, `out_bits/...`): what `rosebridge.cc` uses to
   shuttle socket<->RTL and to gate FireSim advance on the cycle budget.

In FireSim these are separate (target RTL widget + host C++ driver). **In Spike there is no
separate widget — the device IS both**: it presents the SoC-side regmap in `load`/`store` and
runs the synchronizer protocol + cycle gating internally.

## 4. Architecture — `rose_bridge_device_t` (the plugin)

A single `abstract_device_t` in a `.so`:

- **Register model** (`load`/`store`): functional port of `RoseAdapter.scala`'s regmap —
  TX FIFO (store @0x08), per-channel RX FIFOs (load @0x0c/0x10, read = dequeue), STATUS bits
  (@0x00: tx-ready, rx-valid per ch, dma-buffer), DMA config/curr (@0x14/0x18), W1C int-pending
  (@0x20). Same offsets as `rose_port.h` so the Zephyr driver is unchanged.
- **Synchronizer client** (`rose_sync_client`): TCP to `:10001` + the `CS_*` handshake
  (bandwidth/route config at init; the budget/bigstep request-response; data packets
  `[cmd, big_step, budget, num_bytes, data...]`). Ported/factored from `rosebridge.cc`.
- **DMA**: on a request routed to a DMA channel, write the served bytes into Spike DRAM at
  `dma-base-address` (0x88000000) via the `sim_t*`; then raise the DMA-complete interrupt.
- **Interrupt**: assert the configured PLIC source (3) so the Zephyr ISR fires.
- **Cycle-sync / stall** (see §5).

Factory: `REGISTER_DEVICE(rose_bridge, ...)` with `parse_from_fdt` matching
`compatible = "ucbbar,RoseAdapter"`, reading `reg`/`interrupts`/`dma-base-address` from the FDT.

## 5. Cycle-sync / stalling — mechanism + fidelity caveat

**Mechanism (equivalent *semantics* to FireSim's budget gating):**
- **Blocking `load()`**: a SoC read of STATUS/RX_DATA/DMA_BUFFER that has no data yet *blocks the
  hart* while the device pumps the synchronizer (send pending request, receive served data,
  exchange budget) until data is ready. Single-threaded Spike blocking in the handler IS the
  stall — no separate token bridge needed.
- **`tick(rtc_ticks)`** drives the budget/bigstep exchange between accesses: report Spike's cycle
  count, receive a budget, allow the synchronizer to step the gym env per budget.
- Spike's `mcycle`/`minstret` provides the "cycle" for the budget protocol.

**Caveat — functional, not cycle-accurate.** Spike models ~1 cycle/instruction with no
pipeline/memory-hierarchy timing. So the **stalling + lockstep ordering are faithful** (SoC waits
for data; physics advances per granted budget; request/response ordering preserved), but the
**cycle *counts* are approximate**. Great for functional co-sim / driver + algorithm dev (orders
of magnitude faster than FireSim); NOT a substitute for FireSim/FPGA when cycle-accurate timing
matters.

## 6. Reuse map (the big win)

| Reused UNCHANGED | New |
|---|---|
| Python synchronizer + gym envs (incl. `PyBulletDroneEnv-v0`) | `rose_bridge_device_t` plugin (`.so`) |
| Zephyr `rose` driver + `subsys/rose` + `samples/rose` | CMake to build vs `riscv-isa-sim` headers |
| DTS node `ucbbar,RoseAdapter` + register map (`rose_port.h`) | — |
| `CS_*` protocol + packet framing (from `rosebridge.cc`) -> factor into shared `rose_sync_client` used by BOTH the FireSim driver and the Spike device | — |

## 7. Work items (incremental)

1. **`rose_sync_client`** (`soc/src/main/cc/rose_sync_client.{h,cc}`): standalone,
   transport/sim-agnostic synchronizer-protocol client — socket connect, `CS_*` + packet
   framing, init config (bandwidth/route), request + serve + budget handshake. Ported from
   `rosebridge.cc` (leave the FireSim driver working; refactor it to share later).
2. **Spike device** (`soc/src/main/cc/rose_spike/rose_spike_device.cc`): `abstract_device_t`
   regmap + `REGISTER_DEVICE` + `parse_from_fdt(ucbbar,RoseAdapter)`, using `rose_sync_client`.
   Start blocking-load reqrsp (channel 2); add DMA + interrupt; then the `tick()` budget loop.
3. **Build** the `.so` (CMake) against the spike headers we already have
   (`toolchains/riscv-tools/riscv-isa-sim`).
4. **Run**: `spike --extlib=librose_spike.so --device=rose_bridge,... <zephyr.elf>` with the
   synchronizer up + a DTS carrying the rose node; validate `samples/rose/selftest` prints the
   same `ROSE selftest: ... => PASS`.

## 8. Open questions to resolve during prototyping

- **`tick()` granularity** vs the budget loop (may need to wrap `sim_t::step` / an htif callback
  for finer stepping than the coarse rtc tick).
- **Device -> PLIC interrupt injection** API in this Spike version (how ns16550 etc. raise IRQs).
- **DMA memory access** from the device (`sim->addr_to_mem` / bus load-store to 0x88000000).
- **FDT provisioning**: Spike can build its FDT from `--device`, or consume a supplied DTS — the
  `ucbbar,RoseAdapter` node must be present for both the device attach and Zephyr binding.
- **Protocol exactness**: match the synchronizer wire format (see `gym_synchronizer.py` /
  `synchronizer.py`) for budget/bigstep + data packets.

## 9. Recommended first milestone

Prove the loop end-to-end with the *simplest* device: reqrsp channel 2 only, blocking-load
stall, minimal budget handshake — run `samples/rose/reqrsp` on Spike against the live
synchronizer and get `ROSE reqrsp: PASS`. Then layer DMA + interrupt + `tick()` budgeting, and
finally the combined `selftest`.
