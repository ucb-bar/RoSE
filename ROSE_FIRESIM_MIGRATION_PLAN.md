# RoSÉ FireSim/Chipyard Migration Plan

**Investigation date:** 2026-06-10
**Decision (locked):** migrate to the **latest chisel6 stack** — Chipyard **1.13.0**
(Chisel **6.5.0**) with nested FireSim, **chipyard-as-top**, using the **firechip
bridge-stub** architecture. Build on the work already started in the `chipyard-top`
branch. (We take the chisel3→chisel6 jump now rather than parking on an intermediate,
since that hurdle is unavoidable and the stub refactor is already drafted.)

---

## 0. TL;DR

- `fsim-17` (FireSim 1.17) is a dead, never-built stub — **abandon it.**
- The chisel-version *discrepancy* was a **1.17-only** problem (FireSim 3.5.6 vs
  Chipyard 3.6). It does **not** exist at 1.18.0 (both 3.6.0) — which is why `main`
  builds. But we are **not** stopping at 1.18/1.20; we go to the chisel6 latest.
- `chipyard-top` already **started** the right refactor: the **firechip bridge-stub
  split** (`bridgeinterfaces` / `bridgestubs` / `goldengateimplementations`) +
  chipyard-as-top topology. It just never had a consistent submodule pin, so it
  doesn't build as checked out.
- **Target stack:** Chipyard **1.13.0** (`69eba860`, Chisel 6.5.0, Scala 2.13.12) as
  the top-level submodule, FireSim `141bff73` (= `1.20.1-992`, chisel6) nested at
  `sims/firesim`. This tagged release contains the exact firechip-stub layout the
  `chipyard-top` code targets.

---

## 1. How RoSÉ Plugs Into FireSim/Chipyard

RoSÉ keeps its hardware/driver sources in `soc/src/main/{scala,cc}` and **symlinks
them into the Chipyard/FireSim trees** via `soc/setup.sh`, then builds normally. The
injected pieces:

- a **target peripheral** ("Rose adapter": TX/RX queues, a configurable packet
  **arbiter/router** with a routing table, and a **camera DMA engine** that writes
  frames to memory over TileLink);
- a **FireSim bridge** (target stub + host-side Golden Gate module + C++ driver) that
  tunnels packets to an external TCP **synchronizer** (`deploy/hephaestus/`);
- **config/binder glue** wiring it into `DigitalTop`, IOBinders, and HarnessBinders.

---

## 2. Branch / Version Landscape (findings)

| Branch | FireSim | Chisel | Topology | Bridge code shape | State |
|---|---|---|---|---|---|
| **`fsim-17`** (current) | **1.17.0** | firesim 3.5.6 vs chipyard 3.6 — **mismatch** | firesim-as-top | monolithic `AirSim`/`Rose` in `firesim.bridges` | **Dead stub** ("build not fixed", Jul 2023). |
| **`main`** | **1.18.0** | **3.6.0 unified** | firesim-as-top | refactored monolithic `RoSEBridge` in `firesim.bridges` | **Known-good** (ASPLOS'25 CoSMo artifact). |
| **`chipyard-top`** | 1.18.0 pin **(stale)** — code targets **chisel6** | 6.x (intended) | **chipyard-as-top** (intended) | **firechip bridge-stub split** | **WIP, inconsistent — won't build as-is.** Basis for this migration. |
| **TARGET → Chipyard 1.13.0** | `141bff73` (1.20.1-992, chisel6) | **6.5.0** | chipyard-as-top | firechip bridge-stub | the stack we are migrating to. |

### Verified Chisel/Scala discrepancy per release (firesim `sim/build.sbt` vs chipyard `build.sbt`)
| FireSim/stack | FireSim side | Chipyard side | Match? |
|---|---|---|---|
| 1.17.0 | Chisel 3.5.6 | Chisel 3.6 | ❌ original break |
| 1.18.0 | 3.6.0 / Scala 2.13.10 | 3.6.0 / 2.13.10 | ✅ no discrepancy |
| 1.20.1 | 3.6.1 / 2.13.10 | 3.6.1 / 2.13.10 | ✅ matched (chisel3) |
| **Chipyard 1.13.0 (target)** | chisel6 | **6.5.0 / 2.13.12** | ✅ matched (chisel6) |

---

## 3. Target Stack Topology (chisel6 / chipyard-as-top)

```
soc/sim/chipyard/                      <- NEW top-level submodule @ chipyard 1.13.0 (Chisel 6.5.0)
  env.sh                               <- sourced by rose-setup.sh
  sims/firesim/                        <- FireSim nested here (141bff73), sourceme-manager.sh
  generators/firechip/
    bridgeinterfaces/  src/main/scala/   <- RoSEBridgePort.scala  (shared port/params)
    bridgestubs/       src/main/scala/rose/  <- RoSEBridgeStub.scala (target BlackBox)
    bridgestubs/       src/main/cc/bridges/   <- rose.{cc,h}        (C++ driver lives HERE now)
    goldengateimplementations/ src/main/scala/ <- RoSEBridgeModule.scala (host Golden Gate)
    chip/              src/main/scala/   <- BridgeBinders.scala, RoSEFireSimConfigs.scala
  generators/chipyard/src/main/scala/
    DigitalTop.scala, config/RoSEConfigs.scala, iobinders/{IOBinders,Ports}.scala
  generators/rose/   src/main/scala/   <- RoSEAdapter, RoSEIO, RoSEDMA, Dataflow, RoSEGeneratorConfig
```

This matches `chipyard-top`'s `rose-setup.sh` (`CHIPYARD_DIR=soc/sim/chipyard`,
`FIRESIM_DIR=${CHIPYARD_DIR}/sims/firesim`) and its `setup.sh` destinations. Two
deltas to fix in those scripts: (a) the C++ driver is still routed to
`firesim-lib/.../cc/bridges/` — in this stack it belongs under
`firechip/bridgestubs/src/main/cc/bridges/`; (b) the legacy
`firesim.bridges` `RoSEBridge.scala` is superseded by the three stub files and should
be dropped.

---

## 4. What `chipyard-top` Already Did (reuse this)

The big commit `7079073` (Feb 2025) drafted the firechip-stub refactor:
- `RoSEBridgePort.scala` → `package firechip.bridgeinterfaces` — `RosePortIO`,
  `RoseBridgeTargetIO`, `RoseKey`, `RoseAdapterParams`, dataflow/dst-port configs
  (all `HasSerializationHints` so params serialize across the target/host boundary).
- `RoSEBridgeStub.scala` → `package firechip.bridgestubs` — target-side `RoseBridge`
  BlackBox; references the host impl by **string**
  `moduleName = "firechip.goldengateimplementations.RoSEBridgeModule"` (the
  decoupling that removes any need for a shared compiled bundle type).
- `RoSEBridgeModule.scala` → `package firechip.goldengateimplementations` — host
  Golden Gate module (arbiter table, bandwidth writer, rx controller, cycle budget).
- `rose-setup.sh` rewritten for chipyard-as-top; `setup.sh` routes files into the
  firechip subprojects.

What it lacks: a consistent submodule pin (still FireSim 1.18.0, no `soc/sim/chipyard`
submodule), a finished chisel6 port, and the C++ driver moved/renamed into the new
layout.

---

## 5. Plan

### Phase 1 — Make the stack consistent and get a STOCK chisel6 build
*Purpose: a green stock build proves the toolchain before adding RoSÉ.*
1. Add **`soc/sim/chipyard`** as a top-level submodule pinned to **chipyard 1.13.0**
   (`69eba860`); init recursively so FireSim lands at `soc/sim/chipyard/sims/firesim`
   (`141bff73`). Update `.gitmodules`; retire the old top-level `soc/sim/firesim`
   entry (firesim-as-top).
2. Install per chipyard-as-top: `./build-setup.sh` in chipyard, `source env.sh`,
   then `source sims/firesim/sourceme-manager.sh --skip-ssh-setup`. `rose-setup.sh`
   already encodes this — verify its paths.
3. Build a **stock** metasim (e.g. a default firechip `RocketConfig`) with **no RoSÉ
   peripheral** to confirm the chisel6 chipyard+firesim toolchain works end-to-end.
   Capture exact commands as the environment baseline.

### Phase 2 — Port RoSÉ Scala to Chisel 6 + finish firechip-stub wiring
*Purpose: the bulk of the effort — the chisel3→chisel6 API port.*
1. **Mechanical chisel6 port** across all RoSÉ Scala (`RoSEAdapter`, `RoSEDMA`,
   `RoSEIO`, `Ports`, `Dataflow`, `RoSEConfigs`, `RoSEGeneratorConfig`, `IOBinders`,
   `BridgeBinders`, `DigitalTop`, and the three stub files). Expect:
   - import/namespace changes (chisel6 drops `Chisel._` compat; some
     `chisel3.experimental.*` moved); keep `org.chipsalliance.cde.config`.
   - `RegEnable(next=…, enable=…)` named args → **positional** (chisel6 removed them).
   - rocket-chip/testchipip churn: `TLHelper.makeClientNode` is gone — use
     `TLClientNode`/`TLMasterParameters`; recheck `edge.Put`, diplomacy `coupleTo`.
   - `Vec`, `Decoupled`, `Queue`, `Counter`, `Enum`, `switch` are largely unchanged.
2. **Reconcile the stub files against chipyard 1.13.0's firechip conventions.** Use a
   reference bridge present in 1.13.0 firechip (e.g. **UART**:
   `bridgeinterfaces/UART.scala`, `bridgestubs/.../UART.scala`,
   `goldengateimplementations/.../UART.scala`) as the authoritative template for
   package names, the `moduleName` string, annotation/serialization, and
   `genHeader`/`genConstructor`.
3. **Verify `setup.sh` destinations** against the real 1.13.0 firechip paths; ensure
   `BridgeBinders.scala` + `RoSEFireSimConfigs.scala` land in `firechip/chip/...`.
4. **Drop** the legacy `firesim.bridges` `RoSEBridge.scala` (superseded by the stub
   trio).

### Phase 3 — Port the C++ driver into the new layout
1. Move/rename `airsim.{cc,h}` → **`rose.{cc,h}`** into
   `firechip/bridgestubs/src/main/cc/bridges/` (the chisel6 stack's driver home), and
   update `setup.sh` (it currently still points at `firesim-lib/.../cc/bridges/`).
2. Port to the current `bridge_driver_t` API: class `final`, `static char KIND;`,
   ctor `(simif_t&, const ROSEBRIDGEMODULE_struct&, int, const std::vector<std::string>&)`,
   value (not pointer) MMIO access, auto-registration via the registry. The driver
   name must match the bridge's `genConstructor("rose_t", "rose")`.
3. Confirm MMIO struct field names/order match the `genROReg`/`genWOReg` names in
   `RoSEBridgeModule.scala`.

### Phase 4 — Build, iterate, validate (Verilator metasim first)
1. Elaborate + build a **Verilator metasimulation** of a Rose config — this is the
   initial test vehicle (no FPGA/bitstream needed; fastest correctness loop). Iterate
   on chisel6 elaboration + Golden Gate errors here.
2. Confirm the generated header registers `rose_t` and the bridge ticks; bring up the
   `deploy/hephaestus/` synchronizer against the Verilator metasim and validate an
   end-to-end packet round-trip.
3. **Only after** metasim is green: build a bitstream (`xilinx_alveo_u250`) →
   on-FPGA end-to-end run.

### Execution order (fastest to green)
Phase 1 (stock chisel6 build) → Phase 2 (chisel6 port + stub wiring) → Phase 3 (driver)
→ Phase 4 (metasim → FPGA).

---

## 6. Risks / Open Questions
- **chisel3 → chisel6 port is the main cost.** Budget time for rocket-chip/testchipip
  API drift, especially the **DMA TileLink client** (`RoSEDMA`/`CamDMAEngine`:
  `TLHelper.makeClientNode` removed) and diplomacy attach changes.
- **Pin chipyard 1.13.0 (tag) vs chipyard `main`.** Recommend the **1.13.0 tag** for
  reproducibility; `main` (Chisel 6.7.0) moves and risks churn. Confirm RoSÉ-needed
  generators (gemmini, nvdla, etc.) build at 1.13.0.
- **firechip stub conventions** may differ subtly between the chipyard version
  `chipyard-top` was drafted against and 1.13.0 — validate the `moduleName` string,
  package paths, and `HasSerializationHints` usage against 1.13.0's own bridges.
- **C++ driver home** moved to `firechip/bridgestubs/src/main/cc/bridges/`; confirm the
  build globs that path in this FireSim revision.
- **Synchronizer protocol** opcodes are hard-coded in the driver — confirm they match
  the current `deploy/hephaestus/` synchronizer.
- **Config YAML schemas** (`config_runtime`/`config_build_recipes`/`config_hwdb`)
  differ at this FireSim revision; the symlinked `soc/sim/config/*_local.yaml` need a
  schema pass.
- **Promote to a committed generator?** Long-term, making `rose` + the firechip stubs
  first-class committed Chipyard generators (vs symlink injection) would survive future
  bumps better.

---

## 7. Immediate Next Actions
1. Add `soc/sim/chipyard` @ **1.13.0** as a submodule (recursive → FireSim
   `141bff73` at `sims/firesim`); fix `.gitmodules`; retire old `soc/sim/firesim`.
2. Stock chisel6 metasim build (no RoSÉ) to validate the toolchain.
3. Open chipyard 1.13.0's firechip UART bridge as the reference template, then begin
   the chisel6 port of the RoSÉ stub trio + adapter.

> Housekeeping: this investigation left the `soc/sim/firesim` submodule checked out at
> FireSim **1.17.0** (`07efa6a`) and fetched several chipyard/firesim commits into that
> clone. Reset/replace it when setting up the chipyard-as-top topology in Phase 1.
</content>
