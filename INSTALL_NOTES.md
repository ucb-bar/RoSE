# RoSÉ Install Notes — chipyard-as-top / Chisel 6 (`chisel6-migration` branch)

Running the documented install flow on the restructured chipyard-as-top layout
(Chipyard 1.13.0 @ `soc/sim/chipyard`, FireSim 1.20.1-51 nested at
`soc/sim/chipyard/sims/firesim`). Logging every place the real flow breaks or differs
from the README so the scripts can be fixed.

Host: 48 cores, 125 GB RAM, conda 25.7.0 at `/scratch2/dima/miniforge3`. No
`GITHUB_TOKEN` in env.

---

## Documented flow (README, this branch) vs. reality

README still describes the **firesim-as-top** install:
```
git submodule update --init ./soc/sim/firesim
cd ./soc/sim/firesim
./scripts/machine-launch-script.sh --prefix [CONDA_DIR]
./build-setup.sh
source sourceme-manager.sh --skip-ssh-setup
firesim managerinit --platform xilinx_alveo_u250
```
**Differs:** we are now chipyard-as-top. The real flow is Chipyard's:
```
cd soc/sim/chipyard
./build-setup.sh                 # conda env + submodules + toolchain + scala + firesim + circt
source env.sh
cd sims/firesim && source sourceme-manager.sh --skip-ssh-setup
firesim managerinit --platform xilinx_alveo_u250
```
- conda already installed → `machine-launch-script.sh` step skipped (N/A).

---

## STEP 1 — Chipyard `build-setup.sh`  (IN PROGRESS)

Command actually run (deviation noted):
```
cd soc/sim/chipyard && ./build-setup.sh --skip-marshal
```
- **Deviation `--skip-marshal`:** skips FireMarshal buildroot (steps 8/9, ~1–3 h).
  Not needed for an initial **Verilator metasim** bring-up (that needs toolchain +
  RTL build, not a Linux target image). Run marshal later for Linux workloads.
- Steps that DO run: conda env, submodule init, RISC-V toolchain, ctags, chipyard
  scala precompile, firesim setup, firesim scala precompile, CIRCT, cleanup.
- Watch points: CIRCT download (step 10) is anonymous (no token) — may rate-limit;
  scala precompile (steps 5/7) are long/RAM-heavy.

**Result: ✅ SUCCESS** (exit 0, "Setup complete!", ~13 min wall). Steps 1–7, 10, 11
ran; 8–9 (marshal) skipped as intended. Verified artifacts:
- `soc/sim/chipyard/env.sh` created; `.conda-env/` conda env created.
- `RISCV=.../.conda-env/riscv-tools`; `riscv64-unknown-elf-gcc 13.2.0`.
- CIRCT `firtool` installed (anonymous download of 759 MB succeeded — no rate-limit).
- `Verilator 5.022` present (this is our metasim engine).
- FireSim nested sim tree populated (`sims/firesim/sim/midas` present).
- Chipyard **and** FireSim Scala both pre-compiled successfully → the **stock chisel6
  stack builds**.

Benign noise observed (not failures):
- `Connecting to 169.254.169.254:80... failed: Connection timed out` — FireSim's
  EC2-metadata probe; expected off-EC2, handled gracefully.
- Many `[warn]` lines are just Scala identifiers containing "error"
  (`l2_error`, `TLError`, `BuiltInErrorDeviceParams`) — compiler warnings, not errors.

Log: `soc/sim/chipyard/rose-build-setup.log`.

Not run (FPGA-only, deferred): `firesim managerinit --platform xilinx_alveo_u250`
— only needed for bitstream/FPGA management, not for Verilator metasim.

---

## KNOWN BREAKAGES in RoSÉ wrapper scripts (run AFTER chipyard install)

Found by static inspection — these are the next things to fix before `bash soc/setup.sh`
and `bash soc/build.sh` will work on this layout:

### `rose-setup.sh`
- Sources `${CHIPYARD_DIR}/env.sh` — OK in chipyard-as-top (created by build-setup),
  but the script also has leftover `git submodule update --init soc/sim/firesim`
  references to the old path → no-op/incorrect. Needs path cleanup.

### `soc/setup.sh`  (chipyard-as-top version, already partly updated)
1. **Missing dir:** symlink dest `generators/firechip/bridgestubs/src/main/scala/rose/`
   — `mkdir -p` is present (line 14) so this one is OK; confirm at runtime.
2. **C++ driver dest wrong:** lines 74–75 symlink `airsim.{cc,h}` →
   `${FIRESIM_DIR}/sim/firesim-lib/src/main/cc/bridges/`, which **does not exist** in
   FireSim 1.20.1. Bridge **stub drivers now live in chipyard** at
   `generators/firechip/bridgestubs/src/main/cc/bridges/` (verified: `uart.cc` is
   there). → repoint to `${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/cc/bridges/`.
3. **firemarshal path/branch:** `cd ${FIRESIM_DIR}/sw/firesim-software/ &&
   git checkout ubuntu-add` — `sw/firesim-software/` does not exist (FireMarshal moved
   to chipyard `software/firemarshal`), and branch `ubuntu-add` does not exist. → fix
   path / drop the `git checkout`.
4. **sbt patch silently no-ops:** sed targets
   `s/midas, icenet, testchipip, sifive_blocks)/.../` (line 117) and
   `.dependsOn(midas, icenet, testchipip, rocketchip_blocks)` (line 143) **do not match**
   FireSim 1.20.1 `sim/build.sbt` (actual: `.dependsOn(rocketchip, midas, firesimLib %
   "test->test;compile->compile")`). → the `rose` ProjectRef wiring into firesim won't
   apply; rework against the real build.sbt. **NOTE:** in chipyard-as-top the `rose`
   generator + firechip stubs may be wired through chipyard's build, so the firesim-side
   patch may be unnecessary — verify how firechip bridges are aggregated.
5. **chipyard aggregate patch OK:** sed for
   `gemmini, icenet, tracegen, cva6, nvdla, sodor, ibex, fft_generator,` matches
   chipyard 1.13.0 `build.sbt` (line ~159) → applies.

### `soc/build.sh` / `deploy/setup.sh`
- Both hardcode old `FIRESIM_DIR=${ROSE_DIR}/soc/sim/firesim` and
  `CHIPYARD_DIR=${FIRESIM_DIR}/target-design/chipyard` → wrong for chipyard-as-top.
  Image symlinks reference `sw/firesim-software/images/...` which no longer exists.
  **Not yet fixed** — only needed for the DNN/Linux workload build, not metasim.

---

## STEP 2 — RoSÉ setup scripts FIXED + run (✅)

Fixed the wrapper scripts for chipyard-as-top / FireSim 1.20.1, then ran them.

**`rose-setup.sh`** — replaced stale `git submodule update --init soc/sim/firesim`
(wrong path) with a guard that inits the nested `${CHIPYARD_DIR}/sims/firesim` only if
missing. Sourced cleanly (exit 0); downloads `yq`, sources `env.sh` +
`sourceme-manager.sh`, rewrites the RoSÉ `config/*_local.yaml` run/build dirs.

**`soc/setup.sh`** — four fixes, then ran to completion (exit 0):
1. **C++ driver dest** → `${CHIPYARD_DIR}/generators/firechip/bridgestubs/src/main/cc/bridges/`
   (was dead `firesim-lib/.../cc/bridges/`). Verified symlinks land next to `uart.cc`.
2. **FireMarshal block** → `${CHIPYARD_DIR}/software/firemarshal` + dropped the
   nonexistent `git checkout ubuntu-add`; made best-effort (guarded by dir-exists).
   Ran `init-submodules.sh` successfully (buildroot/linux/busybox submodules cloned).
3. **build.sbt wiring reworked for chipyard-as-top** (no longer patches the unused
   firesim standalone `sim/build.sbt`):
   - Defines `rose` generator project `.dependsOn(rocketchip, testchipip,
     firechip_bridgeinterfaces)` — added the missing `firechip_bridgeinterfaces` dep
     (rose sources import `firechip.bridgeinterfaces.*`). Chisel 6 plugin inherited
     via `rocketLibDeps`, matching gemmini/shuttle/constellation.
   - Adds `rose, firechip_bridgeinterfaces` to chipyard's `.dependsOn(...)` (chipyard
     sources IOBinders/Ports/RoSEConfigs import both). Idempotent guards added.
4. **Symlink loop** now skips (with a warning) any dest whose parent dir is missing,
   instead of hard-failing — so the uninitialized `onnxruntime-riscv` ONNX symlinks
   (DNN-build only) are gracefully skipped (6 expected warnings).

**Verified after run:**
- All 11 RoSÉ Scala/C++ symlinks present and pointing at the right RoSÉ sources
  (bridgestubs cc + scala, bridgeinterfaces, goldengateimplementations, firechip/chip,
  rose generator, chipyard iobinders/config/DigitalTop).
- `chipyard` dependsOn line includes `... fft_generator, rose, firechip_bridgeinterfaces,`.
- Exactly one `lazy val rose` def appended (idempotent); firesim `sim/build.sbt` untouched.

### Why this wiring (build dependency facts)
- `generators/firechip/{bridgeinterfaces,goldengateimplementations}` are copied into
  MIDAS/GoldenGate at build time via `TARGET_COPY_TO_MIDAS_SCALA_DIRS`
  (`firechip/chip/src/main/makefrag/firesim/build.mk`) — this is the Chisel-6↔Chisel-3.6
  bridge. So those two symlink dests are correct as source locations.
- `firechip_bridgestubs` / `firechip` already `.dependsOn(chipyard, firesim_lib,
  firechip_bridgeinterfaces)` — so the stubs see `rose.*` transitively through chipyard.

---

## KNOWN NEXT-STEP BLOCKERS (Scala/driver port — Phase 2/3, NOT setup-script issues)

These will surface at first compile/elaboration and are the real migration work:
1. **`RoSEBridgeModule.scala` package mismatch:** declares `package firesim.bridges`
   but is placed in `goldengateimplementations/` (whose convention is
   `package firechip.goldengateimplementations`) and the stub references
   `moduleName = "firechip.goldengateimplementations.RoSEBridgeModule"`. Package +
   class name need reconciling so the stub resolves the host module.
2. **C++ driver `airsim_t` is the REAL Rose driver, not the legacy AirSim bridge.**
   It is internally consistent and would link as-is: `RoSEBridgeModule` emits
   `genConstructor(..., "airsim_t", "airsim")`, and `airsim.h` defines
   `class airsim_t final : bridge_driver_t` whose ctor takes
   `const ROSEBRIDGEMODULE_struct&` (the struct name Golden Gate derives from the
   `RoseBridgeModule` class). So the `airsim_t`/`airsim` naming is legacy cosmetic
   baggage on the Rose driver — **not a blocker**. Renaming to `rose_t`/`rose`
   (driver + `genConstructor` together) is optional cleanup, deferred. Driver already
   uses the modern API (final, `static char KIND;`, const-ref MMIO, `(simif_t&, …)`).
3. **Chisel 3 → Chisel 6 source port** across the rose generator + stubs (named-arg
   `RegEnable`, `TLHelper.makeClientNode` removal in RoSEDMA, etc.).

---

## STEP 3 — Scala elaboration (IN PROGRESS): chisel6 compile of `firechip`

Driving an sbt `compile` of the `firechip` project (transitively builds chipyard +
rose + bridgestubs + bridgeinterfaces) to surface the chisel6 port errors, fixing
iteratively. Command:
`source env.sh && cd sims/verilator && make launch-sbt SBT_COMMAND=";project firechip; compile"`
Log: `soc/sim/chipyard/rose-firechip-compile.log`.

### Round 1 — `firechip_bridgeinterfaces` failed (FIXED)
- Error: `RoSEBridgePort.scala: object chipsalliance is not a member of package org`
  — `firechip.bridgeinterfaces` is intentionally minimal/pure-Chisel (no CDE; it is
  copied to the GoldenGate compiler), but the file declared
  `case object RoseAdapterKey extends Field[...]` (a CDE `Field`).
- Also found the chipyard-top refactor had left **duplicate** definitions of
  `RoseAdapterKey` and `RoseAdapterParams` in BOTH `bridgeinterfaces`
  (RoSEBridgePort.scala) and `rose` (RoSEGeneratorConfig.scala).
- **Fix (clean split):** `RoseAdapterParams` (pure-Chisel, needed by the cross-compiled
  interface + GoldenGate `RoseKey`/`RoseBridgeTargetIO`) is canonical in
  `bridgeinterfaces`; `RoseAdapterKey` (CDE `Field`) lives only in the `rose` generator.
  Verified the GoldenGate host module (`RoSEBridgeModule`) only references
  `RoseAdapterKey` in commented-out lines, so the move is safe.
  Edits:
  - `RoSEBridgePort.scala`: removed `import ...cde.config.Field` + the `RoseAdapterKey` def.
  - `RoSEGeneratorConfig.scala`: dropped `RoseAdapterKey` from the bridgeinterfaces import
    (now defined locally), removed the duplicate `RoseAdapterParams` case class.
  - `RoSEAdapter/RoSEIO/RoSEDMA.scala` (package `rose`): dropped `RoseAdapterKey` from
    their bridgeinterfaces imports (same-package now).
  - `IOBinders.scala`: import `RoseAdapterKey` from `rose`, `RoseAdapterParams` from
    `bridgeinterfaces` (swapped which package each comes from).
  - `RoSEBridgeStub.scala` (`firechip.bridgestubs`): added `import rose.RoseAdapterKey`
    (uses `p(RoseAdapterKey)`; resolves transitively via chipyard→rose).

### Round 2 — `chipyard` failed (FIXED)
- Error: `AbstractConfig.scala:62: type WithGCDBusyPunchthrough is not a member of
  package chipyard.iobinders`.
- Cause: RoSÉ's `IOBinders.scala` **wholesale-replaces** stock chipyard
  `iobinders/IOBinders.scala` (via symlink), and the chipyard-top author had
  **commented out** `WithGCDBusyPunchthrough` — but stock 1.13.0 `AbstractConfig.scala`
  (not replaced by RoSÉ) still references it. This is the divergence cost of the
  replace-whole-file approach: RoSÉ's IOBinders must remain a superset of stock.
- Verified `GCDBusyPort` exists in RoSÉ's `Ports.scala`; it was the only commented-out
  binder. **Fix:** un-commented `WithGCDBusyPunchthrough` in `IOBinders.scala`.

### Round 3 — `firechip` COMPILES ✅
- `[success] Total time: 10 s`. The full Scala stack (chipyard + rose + firechip
  bridgestubs/bridgeinterfaces/chip) compiles under **Chisel 6**. Verified genuine
  (class files produced: `rose/RoseAdapterKey.class`, `RoseAdapterTL.class`,
  `firechip/bridgestubs/RoseBridge.class`, …).
- **Takeaway:** the chipyard-top Scala refactor was already largely Chisel-6-ready; the
  only blockers were the bridgeinterfaces/CDE split and one commented-out stock binder.

---

## STEP 4 — RTL elaboration ✅ (Scala elaboration WORKS)

`make firrtl CONFIG=RoseTLRocketConfig CONFIG_PACKAGE=chipyard.config` in
`sims/verilator` → **exit 0**, produced
`generated-src/chipyard.harness.TestHarness.RoseTLRocketConfig/...fir` (449k lines) +
DTS + annotations + regmaps. The plain chipyard `TestHarness` elaborated the punched-out
Rose port fine (no missing-harness-binder issue).

**Verified the full Rose hardware is in the elaborated design (not optimized away):**
- `module RoseAdapterMMIOChiselModule` — Rose adapter (TX/RX queues + arbiter).
- `TLInterconnectCoupler_pbus_to_RoseAdapter` — adapter MMIO on pbus; DTS node
  `RoseAdapter@2000 { compatible = "ucbbar,RoseAdapter"; }`.
- `TLInterconnectCoupler_fbus_from_camdma...DMA0...` — camera DMA engine on fbus
  (DMA0 @ 0x88000000, from the config's `DstParams(port_type="DMA", ...)`).

**Milestone reached: the RoSÉ Chisel-6 Scala stack compiles AND elaborates to RTL.**
Notably none of the feared Chisel 3→6 API breakage (named-arg `RegEnable`,
`TLHelper.makeClientNode`) appeared in the elaborated path — the chipyard-top refactor
was already Chisel-6-clean apart from the two fixes above.

Log: `soc/sim/chipyard/rose-elaborate.log`.

---

## STEP 5 — FireSim metasim build (IN PROGRESS)

### GoldenGate host-module fixes (pre-emptive, before build)
Confirmed the stock convention from UART: stub `moduleName =
"firechip.goldengateimplementations.UARTBridgeModule"` ↔ goldengate `package
firechip.goldengateimplementations; class UARTBridgeModule`. RoSÉ's were mismatched:
- `RoSEBridgeModule.scala`: `package firesim.bridges` → **`package
  firechip.goldengateimplementations`** (the dir whose contents are copied into MIDAS).
- class `RoseBridgeModule` → **`RoSEBridgeModule`** to match the stub's `moduleName`
  string (`...RoSEBridgeModule`). Safe: the new stub references the host module ONLY by
  that string (its `Bridge[HostPortIO[RoseBridgeTargetIO]]` takes just the target-IO
  type param), so no Scala-name reference breaks.
Imports already matched the stock UART goldengate module (midas.widgets, chisel3, cde
Parameters, firechip.bridgeinterfaces, firesim.lib.bridgeutils).

### Build invocation (metasim, Verilator host)
Canonical makefrag path (from `firesim/deploy/util/targetprojectutils.py`):
`generators/firechip/chip/src/main/makefrag/firesim`. firechip `config.mk` defaults:
`DESIGN=FireSim`, `DESIGN_PACKAGE=firechip.chip`, `TARGET_CONFIG_PACKAGE=firechip.chip`,
`PLATFORM_CONFIG=BaseF1Config`, `TARGET_SBT_PROJECT=firechip`.

Running `make verilog` first (= GoldenGate only: elaboration → MIDAS/GoldenGate →
`$(simulator_verilog)` + bridge driver header; stops before the long Verilator C++
compile — fail-fast for GoldenGate Scala errors):
```
cd sims/firesim && source sourceme-manager.sh --skip-ssh-setup && cd sim &&
make verilog TARGET_PROJECT=firesim \
  TARGET_PROJECT_MAKEFRAG=<chipyard>/generators/firechip/chip/src/main/makefrag/firesim \
  TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig
```
Confirmed it sets up `midas/.../target-symlinks` (copies bridgeinterfaces +
goldengateimplementations into MIDAS) and runs firechip elaboration via
`sims/firesim-staging`. Log: `soc/sim/chipyard/rose-metasim-verilog.log`.

### Round 1 — GoldenGate ran, failed in a custom transform (FIXED)
- Target FIRRTL elaborated fine (`.fir`/`.anno.json` produced). GoldenGate then threw:
  `java.io.FileNotFoundException: ../../../sw/generated-src/rose_c_header/rose_port.h`.
- Cause: `RoSEBridgeModule.scala` `genRoseCPortHeader()` WRITES a generated SoC-side
  register-map header (`rose_port.h`, consumed by the target baremetal SW in
  `soc/sw/rose-images/...`) via a hardcoded relative path. `../../../sw/...` was correct
  for the OLD firesim-as-top layout (CWD `soc/sim/firesim/sim` → `../../../sw` =
  `soc/sw` ✓) but in chipyard-as-top the GoldenGate CWD is
  `soc/sim/chipyard/sims/firesim/sim`, so `../../../sw` = nonexistent
  `soc/sim/chipyard/sw`; `FileWriter` throws because the parent dir is absent.
- **Fix:** resolve the output dir from `$ROSE_DIR` (absolute, layout-independent) with a
  legacy relative fallback, and `mkdirs()` the parent so it never throws. Also
  `export ROSE_DIR` in `rose-setup.sh`, and pass `ROSE_DIR=<repo>` into the build.
- Re-running `make verilog` with `ROSE_DIR` exported (GoldenGate recompiles the changed
  `RoSEBridgeModule`).

### Round 2 — GoldenGate SUCCEEDS ✅
- `make verilog` exit 0. Produced `FireSim-generated.sv` (13 MB simulator Verilog),
  `FireSim-generated.const.h` (bridge driver header), and regenerated
  `soc/sw/generated-src/rose_c_header/rose_port.h` at the committed location.
- The Rose bridge is fully wired into the GoldenGate output:
  - `#include "bridges/airsim.h"` (the Rose driver) + `new airsim_t(...)` registration.
  - `ROSEBRIDGEMODULE_struct` static_asserts (30 fields: out/in/in_bigstep/in_budget/
    in_ctrl/cycle_*/bww_config/config_routing/arb_counter_*).
- **The current `airsim.h` driver struct matches the generated 30-field layout exactly**
  — chipyard-top's driver was already updated for this bridge, so static_asserts pass.
- Conclusion: `RoSEBridgeModule` compiles cleanly under Chisel 3.6 in MIDAS/GoldenGate;
  the package/class-name fix + the `rose_port.h` path fix were the only GoldenGate
  blockers.

## STEP 6 — Full Verilator metasim binary (`make verilator`, IN PROGRESS)
Reuses the GoldenGate output; compiles the driver (`airsim.cc` + midas runtime) and
Verilates the 13 MB `.sv` → `VFireSim` metasim binary. Confirmed:
- Driver source list includes `.../firechip/bridgestubs/src/main/cc/bridges/airsim.cc`.
- `RoSEBridgeModule_0` is present in the generated Verilog
  (`emul.FPGATop.RoSEBridgeModule_0.rxfifo...`) — the bridge is in the simulator RTL.
- Only a benign `SYNCASYNCNET` Verilator lint warning (CDC on the bridge rxfifo reset).
Log: `soc/sim/chipyard/rose-metasim-verilator.log`. (Verilator + g++ of a full FireSim
metasim is the long step, ~20-40 min.)

### Round 1 — Verilator lint fatal (FIXED via flag)
- `%Error: Exiting due to 1 warning(s)` — `SYNCASYNCNET` on the Rose bridge's async
  rxfifo reset (`emul.FPGATop.RoSEBridgeModule_0.rxfifo.sink.widx_widx_gray.output_chain_reset`,
  flopped as both sync and async). FireSim runs Verilator with `-Wall`, which makes any
  unwaived lint warning fatal. (This is the CDC issue the chipyard-top history flagged:
  "FIX: fixing for CDC, FPGA test pending" — it matters for FPGA timing closure, but is
  irrelevant to functional metasim.)
- **Fix (no submodule edit):** the firechip makefrag computes `EXTRA_VERILATOR_FLAGS`
  (default `--assert` for non-cva6) and the midas Makefrag appends it after `-Wall`. A
  command-line assignment overrides the makefile's plain `=`, so pass
  `EXTRA_VERILATOR_FLAGS='--assert -Wno-SYNCASYNCNET'` → waives the lint cleanly.
- **Persistence TODO:** this flag is currently passed on the build command line. To make
  RoSÉ metasim builds reproducible, either bake the waiver in (e.g. a RoSÉ-owned `.vlt`
  injected via setup.sh, or the build invocation) OR properly fix the bridge rxfifo CDC
  reset (the real FPGA-timing fix). Not needed for metasim correctness.
- Re-ran with the waiver: Verilator passed lint, generated the C++ model, and entered the
  driver + verilated-model g++ compile (compiling `airsim.cc`, `uart.cc`, etc.).

### Round 2 — METASIM BINARY BUILT ✅✅
- `make verilator` exit 0. Produced **`VFireSim`** — a 50 MB x86-64 ELF executable at
  `sims/firesim/sim/generated-src/f1/f1-firesim-FireSim-RoseTLRocketMMIOOnlyConfig-BaseF1Config/VFireSim`.
- The final link line includes **`airsim.o`** (the Rose bridge driver) alongside the
  midas runtime + other bridge drivers → the Rose bridge driver compiled against the
  generated `ROSEBRIDGEMODULE_struct` header and linked cleanly.

## ✅✅ MILESTONE: FireSim Verilator metasim of a RoSÉ config builds end-to-end
Full pipeline working on chipyard-as-top / Chisel 6 (Chipyard 1.13.0 + FireSim
1.20.1): chipyard elaboration → GoldenGate (Rose bridge host module, Chisel 3.6) →
generated simulator Verilog + driver header → driver + Verilator C++ compile → linked
`VFireSim` metasim binary.

### Total fixes required to get here (all small, mechanical/structural — no deep RTL work)
1. Submodule restructure to chipyard-as-top (Chipyard 1.13.0, nested FireSim 1.20.1).
2. Setup scripts (`rose-setup.sh`, `soc/setup.sh`) repathed for the new layout +
   build.sbt wiring.
3. `RoseAdapterKey`/`RoseAdapterParams` package split (CDE out of pure-Chisel
   bridgeinterfaces); un-commented `WithGCDBusyPunchthrough`.
4. `RoSEBridgeModule`: `package firechip.goldengateimplementations` + class
   `RoSEBridgeModule` (match stub `moduleName`).
5. `rose_port.h` output path via `$ROSE_DIR` + `mkdirs()`.
6. Verilator `SYNCASYNCNET` CDC lint waived via `EXTRA_VERILATOR_FLAGS`.

### Next steps (not yet done)
- **Run** the metasim (`VFireSim`) against a workload + the RoSÉ synchronizer
  (`deploy/hephaestus/`) to validate an end-to-end co-sim packet round-trip.
- Make the `SYNCASYNCNET` waiver + `ROSE_DIR` export persistent (and/or properly fix the
  bridge rxfifo CDC reset for eventual FPGA bitstream timing).
- Then FPGA bitstream build (`xilinx_alveo_u250`).

---

## STEP 7 — Bridge testing assets (dummy env + baremetal packet tests) — INVENTORY

All present on this branch, and there is a **purpose-built dummy test harness** (no real
simulator needed):
- **Baremetal packet tests:** `soc/sw/rose-images/airsim-packettest/*.c` (~20; simplest
  `airsim-packettest.c` does a TX camera-request → RX-data round-trip using the generated
  `rose_port.h` macros — `ROSE_TX_ENQ_READY`, `ROSE_RX_DATA_2`, etc. — matching our
  `RoseTLRocketMMIOOnlyConfig` port layout). Commands in `rose_packet.h` (CS_CAMERA_LEFT…).
- **Dummy/no-simulator gym envs:** `deploy/hephaestus/envs/customized_env/LQR_gym_env.py`
  (pure numpy/scipy), plus `mujoco` inverted-pendulum and `middlebury` (image dataset) —
  none require AirSim/Unreal/GPU. Configs in `deploy/config/config_gym_*.yaml`.
- **Harness:** `deploy/hephaestus/rose.py` (`--task build` builds the test via
  `DummySynchronizer`; `--task run` starts the synchronizer + FireSim + packet loop),
  `gym_synchronizer.py` (`DummySynchronizer`/`Synchronizer`), `synchronizer.py` (TCP
  **server** on SYNC_PORT 10001 / DATA_PORT 60002). The bridge driver `airsim.cc` is a
  TCP **client** that connects to those ports — so the synchronizer must be listening first.

### Blockers to actually running it (same path-staleness pattern as the other scripts)
1. `soc/sw/build_packettest.py`: uses OLD firesim-as-top paths
   (`../sim/firesim/target-design/chipyard/tests`, `.../deploy/workloads/...`) → need
   `soc/sim/chipyard/tests` and `soc/sim/chipyard/sims/firesim/deploy/workloads/...`.
2. **Bigger:** that script builds via `make PROGRAMS=<t>` in chipyard `tests/`, but
   **chipyard 1.13.0's `tests/` is CMake-based (`CMakeLists.txt`, no Makefile)** — the
   baremetal build mechanism changed. Options: integrate into the CMake tests, or build
   the test standalone with the riscv toolchain against `tests/htif.ld` + `mmio.h`.
3. `gym_synchronizer.py` `genRoSECPacketHeader()` writes to a hardcoded
   `/scratch/iansseijelly/RoSE/...` path → repath to `$ROSE_DIR`.
4. `rose.py --task run` drives the FireSim **manager** (`firesim runworkload`, needs
   manager/config_runtime + metasim-enabled setup). Alternative: run our built `VFireSim`
   binary **directly** on the test `.riscv` payload + start `DummySynchronizer` — simplest
   path for a first bridge test, bypassing the manager.

### STEP 8 — Bridge co-sim test, direct-VFireSim approach (IN PROGRESS)
**Done:**
- Built `airsim-packettest.riscv` standalone (chipyard 1.13.0 tests/ is CMake now, so
  bypassed it): `riscv64-unknown-elf-gcc -march=rv64imafd -mabi=lp64d -mcmodel=medany
  -specs=htif_nano.specs -static -T tests/htif.ld -I tests -I soc/sw/generated-src/rose_c_header
  airsim-packettest.c -o tests/airsim-packettest.riscv`. (`riscv-pk/encoding.h` on default
  toolchain path; `mmio.h` in tests/; rose headers in generated-src.)
- Metasim run command: `make run-verilator TARGET_PROJECT=firesim
  TARGET_PROJECT_MAKEFRAG=<firechip makefrag> TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig
  SIM_BINARY=<.riscv>` → `VFireSim +permissive … +permissive-off <binary>`. UART/printf
  output lands in `sims/firesim/sim/airsim-packettest.out` (via spike-dasm).
- Ran VFireSim alone (no synchronizer): launched (`simulator_entry: start...`) but produced
  no UART. **Confirmed cause:** `airsim.cc connect_synchronizer()` does
  `while (connect(sync_sockfd,…) < 0)` — it **busy-blocks at driver construction until the
  synchronizer is listening on 10001**. So the bridge driver is alive and behaving as
  designed; the sim can't progress without the host synchronizer. (Bridge HW/driver side
  validated.)

**Correction on an earlier overclaim:** a VFireSim-alone run produced empty UART output,
which I wrongly read as "bridge alive, blocking on connect." That was NOT a confirmation —
empty output is also consistent with the sim never advancing. Root cause (verified below):
the Rose bridge gates FireSim advance on a cycle budget that the synchronizer must grant,
so with NO synchronizer the sim never boots; plus the driver's printf is block-buffered and
lost on timeout-kill.

### Minimal stdlib synchronizer → REAL end-to-end confirmation ✅
Wrote `deploy/hephaestus/minimal_sync.py` (stdlib only — NO gymnasium/numpy/opencv): accept
on :10001 → `CS_DEFINE_STEP` → `CS_CFG_ROUTE` → continuously `CS_GRANT_TOKEN` → log SoC TX
packets → answer the camera request on channel 2. Ran it + `make run-verilator`. Observed:
```
[minimal_sync] bridge connected from ('127.0.0.1', 35126)
[minimal_sync] granted 2000…14000 tokens
[minimal_sync] RX packet #1: cmd=0x11 num_bytes=0     <- SoC's CS_CAMERA_LEFT reached host
```
and the `.out` commit-log showed the Rocket core booting/executing. **Genuinely confirmed at
runtime:** (1) bridge TCP connect; (2) sync protocol (`CS_DEFINE_STEP`+`CS_GRANT_TOKEN`)
advances the metasim — proves the cycle-budget gating; (3) SoC boots + runs the baremetal
test; (4) **SoC→Rose-adapter→bridge→host TX path works** (camera request arrived correctly
framed). The earlier empty-output run is now explained: no token grants → no advance.

**Not yet confirmed:** RX-return completing the test. The 16384-word response was sent, but
the metasim (full commit-logging, 256×256 ×5 iters) hit the 180s timeout before the receive
loop finished / printed `byte_read`. Next: run a smaller/shorter test (e.g.
`-double-rdcycle-small`, or reduce IMG/NUM_ITERS) and/or disable commit-logging for speed to
see the test complete + exit(0).

### RX loop (host→SoC): NOT closed with the minimal synchronizer
Tried to close the return path with a status-echo probe (SoC sends request, then
continuously echoes the STATUS register over TX so the host can watch the
`DEQ_VALID_2` bit). Result: **the SoC's status stays `0x1`** (TX-ready set,
`DEQ_VALID_2`=0x2 never sets) — i.e. the host's response is not delivered to channel 2.
Adding `CS_CFG_BW` (bandwidth config, which the bridge rxcontroller needs — threshold
defaults to 0 = blocked) did not fix it. Also observed: with an **infinite-loop** SoC
program (no `exit()`), the sim still terminates early ("connection closed by bridge"),
so the bridge driver is ending the simulation shortly after startup — which also limits
observation of RX delivery.

**Likely root cause:** the minimal synchronizer is an incomplete subset of the RoSÉ sync
protocol. The driver (`airsim.cc`) implements a full **budget / bigstep request-response
handshake** (`send_budget`, `send_bigstep`, `fsim_txbudget`, `CS_REQ_CYCLES →
report_cycles`, `in_budget_*`/`in_bigstep_*` MMIO) that the real `gym_synchronizer.Synchronizer`
drives; the minimal version only blindly sends `CS_GRANT_TOKEN` and never participates in
that handshake. That is enough to prove connection + sim-advance + the **TX path**, but
not a correct bidirectional exchange. **Closing the RX loop reliably needs the real
Synchronizer** (full protocol) — i.e. stand up the gym Python env + the path fixes — or a
much more faithful re-implementation of the budget/bigstep protocol in the minimal sync.

**Confirmed so far (honest):** bridge connect ✅, sync config received by driver (step,
route, bw) ✅, sim advances ✅, SoC boots ✅, **TX path SoC→host works** ✅. **RX path
host→SoC: NOT yet confirmed** (DEQ_VALID_2 never sets; needs the full sync protocol).

---

## STEP 9 — Real gym synchronizer set up + co-sim running ✅

Stood up the real `gym_synchronizer.Synchronizer` in a dedicated Python venv and ran it
against the metasim (manager-free). Result: **the full RoSÉ co-sim loop runs end-to-end.**

### Python environment (documented)
Created a venv with **system Python 3.12** (the chipyard conda env is Python 3.14 — too
new for these packages; keep the synchronizer env separate):
```
cd deploy
python3.12 -m venv .venv-rose
. .venv-rose/bin/activate
pip install -r requirements.txt      # see deploy/requirements.txt
```
`deploy/requirements.txt` (core, no-AirSim path — enough for the MiddleBury/LQR dummy
envs): `gymnasium numpy scipy pyyaml pandas matplotlib opencv-python-headless`.
Optional (only if using the AirSim env, which needs Unreal): `airsim msgpack-rpc-python`,
or `gymnasium[mujoco]` for the inverted-pendulum env.

### Code fixes required (chipyard-as-top / modern Python)
- `deploy/hephaestus/register_envs.py`: removed eager `from envs.airsim…` /
  `from envs.mujoco…` imports — registration uses string entry_points (lazy), so the
  optional envs are only imported if `gym.make()`'d. Keeps the dep set small.
- `deploy/hephaestus/gym_synchronizer.py` `genRoSECPacketHeader`: replaced the hardcoded
  `/scratch/iansseijelly/...` path with `$ROSE_DIR`-based resolution + `os.makedirs`.
- `deploy/hephaestus/envs/middlebury/middlebury.py`: `reset()` now takes
  `(*, seed=None, options=None)` and calls `super().reset(seed=seed)` (gymnasium 1.x API;
  it was old-`gym` style); added `metadata={"render_modes":["rgb_array"]}`.
- `deploy/config/config_deploy_gym.yaml`: `firesim_step` 10_000_000 → **10_000** for
  metasim (the FPGA-scale step would take ~1000s/step at metasim's ~kHz rate).
- New `deploy/hephaestus/run_sync_only.py`: a **manager-free launcher** that runs
  `Synchronizer` + `ServerThread` WITHOUT the `firesim runworkload` manager thread that
  `rose.py --task run` uses — so it pairs with a directly-launched `VFireSim`.

### How to run the co-sim (metasim)
1. Terminal A (synchronizer, in the venv, from `deploy/hephaestus`):
   `ROSE_DIR=<repo> python run_sync_only.py`  → listens on :10001.
2. Terminal B (metasim): `make run-verilator TARGET_PROJECT=firesim
   TARGET_PROJECT_MAKEFRAG=<firechip makefrag> TARGET_CONFIG=RoseTLRocketMMIOOnlyConfig
   SIM_BINARY=<.riscv>` (after sourcing chipyard `env.sh` + firesim `sourceme-manager.sh`).

### Result — co-sim loop LIVE and bidirectional ✅
Synchronizer log showed: bridge connected → sent bandwidth (ch0/1/2) + routes (0x10→0,
0x11→0, 0x12→1, …) → reset/stepped the MiddleBury env → entered "Starting RoSE Simulation
Loop" → **received the SoC's camera request** (`Dequeued data packet: [cmd: 0x11,
num_bytes: 0000]`) → stepped the sim continuously (`sim time = 0.02…0.08s`). Crucially the
**metasim stayed alive the full run (`EXIT 124`=timeout)** — vs the ~1s early-death with
the minimal sync — confirming the bridge needs the real budget/bigstep handshake, which
the real Synchronizer provides. So the RoSÉ bridge co-sim is functional end-to-end on the
chisel6 metasim: SoC↔bridge↔synchronizer↔gym-env, request received, env observations served.

### Remaining tuning (not blocking)
- **Workload/route matching:** the MiddleBury config routes `camera_left`(0x11)→channel 0
  (= DMA0 in `RoseTLRocketConfig`), while `airsim-packettest.c` reads via MMIO channel 2
  (reqrsp1). To have a baremetal MMIO test fully consume the served image, pair it with a
  config that routes the camera header to a reqrsp channel the test reads (or use a
  DMA-consuming workload). The bridge mechanics + protocol are validated; this is
  app-level config matching.
- Two benign gymnasium warnings (obs dtype int64 vs uint8 in MiddleBury `dummy`/`lat_test`;
  render_mode) — cosmetic.

---

## STEP 10 — Close the RX loop with validation: harness complete; delivery precisely localized

Built a full **known-pattern RX-validation** path so the SoC side can verify the bytes it
receives (not just that the request was sent):
- **`deploy/hephaestus/envs/pattern/pattern_env.py`** — `PatternEnv-v0`, a no-simulator env
  serving a known uint32 pattern `0xC0DE0000 + i` (16 words) in obs field `pattern`.
  Registered in `register_envs.py`.
- **`deploy/config/config_gym_PatternEnv-v0.yaml`** — routes `camera_left`(0x11)→**channel 2**
  (= reqrsp1 = `ROSE_RX_DATA_2`, MMIO-readable), `channel_bandwidth: [[0,0,0]]`.
- **`config_deploy_gym.yaml`** → `gym_env: PatternEnv-v0`.
- **`soc/sw/rose-images/airsim-packettest/airsim-packettest-rxvalidate.c`** — requests,
  reads channel 2, **validates the bytes equal the known pattern** (scans, robust to header
  framing), and re-echoes a PASS/FAIL marker + the data over TX (header 0x42, <0x80 so it's
  logged as a data packet) — buffering-immune validation, verifiable in the synchronizer log.

### Key bridge-protocol facts learned (documented for future work)
- **RX data wire format (host→SoC, cmd<0x80):** the bridge TCP reader expects
  `[cmd, big_step, budget, num_bytes, data…]` (5 fields) — the data is *bundled* with
  big_step+budget and queued into `budget_rx_queue`, then fed to `in_bits` by
  `schedule_firesim_data()`. The synchronizer's `Payload_Packet.encode()` produces exactly
  this (via `tag_step`), so the host format is correct.
- **Bandwidth semantics (rxcontroller):** `bandwidth_threshold` = cycles *between* word
  deliveries (inverse bandwidth); **0 = unlimited**. (A large value nearly stalls delivery —
  my initial 1<<20 was backwards.)
- **Channel→port map:** `send_route(header, channel)` → arbiter `io.rx(channel)`; dst_ports
  in `RoseTLRocketConfig` = [DMA0=0, reqrsp0=1, reqrsp1=2]; so channel 2 = `ROSE_RX_DATA_2`.

### STEP 12 — reqrsp RX path WORKS end-to-end (no adapter bug)
Instrumented the SoC-side adapter (`RoseAdapterMMIOChiselModule`) with chisel `printf` on the
reqrsp rx-FIFO (target printf emits via `$display` to the metasim `.out`). With the **real
synchronizer** + PatternEnv (route 0x11→ch2), the trace shows the full chain working:
```
[SOCDBG] rxfifo(1) ENQ bits=0x00000011 / 0x00000040 / 0xc0de0000 / 0xc0de0001 ...   (from bridge)
[SOCDBG] rxfifo(1) DEQ(SoC read) bits=0x00000011        <- SoC MMIO-reads header
[SOCDBG] rxfifo(1) DEQ(SoC read) bits=0x00000040        <- count
[SOCDBG] rxfifo(1) DEQ(SoC read) bits=0xc0de0000        <- SoC reads pattern[0]  ✅
[SOCDBG] rxfifo(1) DEQ(SoC read) bits=0xc0de0001        <- pattern[1]            ✅
```
So **the SoC receives and reads the exact served pattern via `ROSE_RX_DATA_2`** — the reqrsp
adapter is CORRECT; there was no bug. **The earlier reqrsp failures were because `minimal_sync`
never drove the budget/bigstep handshake**, so the bridge arbiter never advanced. With the
real `gym_synchronizer.Synchronizer` (full protocol), data flows host→bridge→SoC and the SoC
reads it correctly. The SoC's own PASS-marker echo just needs the (heavily instrumented, slow:
~11k cycles/s) metasim to run long enough past bare-metal boot for the TX echo to traverse;
the `DEQ(SoC read)` of `0xc0de0000/0xc0de0001` is itself direct SoC-side validation.

**Bottom line for the RX loop:** host→SoC reqrsp delivery is FUNCTIONAL end-to-end on the
chisel6 metasim with the real synchronizer.

### STEP 13 — DMA path (IN PROGRESS); earlier "no memory model" hypothesis was WRONG
Correction: `WithDefaultMMIOOnlyFireSimBridges` (RoSÉ BridgeBinders.scala:167) **already
includes `WithFASEDBridge`** — so the FASED DRAM model IS present (the commented
`WithDefaultMemModel` is redundant). The elaborated DTS confirms `memory@80000000 reg=<0x0
0x80000000 0x4 0x0>` = DRAM at 0x80000000 spanning 16 GB, which **covers the DMA target
0x88000000**. And the SoC boots/executes from this DRAM. So the DMA's writes have valid
backing memory; the earlier DMA failure is NOT a missing-memory issue.
- `CamDMAEngine` (`RoSEDMA.scala`): `mIdle→mWrite(TL Put @ DMA_address+counter)→mResp(wait D)
  →mIdle`; `counter += 4` per completed write; `cam_buffer = counter >= counter_max`. The
  DMA's `TLClientNode` is on the fbus (`fbus.coupleFrom("cam-dma")`), routing to the mem port.
- Instrumented `RoSEDMA` with plain `printf` (`[DMADBG]`) on recv / TL Put / TL resp /
  cam_buffer, and running the DMA validation test (route 0x11→ch0, counter_max=64=16 words).
  This will show directly whether the DMA receives the pattern, the TL writes complete
  (mem.d.fire), and cam_buffer flips — i.e. whether the earlier DMA failure was a real
  CamDMAEngine/TL issue or just metasim slowness (as the reqrsp "failure" turned out to be).

### DMA path WORKS end-to-end ✅
The `[DMADBG]` trace + commit log prove the full DMA chain functions (PatternEnv `0xC0DE0000+i`,
route 0x11→ch0, counter_max=64):
```
[DMADBG] recv word=0xc0de0000 ... 0xc0de000f                 (16 words received from bridge)
[DMADBG] TL Put addr=0x088000000 data=0xc0de0000 ...          (16 TileLink Puts to 0x88000000+)
[DMADBG] TL resp OK; counter 0->4 ... 60->64                  (ALL 16 writes complete via FASED)
```
and the SoC then reads them straight back from DRAM (commit log `lw`):
```
lw a2,0(a2): addr=0x88000000 -> 0xc0de0000
             addr=0x88000004 -> 0xc0de0001
             ...              -> 0xc0de0007   (exact consecutive pattern)
```
So: `CamDMAEngine` receives the served pattern → writes all 16 words to DRAM @0x88000000
(every TL write completes — FASED model responds) → `cam_buffer` flips → the SoC polls the
flip and reads the DMA buffer, getting the **exact pattern**. The DMA version of the RoSÉ
bridge is FUNCTIONAL on the chisel6 metasim. (As with reqrsp, the SoC's final PASS-echo just
needs the slow instrumented metasim to run longer past the read; the `lw … -> 0xc0de000X`
commit-log reads are themselves direct SoC-side validation.)

**Both RX paths now validated end-to-end: reqrsp (MMIO read) AND DMA (mem read).** No bridge,
API, adapter, or DMA RTL bug — the whole RoSÉ co-sim RX loop works with the real synchronizer.

### STEP 14 — Final clean + captured PASS ✅
Removed ALL debug instrumentation (driver `[RXDBG]`/counter reads, and chisel `[RTLDBG]`/
`[SOCDBG]`/`[DMADBG]` prints in RoSEBridgeModule/RoSEAdapter/RoSEDMA; restored RoSEDMA's
original `SynthesizePrintf`). Converted the validation tests to: validate → `printf` PASS/FAIL
→ `exit()` (no infinite re-echo loop), so the metasim finishes fast and flushes cleanly.
Clean rebuild + DMA run captured:
```
DMAVALIDATE: PASS rx[0]=0xc0de0000 rx[15]=0xc0de000f
```
i.e. the SoC validated all 16 DMA'd pattern words and exited cleanly (`[success]`). The RoSÉ
DMA co-sim RX path is confirmed working on a CLEAN (no-debug) chisel6 metasim.

Confirmed the SoC also validates + echoes: after the 4 `DEQ(SoC read)` (incl. `0xc0de0000`/
`0xc0de0001`), the commit log shows the SoC in the test's post-echo pacing delay loop
(`pc=0x80000368`, `addiw a5,a5,1; sw/lw 12(sp)`), i.e. it ran the scan/validate AND the TX
echo. The host just didn't *capture* the `0x42` echo in time — the instrumented metasim runs
~11k cycles/s and the echo→bridge→sync round-trip didn't finish within the timeout. To get a
clean top-level PASS, remove the debug `printf`s (faster metasim) and re-run; not needed for
correctness, which the `DEQ(SoC read)` of the exact pattern already proves.

### STEP 11 — Instrumented confirmation: NOT an API bug, NOT a bridge-RTL bug (it's SoC-side)
Per the request to confirm RTL-vs-API, instrumented BOTH the driver (C++) and the bridge
RTL (chisel `printf` + the arbiter's built-in MMIO debug counters). Layered, definitive trace
(host→SoC, PatternEnv serving `0xC0DE0000+i`):

| Layer | Instrument | Result |
|---|---|---|
| Synchronizer | `[serve]` print | serves exact pattern `0xc0de0000..0xc0de000f` ✅ |
| **Driver / API** | `[RXDBG]` prints | `schedule queued cmd=0x11 bigstep=1 budget=0 num_bytes=64`; bigstep+budget INJECTED; **18× `in_bits INJECTED`** (`0x11,0x40,0xc0de0000..f`, `in_ready=1` each) ✅ |
| **Bridge arbiter** | `arb_counter_*` regs | `tx_fired=18 sheader=1 budget_fired=1 rx0_fired=16` ✅ |
| **Bridge → SoC port** | chisel `printf` on `target.rx(i)` | **`target.rx(2) -> SoC DELIVERED`** ×18 (`0x11,0x40,0xc0de0000..f`) ✅ (and `target.rx(0)`×16 for the DMA route) |
| SoC validation echo | `cmd 0x42` in sync log | **NOT received** ✗ |

**Conclusion: the API/driver and the entire FireSim bridge RX path are CORRECT** — the served
bytes are injected, routed, and physically delivered to the SoC's target port (`target.rx`).
The remaining gap is **SoC-side**, in the `rose` generator's target hardware:
- **reqrsp path (ch2):** `target.rx(2)` receives all 18 words, but the SoC's
  `RoseAdapter` apparently isn't exposing them via `ROSE_RX_DATA_2`/`DEQ_VALID_2` (the SoC
  test never reads/echoes). → suspect the SoC-side adapter rx-channel→MMIO mapping.
- **DMA path (ch0):** `target.rx(0)` receives 16 words, but `RoseTLRocketMMIOOnlyConfig` has
  **no backing memory** (`WithDefaultMemModel` is commented out in RoSEFireSimConfigs.scala),
  so the `CamDMAEngine`'s writes to `0x88000000` have nowhere to land → no buffer flip. This
  is a CONFIG gap, not an RTL bug.

So: earlier "bridge injection is broken" was WRONG. Driver + bridge are proven correct via
instrumentation; the close-out work is SoC-side (`rose` RoseAdapter reqrsp→MMIO path) and/or
adding a memory model to the FireSim config for the DMA path.

### (superseded) earlier diagnosis (before target.rx instrumentation)
Tested the validation both ways: (a) reqrsp → channel 2 (`ROSE_RX_DATA_2` MMIO read), and
(b) DMA → channel 0 (`CamDMAEngine` → mem `0x88000000`, `airsim-packettest-dmavalidate.c`).
In BOTH, the synchronizer log confirms it **serves the exact pattern**
(`[serve] cmd=0x11 obs_data=[0xc0de0000..]`) and routes it, and the SoC's request arrives —
but the SoC **never receives the data** (no `0x42` echo; reqrsp `DEQ_VALID_2` never sets / DMA
buffer never flips). Since the channel/route/bandwidth differ between the two paths but the
result is identical, the cause is NOT routing — it is the **bridge host→SoC RX *injection***
itself: the driver queues the served data into `budget_rx_queue` and `schedule_firesim_data()`
pushes it to `fsim_tx_bigstep`/`fsim_txbudget`/`fsim_txdata`, but it is not reaching the SoC
(the `in_bigstep`/`in_budget`/`in_bits` MMIO handshake to the RTL, or the budget/bigstep
gating, is not completing). This needs **waveform/RTL-level debugging of the bridge's RX
injection path** — beyond the software/config layer.

**Net for STEP 10:** the entire host-side stack is built and CONFIRMED correct (real
synchronizer, full protocol, known-pattern env, both reqrsp + DMA validation tests, correct
routing/bandwidth, host serves the exact bytes). The validation harness will report PASS the
moment host→SoC injection works. The one remaining defect is in the bridge's host→SoC RX
injection (driver/RTL), precisely localized but not yet fixed.

### Status: host serving CONFIRMED; SoC-side RX delivery (both reqrsp & DMA) NOT completing
With the correct env/route/bandwidth, the synchronizer log shows it **serving the exact
pattern** (`[serve] cmd=0x11 … obs_data=[0xc0de0000..0xc0de0007]`) and the SoC's request
arriving (`Dequeued data packet: [cmd: 0x11]`). But the SoC's `ROSE_RX_DEQ_VALID_2` never
sets (no `0x42` echo), i.e. **the bridge does not deliver the served data to the reqrsp
channel** (host→SoC). Tried with both the minimal and the full real synchronizer → same.
The failure is now precisely localized: **host side correct; bridge host→SoC reqrsp delivery
is the remaining issue** — a bridge/RTL-level matter (route-config handshake / `in_bits`
feeding / arbiter reqrsp delivery) needing waveform-level debugging, OR use of the **DMA
receive path** (channel 0 → `CamDMAEngine` → memory @0x88000000), which is RoSÉ's actual
camera-data path and may exercise a different (working) delivery route. The validation
HARNESS is complete and will confirm correctness the moment delivery works.

### Full gym harness — AirSim path (only if using the real AirSim env / Unreal)
Only needed for real env data (images), not bridge mechanics. To use it: install the
hephaestus Python stack (`gymnasium`/`numpy`/`scipy`/`pyyaml`/`opencv` — `deploy/setup.sh`
provisions it; not in the chipyard conda env), fix `genRoSECPacketHeader`'s hardcoded
`/scratch/iansseijelly/...` path, and add a manager-free launcher reusing `Synchronizer` +
`ServerThread` (default env `MiddleBuryEnv-v0` serves stereo images for the camera test).

## STEP 15 — FireSim FPGA bitstream build (Xilinx Alveo U250)

After the metasim path was validated end-to-end (STEP 14 DMA PASS), built a real FPGA
bitstream for on-board testing.

**Target / recipe.** Reused the existing recipe in
`soc/sim/config/config_build_recipes_local.yaml` (symlinked → firesim `deploy/config_build_recipes.yaml`):

```
alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3:
    PLATFORM: xilinx_alveo_u250
    DESIGN: FireSim
    TARGET_CONFIG: RoseTLRocketMMIOOnlyConfig   # the metasim-validated config
    PLATFORM_CONFIG: BaseXilinxAlveoU250Config
    platform_config_args: { fpga_frequency: 60, build_strategy: TIMING }
    bit_builder_recipe: bit-builder-recipes/xilinx_alveo_u250.yaml
```

This is the "simple" target: single Rocket core + RoSE bridge, no NIC/gemmini.

**Config changes.**
- `soc/sim/config/config_build_local.yaml` (→ `deploy/config_build.yaml`):
  `builds_to_run: [ alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3 ]`
  (was the commented-out `alveo-u250_firesim-boom-dual-dma-...`, which references a
  non-elaborating config).
- Build farm: `externally_provisioned`, `build_farm_hosts: [localhost]`,
  `default_build_dir: /scratch/dima/rose-infra/RoSE/soc/sim/bitstreams/`.

**Toolchain.** Vivado 2023.1 at `/ecad/tools/xilinx/Vivado/2023.1`. Launcher
`soc/sim/run_buildbitstream.sh` sources chipyard `env.sh`, puts Vivado on `PATH`
(`settings64.sh`), sources `sourceme-manager.sh --skip-ssh-setup`, then runs
`firesim buildbitstream`. NOTE: use `set -eo pipefail` (NOT `-u`) — the conda
`activate-riscv-tools.sh` references unbound `$RISCV` and aborts under `set -u`.

**Run.** Launched detached (`nohup`), log at `soc/sim/buildbitstream.log`. Manager
confirmed: `Building Verilog for xilinx_alveo_u250-firesim-FireSim-RoseTLRocketMMIOOnlyConfig-BaseXilinxAlveoU250Config`
→ `make ... replace-rtl` (the same elaboration + GoldenGate path the metasim used) →
Vivado synthesis/place-and-route.

**Result ✅ — BITSTREAM BUILT (2026-06-26).** Full flow completed with **0 errors**:
- Verilog generation (Scala elaboration + GoldenGate) — passed (same path as metasim).
- FPGA C++ driver compiled clean (only harmless sign-compare warnings in the deprecated
  `airsim.cc`).
- Vivado 2023.1: `synth_design` → `phys_opt_design completed successfully` →
  `write_bitstream completed successfully` → `Bitgen Completed Successfully` →
  `write_cfgmem completed successfully`. `0 Infos, 0 Warnings, 0 Critical Warnings and 0 Errors`.
- Artifacts (`firesim.bit`, `firesim.mcs`, `firesim.tar.gz` ~25MB) under
  `soc/sim/bitstreams/platforms/xilinx_alveo_u250/cl_…-RoseTLRocketMMIOOnlyConfig-BaseXilinxAlveoU250Config/`
  and packaged tar under `…/deploy/results-build/2026-06-27--01-08-07-…/`.

**hwdb entry** (added to `soc/sim/config/config_hwdb_local.yaml` → `deploy/config_hwdb.yaml`):
```
alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3:
    bitstream_tar: file://…/results-build/2026-06-27--01-08-07-…/…/firesim.tar.gz
    deploy_quintuplet_override: null
    custom_runtime_config: null
```

**To deploy on the U250 board:** set `config_runtime.yaml` `default_hw_config:` to
`alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3`, then
`firesim infrasetup` → `firesim runworkload` (with the gym synchronizer serving the RoSE
bridge, as in the metasim flow).
</content>

## STEP 16 — Bump chipyard 1.13.0 → main (1.14.0) ✅

Bumped the whole stack from chipyard 1.13.0 to **main / 1.14.0** (pinned to tip
`0acc1e1d`, = 1.14.0 + 1 firesim local-build hotpatch).

### Version delta
| | 1.13.0 (was) | 1.14.0 (now) |
|---|---|---|
| chipyard | `69eba860` | `0acc1e1d` |
| Chisel / Scala | 6.5.0 / 2.13.12 | **6.7.0 / 2.13.16** |
| FireSim (`sims/firesim`) | `141bff735` (1.20.1-51) | `da2a1cbc` (**1.21.0**) |
| rocket-chip | `72690b07` | `55bcad0f` |
| testchipip | `c94c1e3f` | `26f821be` |
| spike (riscv-isa-sim) | (1.13 era) | `9c190a07` (rebuilt) |

Revert point: chipyard pin `69eba860a352343e4ac6b6df0f3638a79a86ec78`.

### How the bump was done
1. **De-inject** RoSE from the chipyard tree (so the submodule checkout is clean):
   `git checkout -- build.sbt DigitalTop/IOBinders/Ports/BridgeBinders` (restore the
   RoSE-patched/symlinked tracked files), clean firesim's stray `.out` files. The
   `.conda-env` is gitignored so it survives the checkout.
2. `git checkout 0acc1e1d` in `soc/sim/chipyard`, then
   `./scripts/init-submodules-no-riscv-tools-nolog.sh` to sync the minimal submodule set
   to the 1.14.0 pins.
3. Re-run `soc/setup.sh` (re-symlinks + re-patches build.sbt).

### Breakages & fixes (the actual migration work)
- **build.sbt restructured.** 1.14.0 builds the `chipyard` project in a block from a
  `baseProjects: Seq[ProjectReference]` (no more flat `.dependsOn(...gemmini, icenet,…)`).
  setup.sh's old `sed` target was gone. **Fix:** updated `soc/setup.sh` step-2 to splice
  `rose, firechip_bridgeinterfaces,` into the `baseProjects` Seq after the stable
  `constellation, barf, shuttle, rerocc,` line (with a pre-1.14.0 fallback to the old sed).
- **Stale/uninitialized optional generators.** 1.14.0's build.sbt auto-discovers any
  generator with a `.git` dir and compiles it. The optional generators left initialized by
  the 1.13.0 setup (cva6/ibex/sodor/…) were at stale pins and failed against the bumped
  rocket-chip (`not found: value Annotated`, `BootROMLocated` now returns `Seq` not
  `Option`). **Fix:** `git submodule update` them to the 1.14.0 pins. Also had to
  `--init` **radiance** (+ tacit): stock firechip `TargetConfigs.scala` hard-references
  `chipyard.RadianceClusterSynConfig`, which only exists when radiance is initialized.
- **RoSE's 4 forked chipyard files were stale.** RoSE overrides DigitalTop.scala,
  iobinders/IOBinders.scala, iobinders/Ports.scala, firechip/.../BridgeBinders.scala via
  symlink — these were forks of the **1.13.0** versions and were missing 1.14.0 additions
  (`WithCTCIOCells`, the new example widgets, `testchipip.cosim.SpikeCosimConfig`'s new
  `mems` param, moved `CanHavePeriphery*` traits, etc.), which broke stock 1.14.0
  `HarnessBinders`/`AbstractConfig`. **Fix: re-forked** — re-seeded each `soc/src/main/scala`
  file from the **1.14.0 stock** version and re-applied RoSE's small additive edits:
  - DigitalTop: `with rose.CanHavePeripheryRoseAdapter`
  - Ports: `firechip.bridgeinterfaces` import + `case class RoseAdapterPort`
  - BridgeBinders: import + `class WithRoseBridge extends HarnessBinder(...)`
  - IOBinders: `rose.{CanHavePeripheryRoseAdapter, RoseAdapterKey}` +
    `firechip.bridgeinterfaces.{RosePortIO, RoseAdapterParams}` imports +
    `class WithRoseIOPunchthrough extends OverrideIOBinder(...)`
  RoSE's own generator/bridge Scala needed **no changes** — it compiled + elaborated as-is.
- **Conda env: NOT rebuilt.** The existing 1.13.0 conda env compiled the whole 1.14.0
  stack (Chisel 6.7 is fetched by sbt via firtool-resolver; jdk/sbt/verilator unchanged).
- **Spike (riscv-isa-sim) rebuilt.** testchipip 1.14.0's `cospike_impl.cc` (always in the
  FireSim driver `DRIVER_CC`) needs the newer spike API (`sim_t` ctor signature,
  `read_override_device_t::size()`); the conda env's 1.13-era `libriscv.so` failed to
  compile it. **Fix:** `git submodule update --init toolchains/riscv-tools/riscv-isa-sim`
  to the 1.14.0 pin, then rebuilt + installed just spike into `$RISCV`
  (`soc/sim/build_spike.sh`, replicating the spike steps of `build-toolchain-extra.sh`).

### Result — full pipeline + co-sim revalidated on 1.14.0 ✅
- `make verilog` (Scala compile → chisel elaboration → GoldenGate → firtool): clean.
  GoldenGate memory map shows `[100,17f]: RoSEBridgeModule_0`; `FireSim-generated.sv`
  (13 MB) contains `module FireSim` + 259 `RoseAdapter` refs.
- `make verilator` → **`VFireSim`** (50 MB metasim) built.
- Co-sim revalidated against the real gym synchronizer (PatternEnv, manager-free
  `run_sync_only.py`):
  ```
  DMAVALIDATE: PASS rx[0]=0xc0de0000 rx[15]=0xc0de000f      # channel 0 -> CamDMAEngine -> mem 0x88000000
  RXVALIDATE:  PASS pattern at offset 2 (buf=0x00000011,0x00000040,0xc0de...)  # channel 2 -> reqrsp1 MMIO
  ```
  Both RX paths (DMA + reqrsp) deliver the served pattern to the SoC, clean exit.

### Helper scripts added (soc/sim/)
`compile_check.sh` (make verilog), `build_metasim.sh` (make verilator + SYNCASYNCNET
waiver), `build_spike.sh` (rebuild libriscv), `build_dmatest.sh` / run_dmatest.sh /
run_rxtest.sh (compile + run the validation baremetal tests).

### Submodule housekeeping
Added `soc/sw/xpu-rt` = `github.com/ucb-bar/XPU-RT` (carries `zephyr-chipyard-sw` nested).

## STEP 17 — 1.14.0 U250 bitstream rebuild (+ local-build SSH-key gotcha)

Rebuilding the U250 bitstream on the migrated chipyard 1.14.0 stack (same recipe
`alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3`).

**Gotcha — `firesim buildbitstream` on a LOCAL build farm needs `~/firesim.pem`.**
`firesim buildbitstream` runs `replace_rtl` on the build-farm host (localhost) via
fabric/paramiko, which authenticates with `env.key_filename = ~/firesim.pem` (the FireSim
convention). If that file is missing the build dies immediately with a misleading
`fabric.exceptions.NetworkError: Low level socket error connecting to host localhost on
port 22: No such file or directory` — the real error (further up the log) is
`FileNotFoundError: '/home/dima/firesim.pem'` from paramiko's `_key_from_filepath`. Manual
`ssh localhost` works (the CLI uses `~/.ssh/id_rsa`), which makes this misleading; and it is
NOT an ssh-agent problem. **Fix:** `ln -sf ~/.ssh/id_rsa ~/firesim.pem` (localhost's
`authorized_keys` already accepts `id_rsa.pub`). `run_buildbitstream.sh` now creates this
symlink if missing. (The 1.13.0 build worked because `~/firesim.pem` existed at the time.)

Verified after the fix: `[localhost] out:` command output streams over SSH, Verilog
generation + FPGA driver compile proceed, then Vivado synth/P&R/bitgen. Log:
`soc/sim/buildbitstream_1140c.log`.

**Result ✅ — 1.14.0 BITSTREAM BUILT (2026-06-30).** Full Vivado flow completed with
**0 Warnings, 0 Critical Warnings, 0 Errors**: `Your bitstream has been created!`,
`firesim.bit` + `firesim.mcs` + `firesim.tar.gz` (25 MB) under
`.../deploy/results-build/2026-07-01--01-45-06-…/`. Updated the hwdb entry
`alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3` in
`soc/sim/config/config_hwdb_local.yaml` to point at the new 1.14.0 tar. Deploy with
`firesim infrasetup` → `firesim runworkload` (gym synchronizer serving the RoSE bridge).

## STEP 18 — On-FPGA test of the 1.14.0 RoSE port (U250) ✅

Ran the RoSE DMA co-sim loop on the actual Xilinx Alveo U250 with the 1.14.0 bitstream.

**FPGA host state (already provisioned):** U250 present (`42:00.0`, FireSim shell dev `0x903f`),
`xdma`+`xvsec` kernel modules loaded, `/dev/xdma0_*` runtime nodes, `xbutil` at `/usr/bin`.

**Runtime config (`soc/sim/config/config_runtime_local.yaml`):**
- `metasimulation_enabled: false`; run farm `externally_provisioned` / `localhost: one_fpga_spec`,
  `XilinxAlveoU250InstanceDeployManager`.
- `default_hw_config: alveo_u250_firesim-rocket-singlecore-with-rose-fast-no-nic-l2-llc4mb-ddr3`
  (the 1.14.0 hwdb entry / bitstream).
- `workload_name: rose-dmavalidate.json`.
- Synchronizer: PatternEnv-v0, route `0x11 -> ch0` (DMA), `firesim_step: 1_000_000` (FPGA:
  larger step = fewer host round-trips; `config_deploy_gym.yaml`).

**Workload (`deploy/workloads/rose-dmavalidate.json` + `rose-dmavalidate/`):** baremetal
`dmavalidate.riscv` + `dummy.rootfs`. GOTCHA: FireSim builds the source path by *string
concatenation* `workload_input_base_dir + path` where the base is `workloads/<benchmark_name>/`
— so `common_bootbinary`/`common_rootfs` must be **relative to that benchmark dir** (basenames
here), NOT absolute (an absolute path yields `workloads/<name>//scratch/...`).

**Flow (`soc/sim/run_fpga.sh {infrasetup|runworkload|kill}`):**
1. `firesim infrasetup` — builds the FPGA driver, stages libs, unloads XDMA, **JTAG-flashes the
   U250 with `firesim.bit`**, reloads XDMA, sets slot permissions. `EXIT=0`.
   (Same `~/firesim.pem` requirement as buildbitstream — the launcher ensures it.)
2. `run_sync_only.py` (venv) — gym synchronizer listening on :10001.
3. `firesim runworkload` — loads `dmavalidate.riscv` via TSI, runs on the FPGA.

**Result ✅ — on-hardware:** uartlog (`firesim_run_temp/sim_slot_0/uartlog`) shows the RoSE
co-sim loop live on the FPGA: `[RoSE Bridge]: pushing header to 0x11 and channel to 0`,
`[ROSE DRIVER]: Pushing cmd 11`, bridge connected to the synchronizer, then:
```
DMAVALIDATE: PASS rx[0]=0xc0de0000 ...
```
i.e. the Rocket SoC (booted from the 1.14.0 bitstream) requested a camera frame over the RoSE
bridge, the synchronizer served the pattern, the CamDMAEngine wrote it to U250 DDR @0x88000000,
and the SoC read + validated it — **the whole RoSE bridge works on real hardware.** (uartlog
flushes slowly under synchronizer gating; `PASS` was captured incrementally.) Torn down with
`firesim kill`.

### STEP 18b — Deadlock-detection bypass (the old firesim patch is gone; use +partitioned=1)

The pre-migration firesim FORK carried a patch to bypass FireSim's heartbeat deadlock
detection (so the RoSE bridge's intentional synchronizer-gated stalls aren't flagged as a
hung sim). That patch is NOT in the current stack — `sims/firesim` is now stock upstream
(1.21.0), so the heartbeat bridge deadlock check is active again:
`sim/midas/src/main/cc/bridges/heartbeat.cc` sets `has_timed_out` when the target cycle
doesn't advance between polls, and `heartbeat_t::terminate()` returns it → the sim exits with
`Simulator deadlock detected at target cycle N. Terminating.`

Stock firesim already exposes a disable: the `+partitioned=1` plusarg sets
`ignore_heartbeat=true` (verified `heartbeat.cc` is the ONLY consumer of `+partitioned` in the
driver C++, so no other side effects). Enable it via
`config_runtime_local.yaml` -> `target_config.plusarg_passthrough: "+partitioned=1"`.

**Validated on the U250 FPGA (2026-06-30):** the un-patched dmavalidate run self-terminated
with `Simulator deadlock detected at target cycle 931000002`. With `+partitioned=1` (confirmed
in the driver command line), the same run reached `DMAVALIDATE: PASS` and continued **past
cycle 965000002 with 0 deadlock messages, still running** — i.e. the deadlock detection is
bypassed. (Torn down with `firesim kill`.)

For a durable, checkout-reproducible bypass (option 2), add a `soc/setup.sh` patch to
heartbeat.{h,cc} instead of relying on the runtime plusarg.
