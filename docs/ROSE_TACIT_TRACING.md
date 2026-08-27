# RoSE TACIT Instruction Tracing — end-to-end guide

**TACIT** = *Timing-Accurate Core Instruction Trace* (Chipyard's RISC-V instruction-trace
encoder/format/decoder, `github.com/riscv-tacit`). This guide covers the full pipeline on RoSE:
**build a traced guest → generate a trace on the FireSim FPGA → extract it → decode it into a
Perfetto flamegraph** (or plain text / speedscope / stats). It also shows how the *same* decoder
handles Spike (software) traces.

> TL;DR — everything is committed. Guest: `soc/sw/xpu-rt/.../samples/tacit`. FPGA bridge +
> bitstream: `RoseTLRocketSaturnTacitMMIOOnlyConfig`. Host extractor: `soc/src/main/cc/tacit.cc`.
> Decoder: `tools/tacit-decoder` (auto-detects FPGA **and** Spike trace formats). Config
> template: `tools/tacit-decoder/configs/rose_zephyr_fpga.json`.

---

## 0. Pipeline at a glance

```
 ┌─────────────┐   TraceCoreInterface    ┌──────────────┐   encoded packets   ┌───────────────┐
 │ Rocket core │ ──────────────────────▶ │ TacitEncoder │ ──────────────────▶ │ TraceSinkRaw  │
 │ (in RTL)    │   (retired-insn stream) │ (rocket-chip)│                     │ Byte, id 0    │
 └─────────────┘                         └──────────────┘                     └──────┬────────┘
        guest calls l_trace_encoder_start/stop, configure_target(TARGET_PRINT=0)     │ StreamToHostCPU
                                                                                     ▼
 ┌────────────────────┐   trace.perfetto.json   ┌───────────────┐   tacit0.out   ┌───────────────┐
 │ ui.perfetto.dev /  │ ◀────────────────────── │ tacit-decoder │ ◀───────────── │  TacitBridge  │
 │ speedscope / txt   │   (ELF-symbol-resolved) │ (tools/)      │  (72 MB, raw)  │  host tacit.cc│
 └────────────────────┘                         └───────────────┘                └───────────────┘
```

Key point vs the older DMA-sink approach: the streaming **`TacitBridge`** pushes packets straight
off-FPGA to the host `tacit.cc` driver (`tacit0.out`), so there is **no** "dump a DRAM region"
problem — extraction is automatic.

---

## 1. Prerequisites

- A working RoSE FireSim FPGA flow (see `docs/FPGA_GATENAV_REPRODUCE.md` for the U250 bring-up,
  `firesim infrasetup`/`runworkload`, the shared queue, and the "reprogram before every
  runworkload" rule).
- Rust toolchain (`cargo`) to build the decoder.
- The Zephyr in-tree build env for the guest (see `docs/` zephyr notes: `activate_conda` +
  `set_envvars_sdk`).

---

## 2. Build the traced guest binary

The sample lives at `soc/sw/xpu-rt/zephyr-chipyard-sw/samples/tacit/` (tracked in the `xpu-rt`
submodule). It is a hello-world + FOC (field-oriented-control) loop bracketed by the TACIT
encoder start/stop calls.

**`prj.conf`** — the one switch that matters:
```
CONFIG_STARTUP_TACIT=y     # trace from boot (→ ~216M instrs). Set =n to trace only the
                           # l_trace_encoder_start/stop bracket around the FOC loop.
```

**`src/main.c`** — uses `configure_target(encoder, TARGET_PRINT)`. `TARGET_PRINT` is targetId `0`,
and our config puts `WithTraceSinkRawByte(0)` at targetId 0 — so `TARGET_PRINT` routes to the
streaming bridge with **no guest edit**. The FOC loop is wrapped by
`l_trace_encoder_start(encoder)` … `l_trace_encoder_stop(encoder)`.

> **Known build fix**: the FOC math needs `#include <math.h>` in `src/main.c` (otherwise
> `error: implicit declaration of function 'sqrtf'`). This is already applied; documented in
> the sample's `TACIT_TRACING.md`.

Build it like any spike_riscv64 Zephyr sample (in-tree env), producing `build/zephyr/zephyr.elf`.
Keep that `zephyr.elf` — the decoder needs it for symbol resolution.

---

## 3. Build the TACIT bitstream (once)

The TACIT streaming bridge is already ported into our tree and composed into a RoSE config.

- **Config**: `RoseTLRocketSaturnTacitMMIOOnlyConfig` (Rocket + Saturn + RoSE co-sim bridge +
  TACIT bridge). Elaborates to ~22 MB Verilog.
- **Recipe** (`soc/sim/config/config_build_recipes_local.yaml`):
  `alveo_u250_firesim-rocket-saturn-tacit-with-rose-fast-no-nic-l2-llc4mb-ddr3` (@30 MHz).
- **Submodule pins required** (a fresh checkout must have these + run `soc/setup.sh`, which
  symlinks the 6 firechip/soc TACIT sources):
  - `tacit` submodule @ branch `rose-tacit-bridge`
  - `rocket-chip` submodule @ branch `rose-tacit-encoder`
- Helper: `tools/tacit-decoder/scripts-rose/build_tacit_bitstream.sh`.

Build with the normal FireSim `buildbitstream` flow; the resulting bitstream is registered in
`soc/sim/config/config_hwdb_local.yaml` under the same recipe name. (A prebuilt bitstream tar is
already referenced there.)

---

## 4. Generate + extract a trace on the FPGA

- **Runtime config**: generate with `soc/scripts/gen_firesim_run_config.sh` (it is not committed —
  it holds environment-specific paths). For a local TACIT capture:
  ```bash
  soc/scripts/gen_firesim_run_config.sh --workload rose-tacit --name local_tacit \
      --outputs "uartlog tacit0.out" --run-host localhost \
      --hw-config alveo_u250_firesim-rocket-saturn-tacit-with-rose-fast-no-nic-l2-llc4mb-ddr3
  # -> config_runtime_local_tacit.yaml + workloads/rose-tacit.json  (note tacit0.out in outputs)
  ```
- **Capture script**: `tools/tacit-decoder/scripts-rose/run_tacit_capture.sh`

The TACIT guest is a standalone program (it does not need the Isaac co-sim to drive it), so the
RoSE bridge only needs **cycle grants** — the capture script starts a free-run sync (no FREEZE),
then `infrasetup` + `runworkload`. The guest prints `Hello World!`, runs the FOC loop emitting the
trace, then reboots (HTIF) → the sim ends → the trace is flushed.

The ported host driver `soc/src/main/cc/tacit.cc` (the `TacitBridge` endpoint) writes
**`tacit0.out`** into the sim run dir (`.../firesim_run_temp/sim_slot_0/`). For our FOC guest with
`CONFIG_STARTUP_TACIT=y` this is ~**72 MB** of encoded packets (~4.0 bits/instruction,
~216M instructions). Copy it somewhere stable; e.g. the archived reference is
`experiments/rose_arb_deadlock/traces/firesim_tacit0.out` (gitignored — it's large).

```bash
# from a machine with the local U250 programmed with the TACIT bitstream:
tools/tacit-decoder/scripts-rose/run_tacit_capture.sh
# -> copies sim_slot_0/tacit0.out to a scratch path and prints its size
```

---

## 5. Build the decoder

```bash
cd tools/tacit-decoder
cargo build --release        # ~15 s; binary at target/release/tacit-decoder
```

This is a **vendored, patched** copy of `ucb-bar/tacit_decoder` (`dev` @ `c56be94`). Our patches
make one binary handle **both** the FPGA/RTL format and the Spike format, and add Zephyr-unikernel
symbol handling — see `tools/tacit-decoder/PROVENANCE.md`.

---

## 6. Decode → Perfetto flamegraph

### 6.1 The config

Copy the template and fill in the two paths:

```bash
cp tools/tacit-decoder/configs/rose_zephyr_fpga.json /tmp/my_trace.json
# edit encoded_trace + BOTH binary paths to your tacit0.out and zephyr.elf
```

```jsonc
{
  "encoded_trace": "/PATH/TO/tacit0.out",
  "application_binary_asid_tuples": [ ["/PATH/TO/zephyr.elf", "0"] ],  // <- REQUIRED, asid "0"
  "sbi_binary": "/PATH/TO/zephyr.elf",                                 // <- same elf
  "to_perfetto": true
  // ... (rest of the toggles default to false)
}
```

> **Why the ELF goes in TWO places (the #1 gotcha).** The decoder selects its instruction/symbol
> map by `(privilege, ctx)`. Our FPGA packets carry no prv/ctx bytes, so the decoder defaults to
> `prv=User, ctx=0` for the whole run. It therefore reads the **user / asid-0** map — which is
> populated **only** from `application_binary_asid_tuples`. A Zephyr unikernel is one ELF that is
> both the "firmware" and the "application", so put it in `application_binary_asid_tuples` (asid
> `"0"`) *and* in `sbi_binary` (the latter is opened unconditionally). Passing only `sbi_binary`
> yields an **empty** map → `unwrap() on None` panic. `--application-binary` on the CLI is **not**
> wired to the tuples; use the config file.

### 6.2 Run it

```bash
cd tools/tacit-decoder
./target/release/tacit-decoder --config /tmp/my_trace.json
# writes trace.perfetto.json into the current directory
```

The frontend prints end-of-trace stats (`insn_count`, `bits per instruction`, packet counts).
`trace.perfetto.json` is standard Chrome/Perfetto Trace-Event JSON.

### 6.3 View it

Open **https://ui.perfetto.dev** → *Open trace file* → `trace.perfetto.json`. You'll see nested
function slices on a cycle-timestamp axis. Or drop it into speedscope, or use `"to_speedscope":
true` / `"to_txt": true` instead of perfetto.

**What a correct RoSE FOC decode looks like** (archived at
`experiments/rose_arb_deadlock/traces/firesim_tacit_zephyr.perfetto.json`): 4762 events / 2381 call
slices, 56 resolved Zephyr functions, ts 346→216,597,443, max stack depth 14. The resolved stacks
walk the whole unikernel lifecycle:
- **boot/init**: `z_cstart` → `z_prep_c` → `arch_bss_zero` → `soc_interrupt_init` → `plic_init`
- **FOC loop**: `main` → `FOC_update` → `FOC_invClarkSVPWM` + `sin`/`cos`/`__kernel_sin`/`__rem_pio2`
- **console**: `console_out` → `uart_htif_poll_out` → `htif_wait_for_ready`

---

## 7. The same decoder also reads Spike traces

The decoder **auto-detects** the trace format from the sync-start header (FPGA `sync_type=SyncNone`,
byte `0x16`; Spike `sync_type=SyncStart`, byte `0x36`) and dispatches the correct packet layout.
Nothing to configure — just point it at a Spike `tacit.out` with the matching ELF. Confirm the
detection with:

```bash
./target/release/tacit-decoder --sbi-binary /any.elf --encoded-trace tacit.out --header-only true
# prints e.g. "Detected trace format: Spike"  or  "Detected trace format: Hw"
```

(Spike traces carry real privilege/ctx, so for a Linux/multi-ASID Spike trace you populate
`sbi_binary` + `kernel_binary` + real ASIDs in `application_binary_asid_tuples`, exactly like the
upstream `configs/linux*.json`.)

---

## 8. Reference

### 8.1 Packet-format differences (why one decoder needs two paths)

| field | Spike (`0x36`, `SyncStart`) | HW/RTL FPGA (`0x16`, `SyncNone`) |
|-------|------------------------------|----------------------------------|
| sync/trap `prv` byte, `target_ctx`, `runtime_cfg` byte | present | **absent** |
| sync body | `[hdr][prv][ctx][target][ts]` | `[hdr][brmode][target][ts]` |
| trap `from` address | `>>1`-compressed (needs `refund_addr`) | **absolute** (raw) |

### 8.2 Decoder branches (historical)

- **`main`** (`d3d778f`): parses the FPGA format, but **no** perfetto / Zephyr backend.
- **`misc_decoders`**: Spike-patched; **mis-parses** the FPGA stream (panics, garbage target).
- **`dev`** (`c56be94`): fullest backend (perfetto + Zephyr unwinder) but FPGA-incompatible frontend.
- **`tools/tacit-decoder`** (this repo): `dev` + our patches = perfetto/Zephyr backend **and**
  auto-detecting FPGA+Spike frontend. **Use this one.**

### 8.3 Troubleshooting

| symptom | cause / fix |
|---------|-------------|
| `Error: No such file or directory (os error 2)` | `sbi_binary` is `""`; it is opened unconditionally. Set it to your ELF. |
| `No .text section found` | Zephyr names the section `text` (no dot). Fixed in this copy (iterates all exec sections). |
| `unwrap() on None` / `map_size=0` | ELF not in `application_binary_asid_tuples` with asid `"0"`. Add it (see §6.1). |
| `first packet must be FSync/CNa` | not a TACIT trace, or wrong offset — check `tacit0.out` isn't truncated. |
| trap assert `new_pc != trapping_pc` | format mis-detected/forced; let auto-detect run (don't hand-patch the format). |
| perfetto JSON huge / slow | expected for `CONFIG_STARTUP_TACIT=y` (216M instrs). Use `=n` to trace only the FOC bracket, or `to_stats`/`to_txt` for a lighter view. |

### 8.4 Artifacts & provenance

- Decoder + patches: `tools/tacit-decoder/` (`PROVENANCE.md`, `rose-patches/`).
- Reference perfetto trace: `experiments/rose_arb_deadlock/traces/firesim_tacit_zephyr.perfetto.json`.
- Reference encoded trace: `experiments/rose_arb_deadlock/traces/firesim_tacit0.out` (gitignored, large).
- Bridge port details: memory note `rose-tacit-firesim-bridge`; guest details:
  `samples/tacit/TACIT_TRACING.md`.
