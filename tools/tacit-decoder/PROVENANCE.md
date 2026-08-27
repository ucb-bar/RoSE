# tacit-decoder — provenance & RoSE patches

This is a **vendored, patched** copy of the TACIT instruction-trace decoder.

- **Upstream**: `https://github.com/ucb-bar/tacit_decoder.git`
- **Base branch/commit**: `dev` @ `c56be947734a5ea3cd0fda8d707d7a4cc9f7299c`
  (the fullest backend: `perfetto_receiver` + Zephyr `StackUnwinder` + `$d`/`.L` symbol
  filtering + padded-address unwinder fix). Upstream `dev` could NOT parse our FireSim/FPGA
  RTL-encoder traces; `main` could parse them but has no perfetto/Zephyr backend.

## RoSE patches (make one decoder handle BOTH Spike and FPGA/RTL traces)

TACIT packets differ between the software (Spike) encoder and the RTL `TacitEncoder` on the
FireSim FPGA. This copy **auto-detects** the format from the sync-start header and dispatches:

| field | Spike (`0x36` sync byte, `sync_type=SyncStart`) | HW/RTL (`0x16`, `sync_type=SyncNone`) |
|-------|--------------------------------------|----------------------------|
| sync/trap `prv` byte, `target_ctx`, `runtime_cfg` | present | **absent** |
| sync packet body | `[hdr][prv][ctx][target][ts]` | `[hdr][brmode][target][ts]` |
| trap `from` address | `>>1`-compressed (needs `refund_addr`) | **absolute** (raw) |

Files changed vs upstream `dev` (see `rose-patches/*.patch`):

1. **`src/frontend/packet.rs`** — added `enum TraceFormat { Spike, Hw }`; `read_first_packet`
   detects the format from the first byte's `sync_type` and returns it; `PacketReader.format`
   selects the Spike-vs-HW body layout for every `FSync`/`FTrap` packet. Both layouts are kept
   verbatim from their known-good sources (Spike = upstream `dev`, HW = `main`-compatible).
2. **`src/common/insn_index.rs`** — the machine/SBI instruction map was hard-coded to a section
   literally named `.text` disassembled at the ELF entry point. A Zephyr unikernel names it
   `text` (no dot), boots at `rom_start` (`0x80000000`) ≠ `text` (`0x8000015c`), and spreads
   code across 7 executable sections (`reset`/`exceptions`/`rom_start`/ISR tables). Now it
   iterates **all** executable sections keyed by `section.address()` (mirrors the app-space path).
3. **`src/frontend/decoder.rs`** — the trap "from" address refund is now format-aware (raw for
   HW, `refund_addr` for Spike).

## Validation (2026-08-27)

- **FPGA/HW**: full `firesim_tacit0.out` (72 MB, Saturn+RoSE+TACIT bitstream, Zephyr FOC guest)
  → valid `trace.perfetto.json`: 4762 events / 2381 call slices, 56 resolved Zephyr functions,
  ts 346→216,597,443, max stack depth 14 (`main>FOC_update>FOC_invClarkSVPWM`+trig).
- **Spike**: `--header-only` on a Spike ExecuTorch/ModelBlaster trace auto-detects `Spike`,
  decodes the first packet to `0x8077b464` / `PrvMachine` — byte-for-byte matching the first
  line of that run's reference `tacit.debug`.

See `docs/ROSE_TACIT_TRACING.md` for the end-to-end setup / build / decode guide.
