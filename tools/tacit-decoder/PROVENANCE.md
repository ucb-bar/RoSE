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

## RoSE debug additions (2026-08-28, F2 bring-up)

Three additive, env-gated changes made while diagnosing the AWS-F2 trace path
(`experiments/tacit/F2_TACIT_VERDICT.md`). All are no-ops unless the env var is set; the
U250 reference trace still decodes byte-identically (`--header-only` -> `0x80000202`, ts 346).

1. **`TACIT_START_PC=<hex>`** (`src/frontend/decoder.rs`) — override the start PC taken from
   the sync packet. The sync packet carries the *only* absolute PC in a TACIT trace; if its
   address field is damaged, nothing downstream can recover, because every later packet is a
   delta. This lets you pin the true start PC (read off the ELF) and decode the rest.
2. **`TACIT_BR_MODE=0|1|2`** (`src/main.rs`) — override the branch mode. `read_first_packet()`
   hardcodes `BrTarget` for HW traces because the RTL sync packet carries no runtime_cfg byte;
   if the encoder was left in history/predict mode, every packet is misread.
3. **`[MISS]` diagnostics** (`src/frontend/decoder.rs`) — the two bare
   `insn_map.get(&pc).unwrap()` calls in `step_bb`/`step_bb_until` now print the failing PC,
   its neighbours, the in-block instruction count and the map size before exiting 42. Same
   failure, actionable output. (This is the hunk from
   `experiments/rose_arb_deadlock/traces/tacit_decoder_dev_zephyr_perfetto.patch`, kept.)

**Caveat learned the hard way:** `0x00` is a legal 1-byte TACIT packet (compressed
taken-branch, timestamp delta 0). A trace file that is entirely zeros therefore parses as an
endless run of valid packets. Never use "no parse errors" as a stream-health metric — use the
timestamp-sum integrity check instead (summed deltas must equal the traced window's cycle
count; `experiments/tacit/tsplit.py`).
