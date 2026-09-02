# TACIT instruction tracing on AWS F2 — verdict: capture works, the byte stream is corrupt

**Date:** 2026-08-28  **Branch:** rose-2-dev
**hw-config:** `f2_dual_small_norose_tacit_q31_60mhz` (`SatGemDualSmallTacitMMIOOnlyConfig`)
**Subject:** dronet int8 / RVV, isolated tree `modelblaster/examples/dronet_tacit` + `harness_tacit`

## Verdict

**The trace hardware is present and works. The trace *data* is not decodable.**
No basic-block attribution could be produced. Root cause is in the F2 bridge datapath,
**not** in the TACIT encoder, the sink, the bridge Scala, the host driver, or the decoder —
all five sources are byte-identical to the tree that produced a known-good U250 trace.

## 1. Capture works (positive result)

* Encoder MMIO is live and responds: guest read back `TR_TE_CTRL` `0x1 -> 0x3` (start) `-> 0x1`
  (stop) around `model_run_test()` — uartlog line `=== TACIT === hart=0 info=0x00000000
  ctrl0=0x00000001 ctrl1=0x00000003 ctrl2=0x00000001`.
* Bytes stream off-FPGA: `tacit0.out` / `tacit1.out` appear in
  `/home/ubuntu/sim_slot_0/` on the run host, one file per tile.
* The model itself ran correctly in the traced build: `max_abs_err=0`, full
  `MODELBLASTER_PROFILE` table, 7,539,118 summed op cycles.

### `fq` does NOT collect the trace
`deploy/fpga_queue/fq/backends/firesim.py` hardcodes
`"common_simulation_outputs": ["uartlog"]`, so `tacit*.out` is never copied back.
It must be `scp`'d from `<simulation_dir>/sim_slot_0/` on the run host that owned the lane.
Helper used here: `experiments/tacit/f2_tacit_run.sh` pattern (clear stale traces on every
run host first — `sim_slot_*/` survives between jobs).

## 2. The stream is corrupt (negative result)

### 2a. 4-byte zero blocks are injected into the byte stream

| trace | bytes | zero bytes | longest zero run |
|---|---|---|---|
| U250 reference (`firesim_tacit0.out`, 3 MB head) | 3,000,000 | **3** (0.00%) | 2 |
| F2 run 1 — mid-run enable, dronet model region | 59,520 | 473 (0.79%) | 48 |
| F2 run 2 — reset enable, boot (3 MB head) | 3,000,000 | 2,963,186 (**98.77%**) | 2,963,184 |
| F2 run 3 — reset enable, no console before model | 25,317,376 | **25,317,376 (100%)** | whole file |

Zero-run length histogram for F2 run 1: **92 runs of exactly 4 bytes**, plus one each of
8, 16, 32 and 48 — i.e. quantised 32-bit lane-fill, not random loss.

Run 3 is the degenerate case: tile 0's file is 25 MB of *pure* zeros (`tr -d '\000' | wc -c`
= 0) while tile 1's file on the same run begins with a real sync packet. The bridge streams
zero-filled beats instead of gating on valid.

**This corruption is silent.** `0x00` is a legal 1-byte TACIT packet (compressed
taken-branch, timestamp delta 0), so a decoder does not fault on it — it just tracks the
wrong PC. Any framing/health metric that only counts parse errors will call a 100%-zero
file "clean". Do not trust one.

### 2b. The sync packet's address is truncated — reproducibly

TACIT's sync packet carries the **only** absolute PC in the whole trace; every later packet
is a delta. It is emitted as a LEB128-style varint of `iaddr >> 1` (LSB-first 7-bit groups,
terminator = MSB set).

| run | true PC (from the ELF) | expected bytes | **observed bytes** | decoded PC |
|---|---|---|---|---|
| U250 ref | `0x80000202` | `01 02 00 00 84` | `01 02 00 00 84` | `0x80000202` ✅ |
| F2 run 1 | `0x8000056a` (`sw a5,0(s0)` = the `l_trace_encoder_start` store in `main`) | `35 05 00 00 84` | `35 05 00` | `0x1ec0056a` ❌ |
| F2 run 2 | `0x800002f2` (insn after reset.S's enabling `sw`) | `79 02 00 00 84` | `79 02 00` | `0x170002f2` ❌ |
| F2 run 3 (tile 1) | `0x800002f6` (`j boot_secondary_core`) | `7b 02 00 00 84` | `7b 02 00` | — ❌ |

Exactly **3 of the 5 address bytes arrive**, in three runs, on two harts, two hosts and two
different ELFs. The low 21 bits are always provably correct (they match the exact
instruction in the ELF); the top two groups — including the terminator — are gone, so the
decoder runs the varint on into the following packets and produces a garbage PC.

### 2c. Consequence: the decode cannot track the PC

Even after pinning the correct start PC (`TACIT_START_PC`, added to the decoder), the decode
dies within 1–2 packets in both runs, always at a `jalr`, where the stream supplies a 1-byte
compressed branch packet instead of the required multi-byte `FUj` (uninferable-jump) packet
that carries the target. Only 7 bytes of value `0x0a` (the `FUj` header) exist in the whole
59.5 KB model trace, which must contain at least ~21 kernel-call returns.

### 2d. Integrity check that the U250 reference passes and F2 fails

TACIT timestamps are cycle deltas, so summing them must equal the traced window's cycle count:

* **U250 reference: 216,597,098 vs 216,597,443 expected = 100.0%, zero framing breaks across
  72,173,559 packets.** The pipeline is sound.
* F2 run 1: 46 unrecoverable framing breaks in 59.5 KB (one per 1.3 KB) and a nonsensical
  timestamp sum.

## 3. What was ruled out

* **Decoder** — same binary decodes the U250 trace correctly (`--header-only` → `0x80000202`,
  ts 346; full decode → 100.0% timestamp coverage).
* **TACIT encoder / sink / bridge / host driver source** — `TacitEncoder.scala`,
  `TraceSinkRawByte.scala`, `TacitModule.scala`, `TacitBridge.scala` and `tacit.cc` are
  byte-identical (md5) between garden's tree and the AWS `~/chipyard-rose` tree that built
  the F2 bitstream. rocket-chip's `trace/*.scala` and `RocketCore.scala` are identical too,
  despite the two trees pinning different rocket-chip commits (`7002dd1db` vs `55bcad0f5`)
  and different `tacit` branches (`rose-tacit-bridge` vs `modular`).
* **Branch-predictor mode** — decoding with `TACIT_BR_MODE=0|1|2` changes nothing.
* **Guest enable sequence** — tried mid-run enable from C (run 1), enable at reset via
  `CONFIG_STARTUP_TACIT=y` (runs 2, 3), and with all console output moved after the traced
  region (run 3). Same truncation in every case.
* **Byte drops at the start of the stream** — sweeping the resync offset 4..21 fails at the
  same absolute byte in every case.

## 4. Secondary finding: trace bandwidth throttles the simulation

The RoSE `TraceSinkRawByte` egress is **1 lane, 1 byte per beat**
(`bits = Vec(1, UInt(8.W))`; the upstream sink it was ported from was 4-lane). Observed
off-FPGA throughput was ~50 KB/s. With `CONFIG_STARTUP_TACIT=y` the guest never reached
`main` in 15 minutes because tracing throttles the target clock, which starves the HTIF
console, whose `htif_wait_for_ready()` spin retires more branches, which produces more trace
— a positive feedback loop. Mitigation applied in `harness_tacit`: `CONFIG_BOOT_BANNER=n`
and the harness banner moved to *after* the traced region. Anyone reviving this path should
widen the sink egress first.

## 5. What to fix, in order

1. **Gate the bridge on `valid`/`mask`.** `TraceSinkRawByte` drives
   `out.mask(0) := io.trace_in.valid` and `out.valid := io.trace_in.valid`; check that the F2
   `TacitBridge`/`StreamToHostCPU` path honours them instead of streaming zero-filled beats.
   The 4-byte quantisation says 32-bit lanes.
2. Re-run the two checks in this document — they are cheap and decisive:
   `experiments/tacit/tsplit.py` (framing + zero-run histogram) and the timestamp-sum
   integrity check (must be ~100% of the known cycle count).
3. Only then widen the sink egress (finding 4) so a real kernel can be traced without
   throttling the simulation.

## 6. Reusable pieces produced here

* `experiments/tacit/vbb_analyze.py` — `trace.vbb.csv` → per-basic-block cycle attribution
  grouped by function, with `nm`/`addr2line` source mapping and the `netvar` stall term.
  Untested against real data (no decodable trace existed); ready for when one does.
* `experiments/tacit/tsplit.py` — standalone TACIT packet splitter / stream-health checker.
* `experiments/tacit/xcheck.py` — cross-checks a trace against an ELF's actual control flow
  (objdump-derived), reporting how many packets match before the first contradiction.
* `experiments/tacit/decoder_config_template.json` — decoder config with `to_vbb` enabled.
* `tools/tacit-decoder` additions: `TACIT_START_PC` (pin the start PC when the sync packet is
  untrustworthy), `TACIT_BR_MODE` (override the hardcoded `BrTarget` for HW traces), and
  actionable `[MISS]` diagnostics replacing two bare `unwrap()` panics.
* `modelblaster/harness_tacit/` + `modelblaster/examples/dronet_tacit/` — isolated TACIT
  harness and model tree; shares nothing writable with the shared example trees.

---

## UPDATE 2026-09-02 — root cause identified upstream, fix exists

`firesim/firesim` branch **`f2-pcim-host-dma`** (2026-08-30/31) contains what looks
like the fix for both symptoms above. Two commits, and each matches a measurement
in this document from the other end:

**`3bcdb00f4` "midas: DMA F2 bridge streams into host memory over PCIM"**
> The F2 Small Shell has no DMA engine and the XDMA shell is unsupported
> (aws-fpga ERRATA), so simif_f2 drains FPGA-to-CPU streams with a loop of
> **4-byte `fpga_pci_peek` calls over BAR4**. Each is a non-posted PCIe read the
> core stalls on, and they cannot pipeline, which caps that path at
> **4 bytes per round trip.**

That is section 2a from the other side. We measured "**92 runs of exactly 4 bytes**
... quantised 32-bit lane-fill" and ~50 KB/s off-FPGA throughput; the drain is a
4-byte-at-a-time BAR4 peek loop that returns zeros rather than gating on valid.
The 4-byte quantum was never a TACIT property at all.

**`cb0f400ab` "midas: clamp FPGA-managed pull() to the destination buffer size"**
> `pull()` read how many bytes the FPGA had queued and copied all of them into
> dest, which the caller sized at `num_bytes`. Nothing bounded the copy by that
> size... The `assert(num_bytes >= required_bytes)` at the top looks like it
> guards this but does not: it compares the two arguments to each other, not the
> amount copied against the buffer it is copied into.

**Both are present in our tree** (firesim `15d6ffc41`, detached from `da2a1cbce`):
* `sim/midas/src/main/cc/bridges/fpga_managed_stream.cc:31` is that exact
  non-guarding assert, followed by an unbounded `memcpy` of `bytes_in_buffer`.
* `sim/midas/src/main/cc/simif_f2.cc:208` — `/* rh: attach to BAR4 (for now to do
  a PCIS cuz no XDMA) */`.

This also explains section 4 (trace bandwidth throttles the simulation) without
needing the 1-lane `TraceSinkRawByte` to be the cause: 4 bytes per non-posted
PCIe round trip *is* ~50 KB/s.

### What adopting it costs
Our firesim is ~3 weeks behind the branch, and the change touches
`CompilerConfigs.scala` / `Config.scala`, so it is **a bitstream rebuild**, not a
driver-only bump. Worth it — it unblocks TACIT on F2 permanently, and TACIT is
the right instrument for per-component kernel attribution (the alternative,
`-DMB_GEM_PHASE_TRACE` rdcycle brackets in
`kernels/gemmini/gemmini_conv2d_s8_gemmini_tiled_conv.c`, needs a source change
per kernel and only sees phases you thought to bracket).

**Not yet verified** — no F2 trace has been captured with this branch. The
symptom/commit correspondence is strong but circumstantial until one is.
