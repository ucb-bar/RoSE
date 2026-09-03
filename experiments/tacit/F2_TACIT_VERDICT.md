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

*(Superseded by the section below: it has now been captured, and the fix works.)*

---

# UPDATE 2026-09-02 — adopted, rebuilt, verified. **TACIT on F2 works.**

The heading of this document is now wrong for the PCIM bitstreams. On
`f2_dual_small_norose_tacit_q31_60mhz_pcim`, running the **byte-identical ELF**
that produced run 1 above (`md5 f5304dc0436741d2298068ac228ecfc0`), all three
section-2d checks pass and a full basic-block attribution — the thing this
document was written to say was impossible — was produced.

## 1. The result

| check | F2 BAR4 (run 1, recorded above) | **F2 PCIM (new)** | U250 reference |
|---|---|---|---|
| trace bytes, same workload | 59,520 | **951,488** (16.0x) | 72,180,864 |
| **(2d.1) sync-packet start PC** | `0x1ec0056a` ❌ | **`0x8000056a`** ✅ | `0x80000202` ✅ |
| **(2d.2) windowed timestamp sum** | 57,740,605 vs 7,543,000 = 765% ❌ | **7,543,572 vs 7,543,000 = 100.0%** ✅ | 216,597,098 vs 216,597,443 = 100.0% ✅ |
| **(2d.3) zero bytes** | 473 / 59,520 = **0.79%**, incl. **92 runs of exactly 4** | **13 / 951,488 = 0.00%**, longest run 2 | 8 / 72,180,864 = 0.00% |
| framing breaks | 1 resync + **47 runaway timestamps** | **0 and 0** over 947,372 packets | 0 and 0 |
| full decode | died within 1–2 packets | **951,475 / 951,488 bytes consumed, 5,664,864 insns, 1.3437 bits/insn** | (reference) |

The 4-byte zero-block injection is **gone**, not reduced: the run-length
histogram has no 4-byte entries at all, only five 2-byte and three 1-byte runs,
which is the same character as the known-good U250 trace.

`0x8000056a` is the `sw a5,0(s0)` that is `l_trace_encoder_start` in `main` —
the address section 2b said was provably correct in the low 21 bits and lost
above them. All five varint bytes now arrive.

### The attribution that could not be produced before

414 basic blocks, totalling **7,543,572 cycles**, against a guest that
independently reported `MODELBLASTER_WALL_CYCLES === 7543` (thousands) and a
per-op `rdcycle` profile summing to 7,539,118. Three independent counters
agreeing to 0.06%:

```
        cycles      count      mean  basic block
        851430      35679      23.9  0x80001d6a-0x80001da8
        643183       1000     643.2  0x80001a1a-0x80001aca
        629778      23266      27.1  0x80001c48-0x80001c9a
        454211      43359      10.5  0x80001d34-0x80001d48
```

The model itself was unaffected: `max_abs_err=0`, identical per-op profile,
`*** PASSED ***`.

### Section 4 (bandwidth throttling) also resolved

`FMR: 1.01`, effective target frequency **59.665 MHz** against a 60 MHz host
clock — i.e. tracing no longer throttles the simulation at all. Section 4
blamed the 1-lane `TraceSinkRawByte` egress; it was the 4-byte BAR4 drain.
**The sink does not need widening.**

## 2. Why `integrity.py` reports two timestamp numbers

The FSync packet's timestamp is **not** a delta within the traced window — it is
the encoder's free-running cycle count at the moment tracing was enabled. It is
invisible on a trace enabled at reset (the U250 reference's FSync ts is 346) and
dominant on one enabled from C (**68,162,537** here, i.e. Zephyr boot). Summing
it in inflates the window by 10x. `experiments/tacit/integrity.py` excludes it
and reports it separately; with that fix it reproduces this document's recorded
U250 number, 216,597,098, to the digit.

That number is also a free cross-check: the quad bitstream, a *different* FPGA
running the *same* ELF, reports **68,162,560** — 23 cycles apart.

## 3. The quad (`SatGemQuadHeteroTacitConfig`)

`f2_quad_hetero_norose_tacit_q31_60mhz_pcim` = **`agfi-010049831c489a814`**.

The PCIM datapath is proven on it: **four** hugepage stream buffers at four
distinct physical addresses, four `PCIM Peer Base Addr` programmed, and
**228,126,720 bytes** of trace delivered off tile 0 — against 59,520 bytes on
BAR4, and against run 3's 25 MB of *pure zeros*.

* zero bytes: **10 in 228,126,720 = 0.0000044%**
* framing breaks: **0**, runaway timestamps **0** (first 8 MB, 7,999,803 packets)
* start PC: **`0x8000056a`** ✅
* full decode of all 228 MB: **456,296,821 instructions, 228,126,710 / 228,126,720
  bytes consumed**

**Caveat, stated plainly:** check 2d.2 was *not* obtained on the quad. The ELF
used is the dual-built dronet, and the quad hetero's harts 0–1 are Rocket+Gemmini
with **no** Saturn vector unit, so the RVV kernels trap (`mcause: 2, Illegal
instruction` at `mepc 0x8000170c`) and fq kills the sim, leaving no clean
`Target Cycles Emulated` to compare against. That is an ELF/tile-mix mismatch,
not a bitstream defect — the stream itself is clean over 228 MB. Getting 2d.2
on the quad needs a `harness_tacit` build pinned to hart 2 or 3 (or a
Gemmini-dispatched model). Small, separate job.

## 4. What was adopted, and the one thing upstream gets wrong for us

Seven commits from `firesim/firesim` `f2-pcim-host-dma`, cherry-picked onto our
`15d6ffc41` (preserved, and still the base of branch `rose-f2-pcim`):
`2c9fcee30`, `b41eb50dc`, **`3bcdb00f4`** (the fix), `8164a9881`, `b006d39e2`,
`db7a301c8`, `cb0f400ab`; plus the `aws-fpga-firesim-f2` shell pointer
`80b34d3c2` → `5e4c3c2`, which carries `83d6a5c` "connect PCIM to the FireSim
FPGA-managed stream port" — without it the shim emits `io_pcim_*` and the CL
leaves them tied off, so the bitstream elaborates and never DMAs.

**Deliberately NOT taken: `d85e8c3cb`** (forward `--strategy` to the bitstream
build). Our recipes say `build_strategy: TIMING`, which F2 has always silently
dropped; forwarding it would change the synth directive relative to every AGFI
already in `config_hwdb.yaml`, putting a synthesis variable into an experiment
whose only intended variable is the stream transport. The shell bump still
contains the plumbing, but it is inert because `build-bitstream.sh` never passes
the flag.

### `3bcdb00f4`'s bus-master check must be made lazy

It calls `check_bus_master_enabled()` unconditionally in `fpga_setup()` and
`exit(1)`s if PCIe Bus Master Enable is clear. On the F2 run hosts it *is*
clear, and stays clear with an AGFI loaded:

```
AFI 0  agfi-0662319f4b07483a7  loaded ...   AFIDEVICE 0  0x1d0f 0xf002  0000:34:00.0
$ sudo setpci -s 0000:34:00.0 COMMAND
0002                    # memory space enabled, bus master CLEAR
```

Taken as written it aborts **every** F2 run from this tree, including every
CPU-managed-stream bitstream in `config_hwdb.yaml` that never masters the bus
and is healthy without BME. This was not hypothetical: fq job 478 was in
`INFRASETUP` against this tree during the adoption and its driver was rebuilt
from it. It ran clean only because the check had already been moved.

Fix: remember the app PF in `fpga_setup()`, run the check at the first
`allocate_to_cpu_buffer()` — fatal exactly where PCIM is relied on, a no-op
otherwise.

## 5. Running a PCIM bitstream: two host-side prerequisites

Neither is needed by the BAR4 path, and both fail silently-ish if missed.

1. **Hugepages.** One 2 MiB hugepage per to-host stream; the FPGA masters PCIM
   writes with *physical* addresses and only a hugepage guarantees contiguity.
   `HugePages_Total` is 0 by default → `sudo sysctl -w vm.nr_hugepages=64`.
2. **Bus Master Enable** → `sudo setpci -s <bdf> COMMAND=0x0006`. Programming
   the AFI rescans the app PFs and clears it again, and there is no seam between
   fq's `infrasetup` and its `runworkload`, so
   `experiments/tacit/f2_pcim_tacit_run.sh` holds the bit with a watcher for the
   duration. That script also **pins the lane** and applies both prerequisites
   to that lane's host only — the pool is shared, and reserving hugepages on or
   touching the PCI COMMAND register of someone else's running job is not ours
   to do.

Without BME the run does not error; the trace file is created and stays empty.

## 6. Registration and cost

NEW entries, `_pcim`-suffixed. **The pre-existing AGFIs were not repointed** —
verified by diffing `config_hwdb.yaml` before/after.

| hwdb entry | AGFI |
|---|---|
| `f2_quad_hetero_norose_tacit_q31_60mhz_pcim` | `agfi-010049831c489a814` |
| `f2_dual_small_norose_tacit_q31_60mhz_pcim`  | `agfi-0d79cabf8816f6517` |

Both built in one `firesim buildbitstream -b config_build_f2_pcim.yaml`,
03:13:33 → 07:53:17 UTC = **4 h 40 m**, exit 0, on two `z1d.2xlarge`
(`i-0dd223ef088621b6f`, `i-0a76a545f7811c54d`, tag `rosepcim`) — **both
terminated by buildbitstream**, ≈ **$7** of on-demand build-host time.

### Timing: the quad closed, the dual did not — read this before relying on it

| build | post-route WNS | baseline (`BaseF2Config`) |
|---|---|---|
| quad hetero PCIM | **+0.082 ns (MET)** | +0.019 ns, 84.63% CL LUT |
| dual small PCIM | **−0.045 ns (VIOLATED)** | +0.136 ns, 58.25% CL LUT |

The quad — the one that was expected to be tight — closed, and with more margin
than its own baseline. The dual has **one** violated path, and it is entirely
inside the AWS DDR4 shell IP:

```
Source:      WRAPPER/CL/SH_DDR/.../u_ddr4_mem_intfc/u_ddr_mc_pi/u_ddr_mc_write/...
Destination: WRAPPER/CL/SH_DDR/.../u_mig_ddr4_phy/.../RXTX_BITSLICE
Path Group:  pll_clk[2]_DIV          Data Path Delay: 2.965ns (route 97.3%)
```

Not the FireSim design clock, not the PCIM datapath, and the same shell block
that holds the *baseline* quad's worst path (+0.019 ns) — a route-dominated,
placement-luck path in this shell/Vivado 2025.2 combination. Every number in
section 1 was measured on this bitstream and it ran a full dronet inference to
`max_abs_err=0`, so the result stands. But **−45 ps is still a violation**: if
`..._dual_small_..._pcim` is going to be leaned on for real measurement rather
than this one A/B, rebuild it and take a run that closes.

---

# UPDATE 2026-09-03 — the quad's missing check (2d.2) obtained. Two guest bugs found first.

The section above closed with one stated gap: *"check 2d.2 was **not** obtained on the
quad … Getting 2d.2 on the quad needs a `harness_tacit` build pinned to hart 2 or 3.
Small, separate job."* It was not a small job, because **the pin it prescribes cannot
work as written** and, underneath that, `harness_tacit` could never boot a 4-hart
bitstream at all. Both are guest-side defects; neither is a bitstream or bridge fault.

## 1. The result — quad PCIM, full clean capture

`f2_quad_hetero_norose_tacit_q31_60mhz_pcim` (`agfi-010049831c489a814`), fq job 803,
lane f2-05, guest = `dronet_tacit` int8/RVV on `harness_tacit`, model pinned to hart 2.

| check | quad PCIM (new) | dual PCIM (recorded) | U250 reference |
|---|---|---|---|
| trace bytes, model tile | **1,016,192** | 951,488 | 72,180,864 |
| (2d.1) sync start PC | **`0x800002f6`** ✅ | `0x8000056a` ✅ | `0x80000202` ✅ |
| (2d.3) zero bytes | **14 / 1,016,192 = 0.00%**, longest run 2 | 0.00%, run 2 | 0.00%, run 2 |
| framing breaks / runaway ts | **0 / 0** over 1,011,965 packets | 0 / 0 | 0 / 0 |
| full decode | **1,016,182 / 1,016,192 bytes, 5,926,490 insns, 1.3717 b/insn** | 951,475 / 951,488 | (reference) |
| FMR | **1.02**, 58.607 MHz effective | 1.01, 59.665 MHz | — |

`0x800002f6` is `j boot_secondary_core` — the instruction a **secondary** hart reaches
after reset.S's enabling `sw`, which is exactly right for hart 2 and *different from*
hart 0's `0x800002f2` (`beq a0,t0`) in the same run. The two tiles report individually
correct, statically verifiable start PCs. All five varint bytes arrive on both.

### (2d.2) — the check that was missing, now obtained

Basic-block attribution over **468 basic blocks** totalling **7,891,599 cycles**, split
by address:

| region | cycles | share |
|---|---|---|
| generated model kernels (≥ `0x80001c00`) | **7,571,956** | 95.9% |
| boot / Zephyr / thread setup | 319,643 | 4.1% |

against two independent guest-side counters for the same window:

* `MODELBLASTER_WALL_CYCLES` = 7,541 (thousands) → decode is **100.41%**
* summed per-op `rdcycle` profile = 7,536,560 → decode is **100.47%**

The three agree to under half a percent. The 4.1% boot remainder is corroborated from a
fourth direction: the three idle tiles' windows close at **338,311 / 347,461 / 339,316**
cycles — the moment the traced worker's silencing loop stopped them — which is the same
319,643 cycles of pre-model execution seen from hart 2's own decode.

### The workload result is unchanged by the PCIM patch

| | quad PCIM | dual PCIM |
|---|---|---|
| `max_abs_err` / `max_rel_err` | **0 / 0** | 0 / 0 |
| model output | **`-56`, `127`** | `-56`, `127` |
| summed per-op profile | 7,536,560 | 7,539,118 (**0.034%**) |
| `MODELBLASTER_WALL_CYCLES` | 7,541 | 7,543 |

## 2. Guest defect 1 — `k_thread_cpu_pin()` on the running thread is a silent no-op

`harness_tacit/src/main.c` (and `harness/src/main.c`) did:

```c
k_thread_cpu_pin(k_current_get(), 1);   /* return value discarded */
```

`kernel/cpu_mask.c::cpu_mask_mod()` modifies the mask **only** for a thread that is
`z_is_thread_prevented_from_running()`; otherwise it returns `-EINVAL` and changes
nothing. `main()` is by definition running, so the call never moved anything. The
recorded dual run proves it from the other end: it pinned to hart 1 and printed
`=== TACIT === hart=0`. On the dual that is harmless (both tiles carry Saturn); on the
quad hetero it is fatal, because harts 0,1 have no vector unit.

So the fix the section above prescribes — "a `harness_tacit` build pinned to hart 2 or
3" — could not have worked through this API. The traced region now runs on a worker
created `K_FOREVER`, pinned, then started, and the pin's return value is checked:

```
800005a8: li a5,-1          # K_FOREVER
800005d6: jalr -> z_impl_k_thread_create
800005da: li a1,2           # MODELBLASTER_TACIT_PIN_HART
800005e2: jalr -> k_thread_cpu_pin
800005f0: jalr -> z_impl_k_wakeup      (k_thread_start)
800005fc: jalr -> z_impl_k_thread_join
```

Confirmed on hardware: the run prints `=== TACIT === hart=2`.

## 3. Guest defect 2 — the DT overlay disables the very harts the quad has

This is what actually blocked the quad, and it hid behind defect 1.

`harness_tacit/boards/chipyard_riscv64.overlay` (byte-identical to `harness/boards/`'s,
md5 `6b4eec15…`) disables **cpu@2 … cpu@7**; it was written for the 2-tile Shuttle SoC.
Combined with `CONFIG_MP_MAX_NUM_CPUS=4` from the quad overlay, Zephyr tries to start 4
CPUs while the devicetree exposes 2. The result is a silent SMP-bring-up deadlock with
**no console output whatsoever** — the guest never reaches `main()`, so none of the
harness's own diagnostics can fire, and every tile streams trace forever (the run was
producing ~10 MB/s and had passed 2.4e9 target cycles when it was stopped).

TACIT diagnosed its own harness. Decoding the live `tacit0.out` against the ELF:

```
11,975,433 iterations   0x800037e6-0x800037ea   = arch_cpu_start+0x4c
    800037e6: sd   a2,0(a3)      # riscv_cpu_wake_flag = <hartid of cpu N>
    800037e8: ld   a5,0(a4)
    800037ea: beqz a5,800037e6   # wait for that CPU to report in
```

and `tacit1.out` shows hart 1 *did* come up (it left reset.S's `boot_secondary_core`
spin after 58,888 iterations and reached `arch_secondary_cpu_init`, then parked in
`smp_init_top`'s start-flag wait) — while harts 2,3 never left the reset spin at all.
Hart 0 was waiting for a CPU whose DT node is `disabled`.

`harness_xpurt`, `harness_multi` and `harness_microros` carry **no** `boards/` overlay,
which is the only reason the sweep's and micro-ROS's quad builds ever booted.
`experiments/shard_dim/scripts/build_one.sh` already gates on the symptom — *"the
hart-count mismatch hangs in Zephyr's SMP spinwait before the boot banner (no output at
all)"* — but the gate only checks the Kconfig overlay, not the devicetree.

Fix, additive so the 2-hart builds are untouched:
`harness_tacit/boards/quad_hetero_4cpu.overlay` re-enables cpu@2, cpu@3 and is applied
via `-DEXTRA_DTC_OVERLAY_FILE`. Verified in the generated `zephyr.dts`: cpu@0..3 `okay`
(reg 0..3), cpu@4..7 `disabled`.

**`harness/` has the same latent conflict**: `harness/backends/firesim_chipyard_quad_hetero_q31.conf`
sets `MP_MAX_NUM_CPUS=4` while `harness/boards/chipyard_riscv64.overlay` disables
cpu@2-7. Any build combining those two files deadlocks the same way. Not fixed here —
no current experiment uses that combination — but it should not be trusted as working.

## 4. How to reproduce

```
west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 harness_tacit \
  --build-dir <bd> -- \
  -DMODEL_DIR=examples/dronet_tacit/int8/generated/rvv -DMODELBLASTER_BACKEND=rvv \
  -DMODELBLASTER_KERNEL_CFLAGS="-march=rv64gcv;-mabi=lp64d;-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1" \
  -DMODELBLASTER_TACIT_PIN_HART=2 \
  -DEXTRA_CONF_FILE=harness_tacit/backends/firesim_chipyard_quad_hetero_q31.conf \
  -DEXTRA_DTC_OVERLAY_FILE=harness_tacit/boards/quad_hetero_4cpu.overlay
# then, on the AWS manager:
f2_pcim_tacit_run.sh <tag> <lane> <host> f2_quad_hetero_norose_tacit_q31_60mhz_pcim 2400
# and locally:
experiments/tacit/analyze_run.sh <tacit2.out> <zephyr.elf>
```

Artifacts: `experiments/tacit/traces/f2_pcim_quad_dronet.*` (four tiles + uartlog +
`tacit2.vbb.csv`), `qpcim3.elf`, and the two deadlock decodes
`qpcim2_smpdeadlock_hart{0,1}.vbb.csv`.

## 5. Same-ELF A/B on the quad: PCIM vs BAR4

The 2026-09-02 A/B was done on the dual. Repeating it on the quad with the fixed
guest — byte-identical ELF (`md5 e7eab165…`), same SoC, same tile (hart 2), only the
hwdb entry changed — reproduces it and lands on the *degenerate* BAR4 case:

| | `..._pcim` (`agfi-010049831c489a814`) | baseline (`agfi-0662319f4b07483a7`) |
|---|---|---|
| zero bytes | **14 / 1,016,192 = 0.00%** | **8,000,000 / 8,000,000 = 100.00%** |
| longest zero run | 2 | 8,000,000 (the whole sample) |
| FSync / start PC | `0x800002f6` ✅ | **no FSync packet in the stream at all** ❌ |
| framing breaks | 0 | 0 |
| decode | 5,926,490 insns | undecodable — there is nothing there |

The BAR4 column is section 2a's "run 3" failure mode exactly, and it is the concrete
demonstration of the warning in that section: **framing breaks = 0 on a file that is
100% zeros.** `0x00` is a legal 1-byte TACIT packet (compressed taken-branch, ts delta
0), so the walker parses 8,000,000 "CTb" packets and reports a clean stream. Any health
metric that counts only parse errors calls this file healthy. It is empty.

The *guest* is unaffected by the transport: on the baseline bitstream the same binary
still boots, still reaches the traced worker on hart 2 and still silences the other
three encoders (tacit0/1/3 freeze at 45,056 / 12,288 / 131,072 bytes and stop growing),
while tacit2 keeps being filled with zeros by the BAR4 drain at ~46 KB/s. So the PCIM
patch changes what reaches the host, not what the target computes.

## 6. Per-function attribution vs the guest's own rdcycle brackets

The point of all this. Mapping the decoded basic blocks through the ELF's symbol table
gives a per-function cycle profile built entirely from the off-chip instruction trace,
which can be compared against `MODELBLASTER_PROFILE`'s in-guest `rdcycle` brackets —
two fully independent measurement paths over the same run:

| kernel | TACIT decode (by symbol) | guest `rdcycle` | delta |
|---|---|---|---|
| `conv2d_s8` (`mb_conv2d_s8_tiled_direct`) | 7,221,891 | 7,223,809 | **0.03%** |
| `maxpool2d_s8` | 239,318 | 239,396 | **0.03%** |
| `batchnorm2d_s8` | 44,107 | 45,904 | 3.9% |
| `add_s8` | 20,826 | 21,215 | 1.8% |
| `linear_s8` | 4,639 | 4,806 | 3.5% |

83 functions in total. The two dominant kernels — 91.5% and 3.0% of the traced window —
agree to three hundredths of a percent; the small ops differ by a few percent because
the `rdcycle` bracket includes the dispatch prologue that the symbol split attributes
elsewhere. The remaining 3.90% of the window is `wait_secondary_wake_flag` (308,042
cycles), i.e. hart 2 parked in reset.S before it was woken — the same pre-model
interval the three idle tiles' windows measure independently.

This is the per-component attribution the 2026-08-28 verdict was written to say was
impossible on F2. It is now routine on the quad.
