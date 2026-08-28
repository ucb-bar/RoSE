# ModelBlaster kernel campaign — continuation brief

Written 2026-08-28 for a successor session, after the local workstation
(garden) began crashing every ~12 minutes on an uncorrectable PCIe AER fault
(`vendor 0x1022 device 0x1483`, an AMD host bridge). That fault killed five
background agents repeatedly. **Nothing below depends on garden.**

## Where the work lives

| what | where |
|---|---|
| kernels + pipeline | `ucb-bar/ModelBlaster` branch `kernel-opt-campaign-20260828` (`d201114`) |
| evidence, queue fixes, docs | `ucb-bar/RoSE` branch `rose-2-dev` (`46851ec`) |
| experiment log (245 entries) | `RoSE:experiments/kernel_opt_log.jsonl` |
| live tracker source | `RoSE:experiments/docs/kernel_tracker.html` |
| tracker (published) | https://claude.ai/code/artifact/70ce8322-7549-4a15-8fe7-42f773cc39b1 |
| method log (published) | https://claude.ai/code/artifact/38c779bd-f209-4165-bec6-a9242d7625cd |

The RoSE submodule pointer still references the OLD ModelBlaster commit.
Updating it cascades `xpu-rt` -> `zephyr-chipyard-sw` -> `RoSE`; deliberately
not done, since the branch is unmerged.

## Results — against each backend's OWN original curated kernels

Scoring against scalar reference inflates these ~30x and measures "curated
kernels exist", not "kernels were optimised". Use the on-backend numbers.

| model | backend | baseline | now | speedup |
|---|---|---:|---:|---:|
| fused_full | pure fp16 | 60,371,259 | 7,710,288 | 7.83x |
| dronet | rvv | 17,813,964 | 7,496,398 | 2.38x |
| yolov8_nano | rvv | 333,175,953 | 165,326,342 | 2.02x |
| fused_full | int8 hybrid | 7,587,275 | 4,922,762 | 1.54x |
| dronet | gemmini q0.31 | 20,933,650 | 18,311,704 | 1.14x |
| mlp_generic | rvv (control) | 9,881 | 9,796 | 1.01x |

`fused_full` exists as TWO builds sharing a backend name — an int8 hybrid
(int8 conv encoder + fp16 tail) and a pure fp16 build. Conflating them cost
real time. Never compare across them.

## % of peak (dtype-aware roof: int8 = 8 ops/cyc, fp16 = 16 narrow / 8 widened)

    rvv (int8)      3.36 ops/cyc   42.1%
    gemmini (int8)  1.44 ops/cyc   18.0%
    conv2d_s8       3.52 ops/cyc   44.0%   <- 89% of all measured cycles
    conv2d_f16                     25.2%   (of its widened ceiling)
    gemmini maxpool                 0.5%   <- worst material entry
    gemmini batchnorm               0.3%
    gemmini add                     0.3%

int8 peak is 8 ops/cyc, NOT 64: the backend builds `-march=rv64gcv` with no
Zvqdotq, so `vwmacc` is the only int8 MAC path and writes 32 bits per MAC.

## Hazards that cost real time — read before touching kernels

1. **32-bit index wrap.** A conv kernel indexed with int arithmetic while its
   sibling used `size_t`. BSS lives above 0x80000000, so the index wrapped —
   presenting as err=1/3/11 OR a Store/AMO fault depending on binary layout,
   and invisible to spike. Two of three siblings already had the fix. ALL
   tensor indexing must use `size_t`. Audit: `experiments/kopt/sibling_audit.py`.
2. **`accuracy_class` is a claim, not a fact.** Two kernels declared
   `bit_exact` and measured err=17 in isolation. Isolation-test: the op
   curated alone, everything else forced to reference. Mixed configs let drift
   sources cancel and hide each other.
3. **Validation gaps.** `add_s8` "passed" on dronet only because 100% of its
   calls fell back to scalar there; it measured err=43 on yolov8n. Verify on a
   model that actually exercises the path you changed.
4. **Shared `target_affinity` edits.** `_conv_weight_layout_for_backend()`
   derives ONE weight layout per backend from `conv2d_s8`'s algorithms and
   applies it to every 4D tensor. Adding a backend to an affinity list
   silently transposed an unrelated op's weights and surfaced as uniform error
   in untouched kernels. A trip-wire now hard-fails this; do not defeat it.
5. **Curated kernels are silently OFF unless `GLOBAL_CURATED_DIR` is set**
   (`examples/_run_lib.sh:156`). Confirm selection via `kernel_picks.json`,
   never infer from cycles.
6. **Spike is not a sufficient oracle.** It missed both a wrong-answer bug and
   a hard fault that hardware caught. It IS reliable on gemmini targets
   (matched hardware 6/6) — bisect there, confirm on FPGA.

## Measurement rules (these changed twice; this is the settled version)

- Run-to-run on an identical ELF: **exactly zero** variance. FireSim is
  cycle-exact — verified on a 204-dispatch model across every per-dispatch
  count. Never re-run to average.
- Whole-model across REBUILDS: **~3%**.
- Per-operator across rebuilds: **up to 9-10% even for operators you did not
  touch** (i-cache alignment). A per-op delta under ~10% is NOT established
  without the untouched-operator control.
- Verify every uartlog: the ELF name FireSim embeds in the guest command line,
  `harness: model=`, row count, and `DONE rc=0` in `/var/lib/fq/daemon.out`.
  `status.json` has lied.

## Infrastructure

- **Manager** `ubuntu@3.88.218.39` (m5.xlarge, 4 vCPU — 3 weeks uptime,
  unaffected by garden's fault). fq queue at `~/fpga_queue`,
  `export FQ_SOCKET=/var/lib/fq/fq.sock`.
- **FPGA pool: 3 lanes** (f2-00/01/02 = 192.168.0.8/.31/.91). Five F2
  instances STOPPED (not terminated) after measuring 7% utilisation over 16h;
  restart them if a broad sweep justifies it.
- **Build host** c6i.4xlarge, $0.68/hr, 16 vCPU — this machine. Stop it when
  idle.
- Use `--hw-config f2_dual_small_norose_tacit_q31_60mhz`: cycle-identical to
  20MHz, 2.99x faster wall-clock.
- **NEVER restart the fq daemon with sudo** — `run_key: ~/firesim.pem` becomes
  `/root/firesim.pem`, every lane probe fails, the pool dies for everyone.
  As ubuntu: `cd ~/fpga_queue && setsid nohup python3 -m fq.cli daemon --pool
  /var/lib/fq/pool.yaml >>/var/lib/fq/daemon.out 2>&1 </dev/null &`
- F2 capacity envelope: dual+small+RoSE fits; dual+large WITHOUT RoSE fits;
  all quad configs and dual+large+RoSE fail place/route (10 of 16 builds).

## Open work

1. **iGEMM conv prototype** — kernel exists at
   `kernels/rvv/rvv_conv2d_s8_rvv_igemm.c` (19,556 B) and IS registered.
   Never built or measured. Reference: `/scratch2/dima/zephyr-chipyard-sw-iiswc/
   third-party/XNNPACK` (1,359 igemm files, 260 RVV microkernels) — our own
   submodule is empty. Conv is 89% of cycles at 44% of peak, and the limit is
   structural (MAC/cyc flat in OC), so this is the highest-value open item.
   Its `accuracy_class` defaults to BIT_EXACT and MUST be isolation-tested.
2. **`gemmini_q31_rvv/add_s8` still has the unfixed HW branch** — the exact
   construction proven to give err=43. Its pure-target sibling was fixed; the
   fused copy was missed. Delete the HW branch, verify on yolov8n.
3. **Relabel `gemmini_q31_linear_s8_gemmini_tiled_matmul`** to `bit_exact`.
   It declares `numeric_drift` but its code uses `full_C=true` + exact CPU
   requantize, structurally identical to the audited bit_exact kernel, and was
   isolation-measured at err=0. Its "<=1 LSB/layer" header is leftover text.
4. **Gemmini scalar fallbacks** — maxpool/batchnorm/add at 0.3-0.5% of peak
   are ~8.9M of 19.2M cycles, more than conv. Fusion (conv+relu+pool via
   `tiled_conv`) is approved as an OPT-IN flag built on the EXISTING fusion
   passes — which fire zero times on yolov8n because they key on ReLU while it
   uses SiLU. A `conv2d_silu_s8` spec exists with no pass producing it.
5. **Decompose the 6.2x exact-path cost** (11,598,513 vs 1,860,175): 4x drain
   traffic (int32 not int8) PLUS a CPU requantize that is SCALAR on the pure
   target by design. On the FUSED target it could be vectorised with NO
   numeric change — possibly recovering most of the gap.
6. **`rvv` all-reference measures err=92-100 vs golden** while `scalar` is
   clean. Bisected to `conv2d_s8`: 241 vs 238 instructions from byte-identical
   C differing only in `-march`. Toolchain/codegen bug class, not numerics.
7. **Gemmini coverage is thin**: only dronet has a real profile. yolov8n
   CRASHED (mcause=1, fetch from address 0 — never root-caused, distinct from
   the RVV index wrap). fused_full has NEVER been measured on gemmini.

## Things already tried that did NOT work — do not repeat

- OW-vectorization for conv: **1.7x slower** (makes the input the vector
  operand; at SW=2 that is a strided `vlse8` Saturn serialises at ~1 elem/cyc).
- `rvv_oc_blocked`: **1.66-1.81x slower** than pixel tiling. Obsolete.
- `vsetvl`-hoisting in linear/lstm: spike predicted 32%/5%, FPGA showed
  nothing outside noise.
- Batching Gemmini CPU tiles to 320 rows: removes ~95% of config/flush
  reissue, net **+0.33% regression**. Per-tile RoCC ceremony is NOT the cost.
- fp16 narrow accumulate to double the ceiling: rejected — the kernel spec
  requires fp32 accumulation to avoid cancellation over 576-term reductions.
