# The quad-only xpurt crashes: trap-entry V save vs an in-flight Saturn memop

Seven cells of the `wl_sweep` workload sweep fail on
`f2_quad_hetero_norose_tacit_q31_60mhz` -- six with a fault, one with a silent
hang. All seven are `quad`. Nothing else in the sweep fails.

    wl_bimodal_quad_greedy_shard        mcause 7  mtval 0x424c342794
    wl_control_mix_quad_greedy_base     mcause 5  mtval 0xffffffffffffffc0
    wl_depth_chain_quad_greedy_base     mcause 7
    wl_perception_heavy_quad_greedy_base mcause 5
    wl_scale_ladder_quad_greedy_shard   mcause 7  mtval 0x29e708572c
    wl_vint_intro_quad_greedy_shard     mcause 5  mtval 0x402
    wl_saturation_quad_greedy_base      no fault -- 90 min timeout with no output

Six of the 19 quad cells fault (`wl_saturation_quad_greedy_base` is the hang,
and had no collected uartlog in `wl_sweep` at all -- its evidence is the
`wl_sweep_quad_diag/*.try1.uartlog` re-run). 0 of the 48 non-quad cells fail.
Under a uniform per-cell failure rate, all six faults landing in quad has
probability (19/67)^6 = 5e-4, so the `quad` association is real.

## 1. What the faults actually are

Symbolized with the Zephyr SDK toolchain against `experiments/wl_sweep/elf/<tag>.elf`.
The `-dwarf` line numbers alone are misleading here (they attribute an inlined
chain), so every PC below was confirmed by disassembling the exact `mepc`.

| cell | mepc | instruction | corrupt operand |
|---|---|---|---|
| control_mix_quad_base | `0x8001b3be` | `vle8.v v1,(t3)` in `mb_conv2d_s8_tiled_direct` | `t3 = 0xffffffffffffffc0` |
| scale_ladder_quad_shard | `0x8002f34c` | `vsse8.v v2,(a7),a0` in `mb_conv2d_s8_tiled_direct` | `a7 = 0x29e708572c` |
| bimodal_quad_shard | `0x8001c024` | same `vsse8.v` epilogue (register profile is byte-identical to scale_ladder: `a3=4 a4=-23 a5=32 a6=127 t5=31 t6=-2`) | `a7 = 0x424c342794` |
| vint_intro_quad_shard | `0x80045fd6` | `ld a5,1024(a0)` inside **`z_riscv_vstate_restore_thread`** | `a0 = 2` |
| depth_chain_quad_base | `0x80015e3a` | `mb_conv2d_s8_tiled_direct` (dronet_sf), same `vsse8.v` site | `a7 = 0x8061a263c8` |
| perception_heavy_quad_base | `0x80012e14` | `mb_conv2d_s8_tiled_direct` (yolov8_nano_sf), same `vle8.v` site | `wq = 0x40` |

`mtval` is the sign-extended 40-bit physical address of the faulting register,
which is why it does not always match the register byte for byte:
`a7 = 0x0000008061a263c8` -> `mtval = 0xffffff8061a263c8` (bit 39 set),
`a7 = 0x000019424c342794` -> `mtval = 0x424c342794` (bit 39 clear).

The two `wq` values are worth putting side by side: `control_mix` faulted with
the weight pointer at `-64` and `perception_heavy` with it at `+64`, while the
`control_mix` dump has `t1 = t4 = 0x40` live in neighbouring registers. That is
a register taking on a small value that is live nearby, not an address
computation going wrong -- the same shape as `5057b0e`'s "every register is
correct for iteration ... EXCEPT a0".

Five of the six faults are in `mb_conv2d_s8_tiled_direct`, at one of its exactly
two vector memory operations, across six different models
(`yolov8_nano_sc`, `yolov8_nano_sf`, `yolov8_nano_sh_wls`, `dronet_sf`,
`dronet_sg_wls`, `vint`) and six different binaries. The sixth is inside the V
restore on the trap-exit path.

**The defect survives a rebuild.** `depth_chain_quad_base` and
`perception_heavy_quad_base` were rebuilt on 09-05 and faulted again on the NEW
binaries; `control_mix`, `scale_ladder` and `vint_intro` faulted again on 09-05
from the unchanged 09-03 ELFs, and had already been re-run bit-identically once
before (`wl_sweep_quad_diag/`). So every cell has reproduced at least twice, and
two of them across a rebuild -- this is not a code-layout accident.

Every one of these is a register holding a value the code **provably cannot
compute**:

- `vle8.v v1,(t3)` is `__riscv_vle8_v_i8m1(wq, vl)` with
  `wq = weight + ic*KH*KW*OC + oc_base + kh*KW*OC`.
  `weight` is `yolov8_nano_sc_detect_cv2_1_2_weight_q_rvv`, a link-time constant
  at `0x807222f8` (`R`, rodata) passed straight through `parallel_conv2d_s8`,
  whose pool path is `#if 0`'d on this backend. Every addend is a small
  non-negative `int`. `wq` cannot be `-64`.
- `vsse8.v v2,(a7),a0` is `MB_STORE_PIX`, i.e.
  `__riscv_vsse8_v_i8m1(op + OFF, st, ...)` with
  `op = output + (n*OC+oc_base)*OH*OW + oh*OW + ow`, `output` a `.bss` buffer at
  `0x8xxxxxxx` and the arithmetic hoisted to `size_t` (the old 32-bit index wrap
  documented at `rvv_conv2d_s8_rvv_vsmul_vnclip.c:218` is fixed and excluded --
  and `0x29e708572c` is not a wrap of any `0x8xxxxxxx` base anyway).
- `z_riscv_vstate_restore_thread(thread)` with `thread == (struct k_thread *)2`.
  It reaches there from `fp_trap_exit` via `z_smp_current_get()`, which returns
  `_current_cpu->current`. That can never be 2.
  (`mtval = 0x402 = 2 + 1024`, and `1024` is `offsetof(k_thread, arch.saved_v_context.vstart)`
  -- which is exactly why the earlier "tensor base pointer indexed from NULL"
  reading looked plausible. It is not a tensor pointer at all.)

`t3` and the conv kernel's `s4` are **callee-saved** registers, live across the
loop. A callee-saved register changing under a running function is not a
codegen or aliasing bug; it is the thread's architectural state being corrupted
from outside.

## 2. Not a tiling, alias, scheduler or memory bug -- and not reproducible in software

- **Three of the six are base arm** (control_mix, depth_chain, perception_heavy;
  the saturation hang is a fourth), so no tile/alias explanation covers it.
  Independently: *zero* entries in *any* table of this sweep have `n_harts > 1`
  (checked across all 67 generated tables), so `pool_for_entry()` returns NULL
  everywhere, the `parallel_*` pool paths are `#if 0`'d for this backend, and
  the shard/base distinction never reaches the runtime at all.
- **The generated `_main.c` for quad and hetero are identical** modulo the tag
  name and per-network instance counts. The only quad-vs-hetero delta is which
  harts the dispatch table names.
- **Spike does not reproduce it.** All three failing ELFs I ran (`control_mix_quad_base`,
  `scale_ladder_quad_shard`, `saturation_quad_base`) complete cleanly under
  `spike -p4 --extension=gemmini --isa=rv64gcv_zicntr_zfh_zvfh`, with all four
  workers on harts 0/1/2/3 and `yolov8_nano_sc max_abs_err=0`. Same ELF, same
  schedule, same allocation order. So it is not a NULL allocation, a static
  workspace-slot overflow, a stack overflow, or anything else a functional
  model would show. Per-hart workspace slots were audited anyway: every one
  (`MB_GEM_*_WS_SLOTS`, `MB_RVV_CONVPC_WS_SLOTS`, `MB_RVV_SILU_WS_SLOTS`) is
  sized `CONFIG_MP_MAX_NUM_CPUS`, which is 4 in **every** arm of this sweep --
  the Kconfig overlay is `firesim_chipyard_quad_hetero_q31.conf` for all of
  them, so .bss layout is byte-identical across arms.

## 3. Root cause

This is the defect already root-caused in modelblaster commit `5057b0e`
("harness: mask IRQs during inference -- works around trap-path vector
corruption"), re-exposed by commit `98901c5`.

`zephyr_ws/zephyr/arch/riscv/core/isr.S:202-210` calls `z_riscv_vstate_save`
on **every trap entry**, before `mepc`/`mstatus` are even stacked:

    sr s0, __struct_arch_esf_s0_OFFSET(sp)
    get_current_cpu s0
    call z_riscv_vstate_save          <-- 0x80000058 in the shipped ELF

and `isr.S:915-922` calls `z_riscv_vstate_restore` on every trap exit
(the `fp_trap_exit` label is on the common return path, not only the FP one).

`z_riscv_vstate_save_thread` (`arch/riscv/core/v.c:139-154`) issues

    csrw    vstart, x0
    vsetvli t, x0, e8, m8, ta, ma
    vse8.v  v0/v8/v16/v24, (save_area)

into Saturn's **decoupled** vector unit while the interrupted thread's own
vector memop may still be in flight. On this SoC that corrupts the interrupted
thread's architectural registers. `5057b0e` established this with a bisection
over a rodata function-pointer table (NOP-ing entries preserves layout exactly,
yet the symptom *moves*: fault -> identical fault -> hang -> different fault ->
pass) and two independent levers: an 8-byte patch clearing `mstatus.MIE` made
the identical faulting ELF run clean (job 283), and stubbing
`z_riscv_vstate_save` did the same (job 287). It closed with "the trap-return
path really does corrupt registers on this SoC ... the residual is a
hardware/trap-interaction defect."

`vint_intro_quad_shard` is the direct fingerprint of that mechanism: the fault
is *inside the trap-exit V restore itself*, on a `thread` pointer that the
kernel cannot have produced.  The generator's own guard comment
(`pipeline/generate_xpurt_main.py:706-712`) already records that exact
signature from a previous ViNT run --

        mcause: 5, Load access fault
        mepc -> z_riscv_vstate_restore_thread  (arch/riscv/core/v.c)
        ra   -> kernel_sigmoid_s8_vint_gemmini_q31

-- down to the `ra` pointing into a *model kernel* rather than into isr.S,
which is impossible from the call graph (`z_riscv_vstate_restore` has exactly
one caller, `fp_trap_exit` at `0x80000234`, so `ra` must be `0x80000238`).
Mine reads `ra -> kernel_adaptive_avg_pool2d_s8_vint_rvv`. The stacked `ra` is
itself one of the corrupted registers.

`mepc` is not the corrupt thing here -- the CPU really did execute that code.
Two instructions before the fault, `restore_thread` computes
`csrr a5,misa; slli a4,a5,0x2a`, and the dump has
`a5 = 0x8000000000b4112d` (a valid `rv64imafdcv` + S/U/X misa, so this is a
Saturn hart) and `a4 = 0xd044b40000000000`, which is exactly
`(a5 << 42) mod 2^64`. So `a0 = 2` and `ra = 0x8003fa4e` are corrupt *while the
function is legitimately running*: several registers, not one, and not the PC.

**Why it is live again.** The xpurt dispatch guard used to be `irq_lock()`.
Under `CONFIG_SMP`, `irq_lock()` is `z_smp_global_lock()`, which does two
things: `arch_irq_lock()` (mask this hart's MIE -- the part that actually
suppressed the fault) **and** take Zephyr's big kernel lock (the part that
serialized the harts). Commit `98901c5` correctly measured the BKL as the whole
of a 6632 -> 5231 us makespan gap and replaced the guard with
`k_sched_lock()`. But `k_sched_lock()` only blocks a *thread switch*. It does
not mask interrupts, and the corrupting code runs on **trap entry**, switch or
no switch. So the guard no longer covers the hazard at all.

`98901c5` said so itself: *"NOT yet evaluated: whether the weaker guard still
suppresses the ViNT vector fault ... It needs a fused_vint soak."* This sweep is
that soak, and the answer is no. Its reasoning -- *"what must not happen inside
a vector kernel is a THREAD SWITCH, not an interrupt"* -- is the one thing it
got wrong, and `5057b0e`'s job-287 lever had already shown it.

This is visible in the shipped binary. `k_sched_lock` at `0x800494de` in
`wl_vint_intro_quad_greedy_shard.elf`:

    800494e8:  csrrci s2,mstatus,8        <- mask MIE, save old
    800494f6:  amoadd.d.aqrl ...          <- take _sched_spinlock (ticket)
    80049522:  lbu  a5,27(a0)             <- _current->base.sched_locked
    80049528:  sb   a5,27(a0)             <-   ... minus one
    8004952e:  amoadd.d.aqrl ...          <- release _sched_spinlock
    80049532:  andi s2,s2,8
    80049536:  csrs mstatus,s2            <- RESTORE MIE
    8004953a:  ret

It masks MIE only across the counter update and re-enables it on the way out,
so the kernel call that follows runs with interrupts **on**. And the spinlock
really is dropped before returning, which is why the guard cannot re-serialize
the harts -- `98901c5` was right about that half.

## 4. Why 4 harts and not 2

The hazard rate is (traps taken on a Saturn hart) x (how long a Saturn vector
memop stays outstanding).

I tested the obvious first factor and **it is refuted**. `exposure.py` /
`exp2.py` compute, per cell, the expected number of peer-generated reschedule
IPIs landing while a Saturn hart is inside a vector kernel
(`arch_sched_broadcast_ipi` pokes MSIP on every other online hart --
`arch/riscv/core/ipi_clint.c`). By that metric `rvvpair` scores *higher* than
`quad` on every workload (control_mix: rvvpair base 428 vs quad base 208;
scale_ladder shard: rvvpair 730 vs quad 666) and `rvvpair` never fails. Neither
that metric nor the predicted gemmini/saturn overlap structure separates the
six failing quad cells from the thirteen passing ones. Interrupt *count* is not
the discriminator; I am not going to claim it is.

What is left, and what the arm structure actually points at, is the second
factor -- the length of the in-flight window:

| arm | gemmini harts busy | saturn harts busy | fails |
|---|---|---|---|
| gempair | 2 | 0 | 0/? (no vector at all) |
| rvvpair | 0 | 2 | 0 (most vector time of any arm) |
| hetero  | 1 | 1 | 0 |
| **quad** | **2** | **2** | **6/19** |

`quad` is the only arm in which both Gemmini tiles are issuing RoCC DMA while
both Saturn units are issuing vector memops. Saturn's load/store queue drains
against the same L2/DRAM those DMAs are saturating, so the window during which
a vector memop is still outstanding after the scalar core has run past it is at
its longest exactly there. `rvvpair` has more vector time but no competing DMA;
`hetero` has DMA but half of everything. That is the best-supported reading of
the arm table, and it is consistent with a hazard whose trigger is a
coincidence in time rather than a count of events -- which is also why *which*
quad cell dies is not predictable from the schedule, while *that* it dies is
bit-reproducible: FireSim on a fixed bitstream is cycle-deterministic, so the
same ELF re-runs the same coincidence exactly (confirmed: the five re-runs in
`experiments/wl_sweep_quad_diag/` are bit-identical to the originals).

I have not proven the memory-pressure half. It is an inference from the arm
table, not a measurement.

The cleanest illustration that cell selection is a coincidence and not a dose
is `depth_chain_quad_base` (FAILS) against `depth_nav_quad_base` (PASSES).
`depth_nav` is `depth_chain` plus an independent control loop, and the two have
**identical** Saturn-side conv exposure -- 4 rvv conv2d dispatches, 8.33 ms of
predicted rvv conv time, same 430.6 ms span. Three separate exposure metrics
(peer IPIs into the vector window, gemmini/saturn overlap, rvv conv dispatch
count and time) were computed for all 19 quad cells and none of them orders the
failures. What varies is where an interrupt lands, and that is fixed per
(bitstream, ELF) but otherwise arbitrary.

## 5. Hardware experiment

No rebuild: the shared modelblaster build tree and all three fq lanes were in
use by another campaign, so nothing here writes outside `experiments/quad_debug/`.

`patch_vstate_trap_hooks.py` writes `c.ret` at the entry of **both** isr.S-side
wrappers (`z_riscv_vstate_save`, `z_riscv_vstate_restore`) in the exact
faulting ELF, after asserting each has exactly one call site. It deliberately
leaves `z_riscv_vstate_save_thread` / `z_riscv_vstate_restore_thread`
untouched, so switch.S keeps doing per-thread V context switching. That is
*intended* to be safe because `CONFIG_RISCV_V_KERNEL_ONLY=y` strips `v` from
the global `-march` (`cmake/compiler/gcc/target_riscv.cmake:181`), so no ISR
should be able to clobber `v0..v31` behind the interrupted thread -- though see
the caveat about picolibc's `vse64.v` memcpy in `RESULT.md`.

Sanity check first: the patched `control_mix_quad_base` ELF produces
**byte-identical** `MODELBLASTER_VERIFY` lines to the unpatched one under
`spike -p4`, so the patch is numerically inert in software.

**Result: `wl_scale_ladder_quad_greedy_shard` STILL FAULTS** (fq 912), at the
same `vsse8.v v2,(a7),a0` in the same `mb_conv2d_s8_tiled_direct` source line,
same `sp`, byte-identical neighbouring registers -- only which dronet rung was
executing moved. `wl_control_mix_quad_greedy_base` stopped faulting (fq 911,
232 trace rows) but drifted numerically. Full numbers and both fault frames in
`RESULT.md`. The negative is the load-bearing one: it says the corrupting agent
is the trap, not the V-save code the trap happens to run, which is why the fix
in section 6 is the interrupt mask and not this.

## 6. The fix

`arch_irq_lock()` + `k_sched_lock()` around the dispatch, replacing the
`k_sched_lock()`-only guard. `xpurt_dispatch_guard_arch_irq_lock.patch`.

This is **not a new idea and not an untested one**. It is the change
`harness_microros/src/main.c:284-318` already ships as
`MICROROS_DISPATCH_GUARD_IRQ` (default on), and the acceptance run is in
`experiments/kernel_opt_log.jsonl`:

- `microros-3net-rvv-preempt-004-sched-lock-insufficient` (fq 837) --
  *"Is k_sched_lock() around each dispatch enough, as it was for the xpurt
  scheduled harness? NO."* Fault: `mcause=7`, `mepc = vsse8.v v2,(a7),a4`,
  `mtval=0x529f7725e4` (not in DRAM) with the other registers in the frame
  valid. *"A bare timer tick mid-kernel suffices, and k_sched_lock() does not
  mask interrupts."*
- `microros-3net-rvv-preempt-005-ACCEPTANCE-clean-3net` (fq 847) --
  *"Does arch_irq_lock()+k_sched_lock() per dispatch give a clean 3-model
  micro-ROS run on the Config-B pinning that faults without it? PASS."* Same
  bitstream, 413 trace rows, 0 corrupted slots, `*** PASSED ***`.

So the same defect was found and fixed in the sibling harness two days before
this sweep ran, and `harness_xpurt` -- whose generator is where `98901c5`
weakened the guard -- never got the fix.

`arch_irq_lock()` is a bare per-hart `csrrc mstatus, MSTATUS_IEN`
(`include/zephyr/arch/riscv/arch.h:257`); there is no lock in it, so it cannot
bring back the cross-hart serialization that `98901c5` correctly removed. Cost
is one delayed tick per dispatch, and with `CONFIG_TICKLESS_KERNEL=y` the
elapsed time is recovered from free-running mtime on unmask, while
`k_cycle_get_64()` reads mtime directly -- so every trace and profile number
stays correct.

### What does NOT work, tested here

Gating the isr.S trap-entry/exit V hooks off (`zephyr_isr_v_hooks.patch`, and
the binary-patch experiment in section 5) is **not sufficient**, and this is
the main new result of this investigation -- see `RESULT.md`. With both hooks
stubbed to `ret`, `wl_scale_ladder_quad_greedy_shard` still faults, at the same
`vsse8.v v2,(a7),a0` in the same `mb_conv2d_s8_tiled_direct`, with the same
neighbouring register profile and `a7` garbage. That reproduces
`convfix-007`'s residual on this harness at 4 harts: the corrupting agent is
the trap itself, not the V-save code the trap happens to run. Keep that patch
only as an optional follow-on -- under `CONFIG_RISCV_V_KERNEL_ONLY` those
hooks preserve nothing and cost two calls per trap -- but it must not be
shipped as the mitigation.

### Not applied

Neither patch is applied. Another agent is mid-campaign on this sweep, and the
guard change alters the timing of every cell in every arm; landing it now would
leave the sweep internally inconsistent (some cells guarded one way, some the
other), which is worse than a documented defect. It should go in when that
campaign lands, with a re-run of the quad column and a check that a 2-hart
makespan does **not** move (if it does, the BKL is back and the wrong lock is
being taken).

## 7. What was ruled out along the way

- Hart-count / hart-index sizing in `generate_xpurt_main.py`: `XPURT_MAX_WORKERS=32`,
  `XPURT_MAX_HARTS=256`, `g_hart_acc[]` indexed by worker, claim-coverage audit
  runs before any thread exists. Nothing is sized for 2.
- Tensor ALIAS flattening / `apply_split_hint._register_tile_tensors`: cannot be
  the cause -- two failures are base arm, no schedule in the sweep has a
  composite entry, and the corrupt values are in callee-saved registers of a
  running function, not in a tensor base computed at codegen.
- `CONFIG_RISCV_VECTOR_MAX_LEN` vs actual VLEN (the `_v512` trap): the shipped
  `.config` has 256 and the disassembly agrees (`vreg` at `+256` inside a
  256-aligned context, 4 x 256 B of `vse8.v` exactly filling `char[32][32]`).
- Heap/stack exhaustion from 4 x 1 MB worker stacks in an 8 MB heap:
  `pthread_create` failure is checked and fatal, and spike runs the identical
  allocation sequence clean.
