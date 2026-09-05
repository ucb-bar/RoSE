# micro-ROS 3-net serialization — systematic root cause

**Status: ROOT-CAUSED AND FIXED (2026-09-04).** See the verdict at the bottom.
The register below is kept as written so the refuted hypotheses stay visible.

**Status was: OPEN.** This document is the falsification register. Every hypothesis
below gets an experiment that can *kill* it; a hypothesis is only "confirmed"
once its rivals are dead, not once evidence is consistent with it.

## The symptom

From `modelblaster/examples/microros_demo/NOTES_3NET.md` §"Hart 1 serialization
with rclc multi-timer executors" (2026-05-11, U250 2-hart bitstream):

* 2-net (yolov8 + dronet, separate executors): **91% of dronet's dispatches run
  concurrently** with yolov8.
* 3-net: hart 1 essentially idle, **0.4% concurrent**.
* The trigger is recorded as "adding a third executor **or third timer**".

Ruled out there: HW bus contention (Spike reproduces it with no bus),
yolov8's `irq_lock`, `K_FP_REGS` cost, and multi-executor-on-one-hart —
because **one executor with two timers shows the same gap**. That last control
is the important one: node count, executor count and hart placement are all
held constant and it still serializes.

Concluded there: "`rclc_executor_spin_some`'s internal `rcl_wait` blocks on
something that only unblocks when hart 0's executor activity stops." Never
resolved; `MICROROS_2EXEC_FUSE_BC` is a *workaround* that hides the bug by
making three networks look like one callback.

## Why the May pass could not finish it, and what changed

It had no visibility inside the middleware — it was A/B knob-flipping, which is
why it landed on "blocks on something". Two things are different now:

1. **TACIT works on the quad.** `f2_quad_hetero_norose_tacit_q31_60mhz_pcim`
   traces end to end: 1,016,192 B decoding to 5,926,490 instructions, with
   per-function attribution matching the guest's own rdcycle brackets to 0.03%.
   It already diagnosed an SMP deadlock by showing 11,975,433 iterations of
   `arch_cpu_start+0x4c`. We can *read* where the stalled hart spins.
   NOTE: it must be a `_pcim` bitstream. On the non-PCIM quad the BAR4 drain
   returns 8,000,000/8,000,000 zero bytes with framing breaks = 0 — silent
   corruption (`experiments/tacit/traces/f2_bar4_quad_dronet.tacit2.head8m.bin`).
2. **The global lock the design relies on does not exist in this binary.**
   `libmicroros/include/uxr/client/profile/multithread/multithread.h` under
   `#else // UCLIENT_PROFILE_MULTITHREAD` defines `UXR_LOCK`, `UXR_UNLOCK`,
   `UXR_LOCK_SESSION`, `UXR_LOCK_STREAM_ID` and `UXR_LOCK_TRANSPORT` as EMPTY,
   and this build does not define that profile. So "serializes under a global
   mutex" describes the *design*; the *binary* has N threads mutating the global
   session list, the static buffer pool and each `uxrSession` state machine with
   no synchronisation at all. Those two look identical from outside and have
   opposite fixes.

## Hypotheses

| # | Hypothesis | Mechanism | Kills it |
|---|---|---|---|
| H1 | Unsynchronised session mutation | `UXR_LOCK*` are empty macros, so concurrent executor threads corrupt shared `uxrSession` stream/sequence state | Build with `UCLIENT_PROFILE_MULTITHREAD=ON`. If the result becomes *clean serialization* rather than a stall, H1 is the cause and the "single-threaded by design" story is the real one. If the stall is unchanged, H1 is dead. |
| ~~H2~~ | **DEAD** — static input-buffer pool exhaustion | `rmw_wait` calls `rmw_uxrce_clean_expired_static_input_buffer()` first thing, over a pool sized `RMW_UXRCE_MAX_HISTORY` with `is_dynamic_allowed = false` (`rmw_init.c:221-223`). Scales with session count — matches "2 works, 3 does not". | Killed by source inspection, no build spent. The pool is allocated from in exactly three places — `callbacks.c:82` `on_topic`, `:141` `on_request`, `:202` `on_reply` — i.e. only when a subscription, service or client RECEIVES data. This harness is built with `SUBSCRIBERS=0 CLIENTS=0 SERVERS=0` and runs `MICROROS_NO_PUBLISH=1`, so nothing ever allocates and `rmw_uxrce_clean_expired_static_input_buffer()` walks a permanently empty list. |
| H3 | Transport queue backpressure | `transport_loopback.c:83` `k_msgq_put(..., K_MSEC(50))` blocks 50 ms per call when the target queue is full; a session nobody services fills up | Raise the msgq depth, or read the trace for time inside `k_msgq_put`. |
| H4 | Threads not actually on separate harts | `k_thread_cpu_pin` on a *running* thread returns `-EINVAL` and callers discard it | **DEAD.** `harness_microros/src/main.c:1223-1256` creates every thread `K_FOREVER`, pins, *then* starts — the correct order. Confirmed by `arch_proc_id()` at thread entry printing the requested hart. |
| H5 | `available_contexts == 0` forces the all-sessions path | `rmw_wait.c` sets `need_to_be_ran` only from services/clients/**subscriptions**; this harness has none, so the count is always 0 and the else-branch runs `uxr_run_session_timeout` for EVERY session in the process, from whichever thread called | Add one dummy subscription so a context is genuinely "available", or trace which branch is taken. |

**After H2 died, H1 and H5 collapse into one mechanism**, and it is the leading
candidate:

> Because the harness has zero subscriptions, `available_contexts` in `rmw_wait`
> is *always* 0, so the else-branch always runs — every executor thread calls
> `uxr_run_session_timeout()` on **every session in the process**, not just its
> own. And because `UCLIENT_PROFILE_MULTITHREAD` is undefined, the `UXR_LOCK`
> that is supposed to make that safe is an empty macro. So N threads drive N
> shared `uxrSession` state machines with no synchronisation, and the damage
> scales with session/handle count — which is precisely the recorded trigger
> ("a third executor *or* a third timer"), and precisely why FUSE_BC (one
> executor, one timer, one `rcl_wait` caller) escapes it.

That is a *prediction*, not a finding. The trace decides. H3 remains live and
independent.

## Instrumented arms (building)

Both carry `CONFIG_STARTUP_TACIT=y` and, critically,
`CONFIG_MICROROS_NODES/PUBLISHERS="4"` — at 2 the third
`rclc_node_init_default` fails on a full static pool and that network silently
never runs, which is what invalidated the earlier F2 3-net arms
(`res_cfg{Q,3x}/uartlog:32`).

* `cfg3xT` — 3 rclc executors, 3 nodes. Should reproduce the serialization.
* `cfgBT`  — fused (1 executor, 1 timer). Control that is known to work.

The first question the trace answers is not "which hypothesis" but **where the
stalled hart's PC actually is**. Everything above is a prior.


---

# VERDICT (2026-09-04)

**It is CPU starvation, not middleware blocking.** Every hypothesis in the
register above is refuted, including the one I called leading.

The trace settled it in one step. Hart 2's stream decoded to **130,426,009
instructions**, and its hottest basic blocks resolve to `mb_conv2d_s8_tiled_direct`
and `mb_requant_i32m4` — **dronet's own conv kernel**. Hart 2 was never wedged in
`rmw_wait`, never in `uxr_run_session_timeout`, never in `k_msgq_put`. It was
busy doing dronet's work and the scheduler was never entered. That is why every
"is the middleware stuck" theory — including the May conclusion, and mine —
looked plausible and was wrong.

Three facts compose:

1. `rmw_wait` takes its `available_contexts == 0` branch on **every** call,
   because `need_to_be_ran` is only ever set from subscriptions, services and
   clients, and this harness has zero of all three. That branch calls
   `uxr_run_session_timeout(session, 0)` — **non-blocking** — and returns. So
   `rcl_wait` never waits.
2. `rclc_executor_spin` is `while (true) { spin_some(); }`, so it becomes a hot
   loop that never yields the CPU.
3. dronet and mlp_control are both pinned to **hart 2** at equal priority
   `K_PRIO_PREEMPT(1)`, and **`CONFIG_TIMESLICING` was unset** (Zephyr default
   `n`). A preemptive scheduler does not preempt an equal-priority thread.

Whichever executor starts first owns hart 2 forever. This explains every
recorded observation at once: 2-net works (nothing shares a hart); "a third
executor *or* a third timer" (either puts a second runnable entity on the
shared hart); FUSE_BC works (one thread, both graphs in one callback); NORCLC
works (bypasses the spinning `rmw_wait`); and Spike reproduces it (software,
not bus contention).

## Fix — guest-side only, no third-party patch

    CONFIG_MICROROS_NODES="4" / _PUBLISHERS="4"   so the third node exists at all
    CONFIG_TIMESLICING=y, TIMESLICE_SIZE=1, TIMESLICE_PRIORITY=0
    CONFIG_RISCV_V_DECOUPLED_LAZY=y

**Measured: mlp_control 0 → 14 iterations**, yolov8_nano completes (319,774
cycles). `experiments/microros/logs/tacit_run_cfg3xG.log`.

## One fix tried and reverted

Making `rmw_wait`'s metatraffic branch honour the caller's timeout **hangs**: it
propagates `UXR_TIMEOUT_INF` into `uxr_run_session_timeout`, which blocks
forever. The upstream `0` is there for that reason. Job 812 wedged at 24 minutes
against a ~7 minute healthy run. The guest-side fix alone is sufficient, so the
vendored file is left untouched — which is also the better outcome for
upstreamability.

## The follow-on fault: root-caused (2026-09-04)

Preempting the RVV conv kernel faulted, because dronet's executor had never
once been preempted on hart 2 — the save/restore path was never exercised
there. **It is not lost vector state. It is exactly one scalar GPR clobbered
by a trap taken mid-vector-kernel**, and it is a Saturn defect this repo had
already root-caused and mitigated everywhere *except* here.

### Why it is scalar, not vector

Three faults, three builds, all in `mb_conv2d_s8_tiled_direct`, all on a
vector **memory** op whose base/stride register holds a value the kernel's own
arithmetic provably cannot compute:

| build | faulting insn | mtval | why it cannot be legitimate |
|---|---|---|---|
| cfg3xF (eager V) | `vsse8.v v2,(a7),a0` | `0x3d02e67d70` | `a7 = op = output + (n*OC+oc_base)*OH*OW + …`; needs `(n*OC+oc_base) = 82,874,420`, but the enclosing `while (oc_base < oc_end)` bounds it to ≤ OC ≤ 256 |
| cfg3xG (decoupled-lazy) | `vle8.v v1,(t3)` | `0x480000000` | DRAM_BASE + exactly 2^34 — only bits 31 and 34 set; no weight offset has that shape |
| cfg3xH (= cfg3xG, TACIT off) | store | `0x7bfdfbfbf8` | not in DRAM at all; bytes `f8 fb fb fd 7b` read as int8 tensor data sitting in a scalar register |

A stale `vstart` cannot explain any of them: `vstart` is bounded by VLMAX (256
for e8m8 at VLEN=256), so it can move an address by at most 256 bytes — not
2^34. Every "V state was lost" story dies on that bound.

### The known defect

`experiments/kernel_opt_log.jsonl`, entries `convfix-002-deterministic` …
`convfix-008-harness-irq-guard` (2026-08-29), and the comment block at
`modelblaster/harness/src/main.c:68-95`:

> a trap taken while a vector kernel is executing can come back with
> **EXACTLY ONE scalar register corrupted**. It is deterministic and
> bit-reproducible, and it is invisible on spike.

`convfix-005` narrowed the trigger to vector instructions issued into
Saturn's *decoupled* vector unit from the trap path while the interrupted
thread's own vector memop is still in flight (stubbing `z_riscv_vstate_save`
alone made the fault vanish, fq job 287).

### The fix does NOT belong in `arch/riscv/core/v.c`

That patch was already written, hardware-proven and then **reverted**:
`convfix-006` (fq job 289) removed the ISR-entry V save and replaced the exit
restore with `li t0,0x400 ; csrs mstatus,t0`, and it passed — but
`convfix-007` then showed the hazard **survives** it (same binary, shape 198,
`vsse16.v v4,(a3),a6` with stride `a6=0x5401` where the correct value is 40).
It is trigger removal, not a cure, and it perturbs a component every model in
the tree links.

The mitigation is a **harness dispatch guard**, and every other harness
already had one — `modelblaster/harness/src/main.c`
(`MODELBLASTER_MASK_IRQ_DURING_RUN`) and the generated scheduled harness
(`XPURT_DISPATCH_IRQ_GUARD` in `pipeline/generate_xpurt_main.py`).
`harness_microros` is the one that never got it, because until
`CONFIG_TIMESLICING` was turned on the dronet executor monopolised hart 2 and
the hazard could not fire.

Added here as `MICROROS_DISPATCH_GUARD` (default 1): `k_sched_lock()` /
`k_sched_unlock()` around each single dispatch call in `run_graph_b` and
`run_graph_c`. **Not `irq_lock()`** — `include/zephyr/irq.h:259` redefines it
to `z_smp_global_lock()` under `CONFIG_SMP`, which serialises the harts
(measured, FPGA job 375). `MICROROS_DISPATCH_GUARD_IRQ` escalates to
`arch_irq_lock()`, a genuine per-hart `mstatus.MIE` clear, for the residual
bare-tick hazard of `convfix-007`.

`k_sched_lock()` alone is **not** enough, and this was measured, not assumed:
cfg3xM (guard on nets B and C, net A untouched) faults again at the same
`vsse8.v v2,(a7),a4` / kernels.c:554 with `mtval=0x529f7725e4` — while the
other saved registers in the same frame are valid DRAM pointers
(`a0=0x807e2de6`). That is `convfix-007`'s residual hazard: a **bare timer
tick** mid-kernel is sufficient, and `k_sched_lock()` does not mask
interrupts. `MICROROS_DISPATCH_GUARD_IRQ=1` is therefore the shipped default.

An earlier arm, cfg3xJ, ran 600 s with no `mcause` and looked like a pass —
it was not. It had also swapped net A's whole-loop `irq_lock()` for the
per-dispatch guard, and it stalled at session bring-up with only 2 of 3
broker sessions up, so dronet's dispatches may never have run at all. "No
fault" from a run that never reached the faulting code is not evidence.
Recorded here so it is not re-derived.

### Acceptance run

**cfg3xN, fq job 847 — CLEAN.** Same Config-B pinning that faults without the
guard (yolov8_nano → hart 0 gemmini_q31, dronet **and** mlp_control → hart 2
rvv, broker → hart 3):

```
[yolov8_nano] done — total iters=1,  last wall_cycles=319562
[mlp_control] done — total iters=14, last wall_cycles=582
[dronet]      done — total iters=8,  last wall_cycles=8256
=== TRACE_MAGIC_AUDIT: 0/413 slots missing ROS_TRACE_MAGIC ===
=== MODELBLASTER_ROS_TRACE_END (skipped=0 corrupted) ===
*** PASSED *** after 15077837567 cycles
```

413 trace rows, none corrupted: dronet 168, yolov8_nano 147, mlp_control 98.
All three networks genuinely dispatch, and dronet's rows interleave with
yolov8's in the trace rather than serialising behind them. Compare the
pre-fix baseline, where mlp_control completed **zero** dispatches.

### Two latent defects found in the decoupled-lazy V path (not the cause here)

Both are real and both are live with `CONFIG_RISCV_V_DECOUPLED_LAZY=y`:

1. `arch/riscv/core/isr.S:246-248` routes only opcode `0x57` (OP-V) to
   `z_riscv_v_trap`. Vector **load/store** are opcodes `0x07`/`0x27`, so a
   thread that resumes with `VS=OFF` and whose first V instruction is a
   `vle8.v` is misrouted into `z_riscv_fpu_trap`, which grants FS and not VS
   — the retry then falls to `no_fp` and dies as a spurious illegal
   instruction. Discriminator if fixed: for LOAD-FP/STORE-FP, funct3 ∈
   {0,5,6,7} is vector, {1,2,3,4} is scalar FP.
2. `arch/riscv/core/v.c:528-530` (`z_riscv_v_exit_exc`) ORs
   `_current_cpu->arch.v_state` into `esf->mstatus` on the "owner ==
   _current" path. `z_riscv_v_disable()` only ever writes `v_state` when
   `VS != OFF`, so it can still be 0 — the thread then resumes with `VS=OFF`
   while `v_access_allowed()` claimed access was granted, walking straight
   into defect 1.

### The canary cited as evidence for this config does not exist

`prj.conf:54-57` justifies `CONFIG_RISCV_V_DECOUPLED_LAZY=y` with "validated
by FireSim `v_save_smoke` (0/8 mismatches per thread)". **There is no
`v_save_smoke`** — no directory, no source, no build log, and nothing in the
git history of any of the four nested repos. The string occurs in exactly
three places, all prose (`prj.conf:56`, `modelblaster/README.md:221`,
`arch/riscv/Kconfig.isa:156`), and the commit that wrote the claim into
prj.conf is `b4c47e1cdb0` — the same commit that added the FUSE_BC workaround.
The justification and the workaround it justifies were authored together, with
no test behind either.

The nearest surviving analogue, `samples/test_mt_rvv/src/main.c`, switches via
**`k_yield()`** — a cooperative switch at a scalar instruction boundary, one
vector register, 4 bytes, one thread per hart, no `CONFIG_TIMESLICING`. It
cannot produce a non-zero `vstart` or two RVV threads contending for one hart,
so it could not have caught this even if it had been run.

Also refuted while here: the premise at `v.c:131-136` that "Saturn does not
flip MSTATUS.VS=DIRTY for every V op". The RTL in this tree contradicts it —
`rocket-chip/.../RocketCore.scala:899` drives
`v.set_vs_dirty := wb_valid && wb_ctrl.vec` and `CSR.scala:1193-1199` sets
`reg_mstatus.vs := 3.U` on it, at full 2-bit precision (`CSR.scala:1671`).
The unconditional full save that comment justifies is itself what perturbs
Saturn.

### Still open

* The two latent decoupled-lazy V defects above are unfixed. Neither caused
  this fault, but both are live with `CONFIG_RISCV_V_DECOUPLED_LAZY=y`.
* `MICROROS_DISPATCH_GUARD_IRQ` masks this hart's interrupts for one dispatch.
  yolov8_nano's longest single dispatch is ~28 k cycles (~0.5 ms at 60 MHz),
  so at most one tick is delayed and the RISC-V `mtimecmp` driver announces
  elapsed time rather than losing it — but a model with a much longer single
  kernel would want the guard pushed inside the kernel's tile loop instead.
* The underlying Saturn defect is untouched. This is containment, the same
  containment `harness/` and `generate_xpurt_main.py` already ship. A real
  fix is RTL-side.
