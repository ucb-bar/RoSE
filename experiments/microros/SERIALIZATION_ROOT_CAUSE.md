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

## Still open, and it blocks using this as a ROS baseline

Preempting the RVV conv kernel now faults, because dronet's executor had never
once been preempted on hart 2 — the save/restore path was never exercised there.

| build | V mode | outcome |
|---|---|---|
| cfg3xF | eager (`V_LAZY=n`) | `vsse8.v v2,(a7),a0`, mcause=7, mtval=0x3d02e67d70 |
| cfg3xG | decoupled-lazy | 14 mlp iterations, then `vle8.v v1,(t3)` at kernels.c:509, mcause=5, mtval=0x480000000 |

`0x480000000` is DRAM base + 2^34 in a **scalar** base register, so this is
context corruption across a preemption, not the vector-state bug the eager mode
hit. It is a separate latent defect that the starvation fix uncovered.
