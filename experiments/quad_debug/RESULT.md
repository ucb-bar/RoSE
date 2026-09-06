# Hardware result: removing the trap-path V hooks is NOT the fix

Two FPGA jobs, `f2_quad_hetero_norose_tacit_q31_60mhz`, submitted 2026-09-06.
Both ran the **exact ELF that faults**, binary-patched by
`patch_vstate_trap_hooks.py` so that isr.S's trap-entry `z_riscv_vstate_save`
and trap-exit `z_riscv_vstate_restore` both return immediately
(`switch.S`'s calls to the `*_thread` variants are untouched, so per-thread V
context switching still happens). No rebuild -- the shared modelblaster build
tree was in use by another campaign.

The patch is numerically inert in software: the patched
`wl_control_mix_quad_greedy_base` produces byte-identical
`MODELBLASTER_VERIFY` lines to the unpatched one under `spike -p4`.
`mstatus` in the fault frames confirms the patch took effect on hardware --
`0x...7e80` (VS=Dirty) with the hooks gone against `0x...7c80` (VS=Clean, i.e.
forced by the trap-entry save) without.

| cell | fq job | before | after the V-hook stub |
|---|---|---|---|
| `wl_control_mix_quad_greedy_base` | 911 | `mcause 5`, 0 trace rows | **no fault**, 236 trace rows, all 3 models verified |
| `wl_scale_ladder_quad_greedy_shard` | 912 | `mcause 7`, 0 trace rows | **still `mcause 7`**, 0 trace rows |

## The negative is the important one

`wl_scale_ladder_quad_greedy_shard` with the hooks gone:

     mcause: 7, Store/AMO access fault
      mtval: 64a46e26a2
         a3: 4   a4: -23   a5: 32   a6: 127   t5: 31   t6: -2
         a7: 00000064a46e26a2
         sp: 00000000870eacb0
       mepc: 000000008001e5f0   -> mb_conv2d_s8_tiled_direct,
                                   dronet_se_wls .../rvv/kernels.c:637
                                   = vsse8.v v2,(a7),a0   (MB_STORE_PIX)

against the unpatched run of the same binary:

     mcause: 7, mtval 29e708572c
         a3: 4   a4: -23   a5: 32   a6: 127   t5: 31   t6: -2
         a7: 00000029e708572c
         sp: 00000000870eacb0
       mepc: 000000008002f34c   -> mb_conv2d_s8_tiled_direct,
                                   dronet_sg_wls .../rvv/kernels.c:637
                                   = the same vsse8.v v2,(a7),a0

Same kernel, same source line, same instruction, same `sp`, byte-identical
neighbouring register profile, `a7` garbage both times. Removing the V hooks
moved *which* dronet rung was executing when it happened and nothing else.

That reproduces `convfix-007` (`experiments/kernel_opt_log.jsonl`) on the xpurt
harness at four harts: with the ISR-entry V save removed and no thread switch
involved, a bare interrupt mid-kernel still returns with one scalar register
wrong. **The corrupting agent is the trap, not the V-save code the trap runs.**

So `zephyr_isr_v_hooks.patch` must not be shipped as the mitigation. The
mitigation is `xpurt_dispatch_guard_arch_irq_lock.patch` -- don't take the
interrupt during the kernel -- which is the same change
`harness_microros/src/main.c` already ships and which passed on this bitstream
as fq 847 (`microros-3net-rvv-preempt-005-ACCEPTANCE-clean-3net`), against fq
837 which failed with `k_sched_lock()` only.

## About job 911's numerics

`wl_control_mix_quad_greedy_base` stopped faulting but reported
`dronet_se max_abs_err=17` where hetero and gempair report 2 and rvvpair
reports 0 (`yolov8_nano_sc` was `max_abs_err=0` over 48384 elements and
`mlp_control_sd` matched hetero exactly). Two readings, and this experiment
does not separate them:

1. the corruption still happened, it just landed on data instead of on a
   pointer -- which is what the scale_ladder result says is going on; or
2. the stub is genuinely unsafe. It is only V-correct if no ISR touches
   `v0..v31`. `CONFIG_RISCV_V_KERNEL_ONLY=y` strips `v` from the global
   `-march`, but `arch/riscv/core/v.c:110` records a Saturn fault "inside
   picolibc's `vse64.v` memcpy", so vector code may exist on an ISR-reachable
   path in this tree.

Either way it is another reason not to ship the stub. It does not affect the
diagnosis: `n=2` for dronet, and the run that matters for the mechanism is 912.

## Reproducing

    python3 experiments/quad_debug/patch_vstate_trap_hooks.py \
        experiments/wl_sweep/elf/<tag>.elf experiments/quad_debug/elf/<tag>_novtrap.elf
    bash experiments/quad_debug/submit.sh <tag> experiments/quad_debug/elf/<tag>_novtrap.elf
    python3 experiments/quad_debug/verify.py

Patched ELFs for all seven failing cells are in `elf/`. Only the two above were
run; the lanes belong to another campaign.
