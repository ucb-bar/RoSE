# kcov — curated-kernel coverage for the 2-backend (rvv + gemmini_q31) sweep

Closes the curated-kernel gaps that made the `gemmini_q31` arm of a sharding
sweep run at scalar-reference speed on ops the `rvv` arm has curated kernels
for. All kernels land in `modelblaster/kernels/gemmini_q31/`; their algorithms
are registered on the op's `KernelSpec` in
`modelblaster/pipeline/reference_kernels.py`.

## Reproducing a measurement

Every number in `kernel_opt_log.jsonl` under experiment `kcov_*` comes from
one invocation of:

    experiments/kcov/run_arm.sh <tag> <exdir> <target> <quant>

which activates the in-tree zephyr conda env + SDK, sets `PYTHONPATH`, and runs
`modelblaster/examples/<exdir>/run.sh` with `PROFILE_OUT_ROOT` set. It writes
`experiments/kcov/logs/<tag>.log` (full build + per-op verify + the
`MODELBLASTER_PROFILE` table) and `experiments/kcov/prof/<tag>.picks.json`
(which algorithm each op actually got — the ONLY reliable check that a curated
kernel was selected rather than silently skipped).

Env knobs used:

  MB_DRIFT_ATOL=2      the coordinator's gemmini arm setting; lets
                       conv2d_s8/gemmini_tiled_conv through. Set for the
                       PERFORMANCE arms so the conv matches the sweep's.
  (unset)              the tight gate. Used for the bit-exactness runs
                       (`*_bitexact`), where conv falls back to
                       gemmini_im2col_full_C and the whole model must read
                       max_abs_err=0.
  GLOBAL_CURATED_DIR=  empty -> every op is spec.reference_impl. Used once, to
                       establish that ViNT's own reference build already
                       misses the PyTorch golden by 0.3279.

## Isolated example trees

`examples/{mlp_control,dronet,yolov8_nano,vint}_kcov/` (and `vint_kcovref`) are
copies of the corresponding `examples/<model>/` IR, with their own run.sh. They
exist because the coordinator runs sharding sweeps out of the shared trees and
`_run_lib.sh` skips extract when graph.json is present, so a re-extract in a
shared tree would corrupt an in-flight sweep. Two runs may share a tree only if
their TARGET differs (build/, cache/ and generated/ are all per-target) --
`generated/profile.csv` is NOT, so read the profile table out of the log.

## Files

  run_arm.sh            spike arm: build + per-op curated verify + profile
  fpga_build.sh         one single-model FireSim ELF, stashed under elf/
  fpga_submit.sh        ship an ELF to the AWS manager, submit one fq job
  f16_cvt_selftest.c    exhaustive host validation of the inline fp16 <-> fp32
                        conversions the soft_f16 kernels are built on
                        (gcc -O2 -o f16 f16_cvt_selftest.c && ./f16)
  kernels_pre_kcov/     copy of modelblaster/kernels/ WITHOUT this work's
                        files -- the A arm of the FPGA A/B
