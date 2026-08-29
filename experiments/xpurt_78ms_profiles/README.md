# Profile data behind the 78 ms greedy XPU-RT makespan

3-model workload (mlp_control + dronet + yolov8_nano) scheduled onto a 2-core SoC —
hart 0 = Gemmini (CPU_P), hart 1 = Saturn RVV only (CPU_E) — and run on AWS F2 FireSim.

    predicted makespan   78.631 ms
    actual makespan      78.397 ms      ratio 1.00
    fq job 234, hw-config f2_dual_small_norose_tacit_q31_60mhz, DONE rc=0
    banner: entries=295 kinds=2, worker[0] kind=gemmini_q31 pinned_hart=0,
                                worker[1] kind=rvv pinned_hart=1

## profiles/
The six per-op profiles the scheduler consumed, under target tag
`firesim_f2_rocket_saturn`. Directory structure is preserved because
`profile_loader.find_profile_csv` keys on it:
`gen/profile/<hw>/<target>/<model>/<basename>/<input_tag>/<topo_tag>/results.csv`

    backend         model          rows   total (M cycles)
    gemmini_q31     yolov8_nano     155   315.5
    gemmini_q31     dronet           21    14.1
    gemmini_q31     mlp_control       7     0.5
    V256D128_rvv    yolov8_nano     155   165.2
    V256D128_rvv    dronet           21     7.5
    V256D128_rvv    mlp_control       7     0.6

Generated with the curated kernel set as of commit d201114 + the RVV/Gemmini
campaign work — i.e. BEFORE the hardware-im2col conv landed (d9b279c) and before
the flag-gated int8-drain path (f10d0b6). Both change conv2d_s8 materially, so
these numbers are a snapshot, not the current tree.

## schedule/
CAVEAT ON PROVENANCE: the original schedule file was overwritten in a later
solver sweep, because `run_xpurt_schedule.py` names outputs
`{spec}{_solver}{_profiled}` and encodes no other flags, so runs differing only
by a flag collide. The file here is an archived re-run of the identical
configuration (`--solver greedy --profiled`, same spec, same profiles) which
reproduced makespan 78.631 ms, 295 operations, 8 mlp_control / 4 dronet
instances — identical to the run that produced the hardware result above.

## spec/, dispatch_graphs/
The workload spec and the per-network dispatch graphs it references.
Instance counts in the spec are 16 mlp / 8 dronet; `prune_periodic` reduces them
to the 8 / 4 actually scheduled, derived from yolov8_nano's completion time.

## run/, plots/
Raw F2 uartlog, the per-dispatch trace CSV, and the predicted-vs-actual Gantt.

## Reproducing
    cd soc/sw/xpu-rt
    /scratch2/dima/miniforge3/envs/xpurt/bin/python scripts/run_xpurt_schedule.py \
      --networks-json data/toplevel/networks_mlp10_dronet20_yolov8_f2_q31profile.json \
      --solver greedy --profiled
Use the xpurt interpreter, not the zephyr one — the scheduler needs cvxpy, which
only that env has.
