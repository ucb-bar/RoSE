# FPGA validation set for the ten-solver scheduler bench

Everything in `../results/` is a **predicted** makespan from the xpu-rt cost
model. This directory turns those solves into `harness_xpurt` ELFs so the
predictions can be checked against `f2_quad_hetero_norose_tacit_q31_60mhz`.

**Nothing here submits to the FPGA queue.** `submit_cell.sh` exists and is
correct, and is not called by the build pass.

## What is built

One ELF per **distinct schedule**, not per solve. All 1056 (arm, workload,
solver) rows are represented; `manifest.json` maps every row to the ELF that
realises it, so a row that shares an artifact still has an executed result
attributable to it.

Dedupe is on the schedule's **content** — every dispatch's target, start,
duration and dependencies (`emit_schedule._sched_hash`) — and never on the
objective. On `wl_sweep/networks_depth_nav_gempair` eight solvers all report
makespan 592.4161 and yet form **four** distinct assignments; collapsing those
on the number would have discarded three quarters of that cell's coverage.
Conversely `greedy` and `greedy_periodic` return bit-identical schedules on
three cells, so those genuinely are one ELF.

## Why the schedules are re-solved here rather than reused

The sweep never saved its schedules — `sweep10_runner` records the objective,
the miss count and the validation audit, and throws `(t, alpha)` away. So they
have to be re-solved, and the re-solve has to be the *same* solve:

* `scripts/run_xpurt_schedule.py` cannot produce them. It exposes four solvers
  (milp, greedy, greedy_periodic, decomposed); ten of the twelve entries here —
  and every winner — live in the standalone XPU-RT tree
  (`metaheuristics.py`, `cpsat_scheduler.py`).
* It also wraps greedy in a periodic-instance refinement loop and post-trims
  the result, where the sweep builds the workload once at the spec's own
  `num_instances` and gives that same instance to every solver.

`emit_schedule.py` therefore uses `sweep10_runner`'s own `build()` and
`make_solver()` and feeds `(t, alpha)` to `postprocessing.output_scheduled_json`
— the same emitter `run_xpurt_schedule.py` calls, and a file that is identical
in both trees. The only thing that is new is the emit step.

## Fidelity: what reproduces and what does not

Re-solving `greedy` locally reproduces the recorded objective **to the last
recorded digit on 72 of the 88 cells**. The 16 that differ do so because the
workload specs were retuned *after* the sweep ran (`soc/sw/xpu-rt` commits
`db300308` and `1e86a8f5`, 2026-09-05 23:50 and 09-06 01:26; the sweep's own
solves finished 09-05 22:43):

* **`tight_loop`, all 8 workload-arms: the workload is no longer the same
  problem.** `dronet_sa`'s period went from ~2.3 ms to 222.162 ms, so greedy
  now returns **0 missed windows at ~700 ms** where the sweep recorded 84/140
  misses at ~85-100 ms. §5 of `../README.md` ("infeasible by construction")
  describes the pre-retune spec and does not apply to the tree as it now
  stands. Winner identity from `all_results.json` is meaningless for these
  eight cells; they are built from the whole 12-solver set like every other
  cell, and the ELF's own emit-time prediction is what its FPGA run validates.
* **8 further cells drift by −2.5% to +1.1%** (bimodal base-hetero and all four
  shard, control_mix shard-rvvpair, perception_heavy shard-rvvpair,
  vint_intro base-rvvpair). Small relative to the +8-10% effect being measured,
  but recorded per row in `manifest.json` all the same.

Each manifest row therefore carries **both** numbers: `emit_objective` (the
prediction the ELF actually encodes — this is the one an FPGA run validates)
and `sweep10_objective` (the published row), plus `reproduces`.

CP-SAT is the other non-reproducible case, and it is inherent, not drift:
`num_search_workers=8` makes it nondeterministic (`../README.md` caveat 2
measures cold-CP-SAT seed spread at 2.73% mean / 14.90% max). A re-solve is a
fresh draw, so for CP-SAT rows `emit_objective` is the number to compare
against hardware and `sweep10_objective` is a reference only.

## Flow

```bash
source env.sh                                   # code root, data root, ortools venv
python3 pick_winners.py --results ../results/all_results.json --out winners_recorded.json
python3 plan.py jobs   --stage winners --out jobs_winners.json     # or greedy / rest / all
python3 emit_all.py    --jobs jobs_winners.json --outdir schedules --out emitted_t1.json
python3 plan.py dedupe --emitted emitted_*.json --out elf_plan.json   # + manifest.json
python3 build_all.py   --plan elf_plan.json --out built.json --workers 6 --skip-existing
python3 summarize.py
```

`build_cell.sh` is `experiments/workload_gen/build_workload.sh` with two forced
changes:

1. **The schedule path is passed in, not derived.** `build_workload.sh` computes
   `schedules/scheduled_<basename>_<solver>_profiled.json`, and the base and
   shard arms share a basename — so the two arms overwrite each other's
   schedule. Survivable when you solve-then-build one cell at a time (what
   `run_wl_sweep.sh` does); not survivable with 1056 schedules emitted up front.
2. **Per-worker example dir.** `xpurt_demo_armB/run.sh` derives `BUILD_DIR` from
   the example dir it lives in, so N concurrent builds need N example dirs
   (`examples/xpurt_s10_w<k>`, driven by the one-line `XPURT_EXAMPLE_DIR` hook in
   `examples/xpurt_s10/run.sh`). One shared build dir means concurrent builds
   overwrite each other's objects and no ELF can be traced to its schedule.

## Gates

* `harness_xpurt`, never `harness/` — the single-model harness does not boot on
  this bitstream, and only `harness_xpurt` places workers onto harts by kind.
  Harts 0-1 are Rocket+Gemmini with **no vector unit**, harts 2-3 Saturn with
  **no Gemmini** (`cores/chipyard_quad_hetero_gemmini_q31.json`), so a
  misplaced kernel takes an illegal-instruction trap. All four machine
  configurations ride the same quad bitstream: `gempair` uses CPU_P#0-1 only,
  `rvvpair` CPU_E#0-1 only, `hetero` one of each, `quad` all four.
* `MB_DRIFT_ATOL=2` is a gate, not a tolerance knob — unset, the gemmini
  backend silently falls back to a software im2col 3.45x slower than the one
  the schedule was costed against.
* `XPURT_TRACE=1` — without the per-dispatch trace block there is no
  predicted-vs-actual, only a makespan.
* **ELF freshness, not existence.** `build_cell.sh` requires the ELF to
  post-date a stamp taken immediately before the build, and records the
  schedule's dispatch count in its log. `submit_cell.sh` re-checks the
  `xpurt-runner: schedule=<tag>` string the running binary prints against the
  tag it submitted, because `fq` copies results from the run host's
  `sim_slot_*/`, which survives between jobs — a cell can otherwise silently
  collect the previous job's uartlog.
* Trace cycle columns are `k_cycle_get_64()` ticks at 1 MHz: 1 tick = 1 us.
  Not 60.

Alias flattening (`xpu-rt/scripts/flatten_schedule_aliases.py`) is **not**
needed here: it remaps rate-group `(alias, instance)` pairs, which
`gen_random_workload` emits and `mk_workloads.py` does not — every network in a
wl_sweep spec is its own entry. The 88 ELFs of the earlier greedy sweep built
and ran through this same path without it.

## Layout

```
env.sh              code root / data root / the ortools venv CP-SAT needs
emit_schedule.py    one (arm, workload, solver) -> scheduled_*.json + content hash
emit_all.py         two-pool fan-out (CP-SAT wants 8 threads, the rest want 1)
pick_winners.py     feasible-first winner per cell from all_results.json
plan.py             job lists; content-hash dedupe -> elf_plan.json + manifest.json
build_cell.sh       one schedule -> one harness_xpurt ELF (freshness-gated)
build_all.py        N builds at once over N per-worker example dirs
summarize.py        coverage / dedupe / size tables
submit_cell.sh      dispatch ONE built ELF to fq (never called by the build pass)
schedules/          the emitted schedule JSONs (gitignored, ~210 MB)
logs/               per-build and per-emit logs
../elf/             the ELFs (gitignored, ~15 GB)
```
