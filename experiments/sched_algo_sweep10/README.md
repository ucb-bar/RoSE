# Ten-solver scheduler bench over all 44 wl_sweep workloads, both arms

**Supersedes the four-solver sweep in `experiments/sched_algo_sweep/`.** That run
compared only the four solvers RoSE's `run_xpurt_schedule.py` exposes (milp,
greedy, greedy_periodic, decomposed) and concluded "decomposed wins, +2.18% over
greedy". Six more solvers exist in the standalone XPU-RT tree and were never in
the comparison. With all ten in, **that conclusion does not survive: `decomposed`
is 4.9% WORSE than greedy on average**, and the top of the table is 8-10% better
than greedy, not 2%.

- 44 workloads x 2 arms (`wl_sweep`, `wl_sweep_shard`) x 10 solvers = **880 solves**
- plus 160 seed-repeat solves and 88 solves of one supplementary variant = 1128 total
- AWS EC2 `i-02251dea96ee5f6be`, c7i.24xlarge, 96 vCPU / 185 GB. **Stopped** after
  the results were copied back.
- Every number here is a **predicted** makespan from the xpu-rt cost model. No FPGA.

## 1. Headline: which solver to use

Ranked feasible-first. `usable` counts workload-arms where the solver returned a
schedule with **zero missed periodic windows** (and greedy also did, so the
comparison is defined); `tight_loop` is excluded throughout because it is
infeasible by construction (§5).

| solver | usable | mean impr vs greedy | median | worst case | mean wall |
|---|---|---|---|---|---|
| **cpsat:warmbest** (supplementary) | 78/78 | **+9.75%** | +9.06% | −0.03% | 27.5 s |
| **pso** | 78/78 | **+9.10%** | +8.63% | +0.00% | 12.2 s |
| **sa** | 78/78 | +8.75% | +8.55% | +0.00% | 15.3 s |
| **cpsat:warm** | 78/78 | +8.30% | +8.58% | −28.11% | 29.2 s |
| **cheap_portfolio** (supplementary) | 78/78 | +7.63% | +6.59% | +0.00% | **0.77 s** |
| **cpsat** | 78/78 | +6.03% | +7.42% | −38.90% | 34.4 s |
| heft_edf | 78/78 | +2.37% | +6.29% | −119.07% | 0.46 s |
| greedy | 78/78 | 0 | 0 | 0 | 0.06 s |
| decomposed | 77/78 | **−4.92%** | 0.00% | −119.07% | 0.07 s |
| greedy_reserved | **57**/78 | −0.10% | 0.00% | −8.39% | 0.07 s |
| greedy_periodic | **56**/78 | +1.04% | 0.00% | −5.73% | 0.06 s |
| heft | **50**/78 | −1.66% | +3.60% | −119.07% | 0.05 s |

Two supplementary arms are in this table, clearly marked. They are not among the
ten; the ten-solver data motivated them (§3) and they are the actual
recommendation, so hiding them would be the wrong call:

- **`cheap_portfolio`** — run all six sub-second heuristics and keep the best
  *feasible* one. A deployable policy, costed at the sum of all six walls.
- **`cpsat:warmbest`** — CP-SAT hinted from that portfolio instead of from
  `heft_edf` unconditionally, which is what `cpsat:warm` does.

### Does "decomposed wins" survive? No.

`decomposed` places **9th of 12**. It ties greedy exactly on the median workload
(it returns a bit-identical objective to greedy on 43 of 80 workload-arms) and its mean is dragged to −4.92% by
`depth_contended`, where it is 46.6% worse than greedy. It also has the single
worst deadline behaviour on `tight_loop`, missing **every** window (252/252 base,
308/308 shard) where every other heuristic misses 84/140. The prior +2.18% is
not reproducible in this harness even in sign.

The two runs are not measuring quite the same thing — the earlier sweep drove
`run_xpurt_schedule.py`, which wraps each solver in a periodic-instance
refinement loop that rebuilds the workload between passes, and it excluded cells
on op-count mismatch. This bench builds each workload once at the spec's own
`num_instances` and every solver schedules that same instance, which is the
comparison the question actually asks. But the direction of the disagreement is
not a harness artefact: `decomposed` is a greedy variant, and on the 80
workload-arms here it is never more than 4.8% better than greedy and is
sometimes 47% worse.

### Deadline misses are where the ranking is really decided

| solver | workload-arms with ≥1 miss (of 80, ex tight_loop) | total missed windows |
|---|---|---|
| heft_edf, pso, sa, cpsat, cpsat:warm, cpsat:warmbest, cheap_portfolio | **0** | 0 |
| greedy | 2 | 42 |
| decomposed | 1 | 130 |
| greedy_reserved | 23 | 2398 |
| greedy_periodic | 24 | 2285 |
| heft | **30** | 2930 |

`heft` is the cautionary case the brief called out, and it generalises: it has
the best raw makespan on 13 workload-arms but misses windows on 30 of 80. On
`control_mix_gempair` it reproduces exactly the reported 54.07 with 14 misses.
`greedy_periodic` and `greedy_reserved` are worse still — they trade windows for
makespan on nearly a third of the set, which is why their small positive means
(+1.04%, −0.10%) are meaningless: those means are computed over only the ~56
cases where they stayed feasible.

## 2. Quality vs cost

`cheap_portfolio` is the number that matters for the default path:

| | wall | mean impr vs greedy | % of the best available gain |
|---|---|---|---|
| greedy | 0.06 s | 0 | 0% |
| heft_edf alone | 0.46 s | +2.37% | 24% |
| **cheap_portfolio** | **0.77 s** | **+7.63%** | **78%** |
| pso | 12.2 s | +9.10% | 93% |
| cpsat:warm | 29.2 s | +8.30% | 85% |
| cpsat:warmbest | 27.5 s | +9.75% | 100% |

**0.77 s buys 78% of what 27.5 s buys.** The remaining 2.1 points cost 36x more
wall-clock. Paired against `cpsat:warmbest` the portfolio's median gap is
−0.23%, i.e. on a typical workload they produce the same schedule; the 2.1-point
mean gap lives entirely in `depth_contended` and `saturation`.

**Recommendation.**

- **Default path: `cheap_portfolio`.** Sub-second, feasible on 80/80, never worse
  than greedy, and within a quarter of a percent of the best solver on the median
  workload. Critically, the portfolio's *feasibility filter* is what makes it
  safe — `heft` supplies its answer 33 times and `heft_edf` 31 times, and both of
  those are unsafe on their own.
- **Offline / build-time: `cpsat:warmbest`.** Worth the ~30 s only when the
  schedule is baked once and reused, and mainly on `depth_contended`,
  `saturation` and `control_mix`, where it is 15-20% better than greedy against
  the portfolio's 9-16%.
- **Do not ship `cpsat` cold, `heft`, `greedy_periodic`, `greedy_reserved`, or
  `decomposed`.** The first is worse than greedy on a fifth of the set; the rest
  miss deadlines.

`pso` is the honourable mention: at 12 s it is +9.10% and, like `sa`, it is
**greedy-dominating by construction** — `_heuristic_seeds` seeds the population
with heft, heft_edf and all four greedy pickers and only accepts improvements, so
its worst case over 78 workload-arms is exactly +0.00%. That structural floor is
why pso/sa never miss a window while heft, which they are seeded from, misses 30.

## 3. Where each solver is strong and weak

Full tables in `results/breakdown.md`. The headline patterns:

- **`depth_chain` and `depth_nav` have no scheduling freedom at all.** Every
  solver lands within 0.03% of greedy. Any aggregate that includes them is
  diluted by two families where the answer is "it does not matter".
- **`depth_contended` is where solvers separate.** heft_edf −41.9%, heft −42.0%,
  decomposed −46.6% against greedy; pso +14.6%, sa +14.8%, cpsat:warmbest +15.1%.
  Cold `cpsat` manages only +2.1% and `cpsat:warm` +2.6% — because `cpsat:warm`'s
  hint *is* heft_edf, and here heft_edf is 119% worse than greedy, so the hint
  drops CP-SAT into a bad basin it cannot climb out of in 60 s. That single
  observation is the whole reason `cpsat:warmbest` exists, and hinting from the
  best feasible heuristic instead takes that family from +2.6% to +15.1%.
- **`control_mix` and `saturation` are CP-SAT's families**: +18.4% and +19.8% for
  `cpsat:warmbest` vs +16.0%/+12.3% for the portfolio.
- **The shard arm is harder for CP-SAT.** Cold `cpsat` averages +10.18% on the
  base arm and only **+1.87%** on shard, where op counts run to 801: it does not
  close the gap inside 60 s. `cpsat:warmbest` holds +8.42% there.
- **The `quad` machine pair (6 combinations) breaks cold CP-SAT**: −0.41% mean,
  i.e. worse than greedy on average, while it is +6.6 to +10.7% on the other
  three pairs. The heuristics are insensitive to the pair.
- CP-SAT proves optimality on 44/80 workload-arms cold, **53/80** warm-started
  (either hint). The 27 `FEASIBLE`-only cases carry gaps of 0.12-0.52 and are
  exactly where it loses to greedy.

## 4. Failures, and the only degenerate results

**16 hard failures of 880, all the same thing**, with the real error text:

```
RuntimeError: cpsat returned INFEASIBLE with no solution
```

`cpsat` and `cpsat:warm` on all four `tight_loop` pairs in both arms (16 = 4
pairs x 2 arms x 2 CP-SAT variants). `cpsat:warmbest` fails identically on the
same 8 workload-arms. This is **correct behaviour, not a crash**: CP-SAT enforces
`end <= max_end` as a hard constraint and proves no assignment satisfies it. The
heuristics have no such notion and return a schedule with 84-308 missed windows
instead.

It is also an operational caveat: **a hard-constraint solver returns nothing at
all on an over-subscribed workload**, so anything that puts CP-SAT in the default
path needs a heuristic fallback for exactly the case where scheduling matters
most.

No other solver failed, and **no solver returned a degenerate schedule**:

```
0 of 1032 schedules have a precedence/overlap/assignment violation
```

(1032 = the 944 schedules the solvers actually returned — 880 + 88 jobs minus the
24 CP-SAT infeasibility failures — plus the 88 portfolio selections.)

That is an independent audit (`sweep10_runner.validate`), not the solver's own
report — it re-checks every returned schedule in float arithmetic for
predecessor-plus-transfer ordering, per-machine no-overlap (charging every
machine a multi-core combination occupies, not just the first), negative starts,
starts before `min_start`, and operations assigned to a combination they cannot
run on.

## 5. tight_loop, reported separately

`tight_loop` is infeasible by construction in both arms and all four pairs, as
documented: one `dronet_sa` instance's own critical path is 35.3 ms (base) /
16.9 ms (shard) against declared windows of 2.3 / 3.6 ms, because the workload
generator derives periods from a MAC-scaled estimate anchored to the default rung
while `dronet_sa`/`dronet_sb` fall back to reference scalar kernels ~61-64x
slower. Confirmed here, and quantified:

- **Every one of its operations is periodic** (252/252 base, 308/308 shard), so
  the objective degenerates to the all-operations makespan and there is no
  non-periodic makespan to trade against.
- Best case for any heuristic is 84 missed windows (base) / 140 (shard).
- `decomposed` misses **all** 252/308.
- CP-SAT proves it infeasible in all 8.

It is excluded from every aggregate above. `results/RESULTS.md` has its full
table.

## 6. Cross-check against the Sep-3 baseline

`/scratch2/dima/misc_sw/XPU-RT/scripts/solver_study/data/wl_sweep_baseline.json`
(24 of 32 workloads OK; the 8 that errored on missing `yolov8_nano_sf`/`sh`
profiles now build — all 88 workload-arms built clean here, none skipped).
Comparison in `results/crosscheck.txt`:

- **`greedy`, `greedy_periodic`, `greedy_reserved`, `decomposed`, `heft`:
  24/24 cells each reproduce exactly** (to the baseline file's 3-decimal rounding
  floor). The workload data and the build path are the same.
- **`heft_edf` differs on 16/24, always faster now** (up to −37.5%), and the
  stochastic solvers all improved: cpsat mean −14.5%, pso −7.5%, sa −5.7%,
  cpsat:warm −3.4%. This is a **real disagreement with a known cause**: the
  standalone solver tree changed between Sep 3 and now. The redundant-bound work
  in `_cpsat_solve.py` (`prec,dur,tail,load`) and the laxity-gate repair in
  `metaheuristics.heft_edf_schedule` both landed in that window. On
  `control_mix_gempair`, the brief's sampled workload, Sep-3 CP-SAT got 59.53
  after a full 60 s; the current code proves **OPTIMAL at 51.81 in 9.7 s**.
- Consequence for the brief's premise: `cpsat:warm` beating greedy by 9.2% with
  zero misses on that workload is confirmed in direction, but `heft_edf` has
  since caught up there — it now reaches 54.07 feasible, which is what plain
  `heft` scored *with 14 misses* in the old data. **These numbers are for the
  current standalone tree and are not comparable cell-by-cell with anything
  produced before ~Sep 3.**

## 7. The CP-SAT integer-rounding caveat: does not apply here

`run_xpurt_scheduler_multi.py:422` warns that `scheduler_cpsat._to_int_us`
rounds processing times to whole integers (min 1), collapsing sub-ms ops so that
"the schedule looks valid to CP-SAT but the back-projected float schedule has
overlap / precedence violations of up to ~0.5 ms".

That warning is **stale and refers to a module that no longer exists** —
`_to_int_us` appears nowhere in either tree except in that comment. The CP-SAT
backend used here is `xpu-rt/cpsat_scheduler.py`, which is a different
implementation:

- it scales by 1000, so the grid is **microseconds, not milliseconds** — 1000x
  finer than the warning describes;
- it rounds **outwards, never to nearest**: `_ceil` on durations and `min_start`,
  `_floor` on `max_end`. That makes the integer model strictly conservative —
  satisfying it *implies* satisfying the float constraint — so it cannot create
  the validity hole the warning describes, only mild pessimism.

Measured on these workloads (`scripts/` reproduce it): the smallest operation is
**0.107 µs**, and ceiling it to 1 µs inflates that one op by 834%. But
**work-weighted**, over the durations a schedule actually places, the µs grid
costs a median of **0.077%** and at most **0.365%** of total work. And
empirically it does not bite at all: **not one of the 944 returned schedules**
has a precedence or overlap violation in float arithmetic, CP-SAT's included.

## 8. Honest caveats

1. **These are predicted makespans, not measurements.** The cost model is known
   to be +37.9% off on yolov8_nano's gemmini pair. **Cross-family absolute
   comparisons are unsound** — do not read "saturation is slower than
   control_mix" out of this data. The valid comparison is *within* a
   workload-arm, where all twelve solvers see identical durations.
2. **Differences below ~0.5% are inside the noise.** Seed-repeat over
   {0,1,2} on 20 workload-arms (`results/variance.txt`): mean spread pso 0.50%
   (max 4.14%), sa 0.14% (max 0.94%), **cpsat 2.73% (max 14.90%)**, cpsat:warm
   0.27% (max 2.20%). CP-SAT with `num_search_workers=8` is not reproducible, and
   cold CP-SAT's spread is large enough that its rank relative to `heft_edf`
   should not be treated as settled. The warm start does not just improve
   CP-SAT's answer, it makes it an order of magnitude more stable.
3. **pso vs sa vs cpsat:warm is a tie on the typical workload.** Paired medians:
   pso−cpsat:warm −0.00%, sa−cpsat:warm −0.01%, pso−sa +0.00%. Their means differ
   (9.10 / 8.75 / 8.30) only because of a hard tail: pso and sa beat cpsat:warm by
   24-30% on `depth_contended`, cpsat:warm beats them by 11-27% on `saturation`
   and `control_mix`. **Do not read the mean ordering as a general ranking of
   those three.** `cpsat:warmbest` is a genuine improvement on `cpsat:warm`
   (+1.19% mean, worst case −1.51% vs −28.11%) because it removes a specific
   failure mode rather than searching better.
4. **The gap of the top group over greedy (+8 to +10%) is far outside the noise**
   and is the one conclusion that is safe.
5. `milp` is not in this study. It is in the four-solver sweep it supersedes,
   where it failed on 66 of 80 cells and averaged −4.19%; at 126-801 ops the
   big-M pairwise encoding is not tractable, which is the stated reason the
   CP-SAT backend exists.
6. `cheap_portfolio` is scored on the same 88 runs as its members, not on an
   independent execution. Its wall time is the honest sum of all six, but it
   inherits their determinism, which is exact for all six.

## Layout

```
scripts/sweep10_runner.py            one (arm, workload, solver) solve + independent validation
scripts/sweep10_dispatch.py          two-pool fan-out (CP-SAT wants 8 threads, everything else 1)
scripts/sweep10_variance.py          seed-repeat run
scripts/sweep10_analyze.py           tables + plots; synthesises cheap_portfolio
scripts/sweep10_breakdown.py         per family / pair / arm, and the paired head-to-heads
scripts/sweep10_variance_analyze.py  seed spread
scripts/sweep10_crosscheck.py        against the Sep-3 baseline
raw/out/       880 per-job JSONs (the ten solvers)
raw/outwb/      88 per-job JSONs (cpsat:warmbest)
raw/outvar/    160 per-job JSONs (seeds 1,2)
raw/logs/      dispatcher logs
results/RESULTS.md      full workload x arm x solver table
results/results.csv     the same, flat
results/breakdown.md    per family / pair / arm + paired head-to-heads
results/*.json          summary, headline, variance, all_results
results/failures.txt    the 16 failures with real error text
results/validation.txt  the independent feasibility audit
results/crosscheck.txt  vs the Sep-3 baseline
results/*.png           plots (also copied to plots/)
```

## Reproducing

```bash
export XPURT_CODE_ROOT=/path/to/XPU-RT          # standalone solver tree
export XPURT_DATA_ROOT=/path/to/RoSE/soc/sw/xpu-rt
export XPURT_CPSAT_PYTHON=/path/to/venv/bin/python   # needs ortools
python3 scripts/sweep10_dispatch.py --data-root $XPURT_DATA_ROOT \
  --code-root $XPURT_CODE_ROOT --cpsat-python $XPURT_CPSAT_PYTHON \
  --outdir out --cheap-workers 90 --cpsat-parallel 12 --cpsat-workers 8
python3 scripts/sweep10_analyze.py --outdir out --extra-outdir outwb --dest results
python3 scripts/sweep10_breakdown.py results
```

Concurrency matters: CP-SAT sets `num_search_workers=8` inside its own
subprocess (measured at 750% CPU), so 96 concurrent CP-SAT solves would
oversubscribe 8x and inflate every wall-clock number in the study. The
dispatcher runs the eight single-threaded solvers 90-wide (704 jobs in **43 s**)
and the CP-SAT arms 12-wide (176 jobs in **447 s**).
