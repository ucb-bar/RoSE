# Breakdown: where each solver is strong and weak

Every cell is mean % makespan improvement over `greedy`, counted only on workload-arms where greedy AND the solver both finished with ZERO missed periodic windows. `n` is how many of the group's workload-arms that leaves; a small `n` is itself the finding (the solver was infeasible on the rest).

### By family (both arms pooled)

| group | cheap_portfolio | pso | sa | cpsat | cpsat:warm | cpsat:warmbest | heft_edf | heft | decomposed | greedy_periodic | greedy_reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bimodal | +5.95 (n=8) | +6.84 (n=8) | +6.72 (n=8) | +7.06 (n=8) | +7.06 (n=8) | +7.22 (n=8) | +5.95 (n=8) | +7.15 (n=5) | +0.00 (n=8) | +0.00 (n=8) | -0.03 (n=8) |
| control_mix | +16.02 (n=7) | +17.05 (n=7) | +16.47 (n=7) | +11.40 (n=7) | +18.41 (n=7) | +18.41 (n=7) | +16.02 (n=7) | +14.11 (n=1) | +0.68 (n=7) | -- | -- |
| depth_chain | +0.03 (n=8) | +0.03 (n=8) | +0.03 (n=8) | +0.00 (n=8) | +0.01 (n=8) | +0.01 (n=8) | +0.03 (n=8) | +0.03 (n=8) | +0.02 (n=8) | +0.00 (n=8) | +0.00 (n=8) |
| depth_contended | +9.21 (n=8) | +14.58 (n=8) | +14.78 (n=8) | +2.07 (n=8) | +2.63 (n=8) | +15.06 (n=8) | -41.91 (n=8) | -41.98 (n=8) | -46.57 (n=8) | +9.20 (n=7) | +4.68 (n=6) |
| depth_nav | +0.03 (n=8) | +0.03 (n=8) | +0.03 (n=8) | -0.01 (n=8) | +0.01 (n=8) | +0.01 (n=8) | +0.03 (n=8) | +0.02 (n=6) | +0.02 (n=8) | +0.00 (n=8) | +0.00 (n=8) |
| perception_heavy | +6.73 (n=8) | +8.37 (n=8) | +8.21 (n=8) | +9.25 (n=8) | +9.28 (n=8) | +9.29 (n=8) | +6.73 (n=8) | +7.20 (n=7) | +0.00 (n=8) | -0.88 (n=7) | -4.88 (n=7) |
| saturation | +12.33 (n=7) | +16.73 (n=7) | +14.03 (n=7) | +10.46 (n=7) | +18.39 (n=7) | +19.83 (n=7) | +12.13 (n=7) | +18.28 (n=1) | -1.94 (n=6) | -- | -- |
| scale_ladder | +12.79 (n=8) | +13.24 (n=8) | +13.17 (n=8) | +13.11 (n=8) | +13.27 (n=8) | +13.32 (n=8) | +12.79 (n=8) | +12.79 (n=8) | +0.00 (n=8) | +0.00 (n=8) | +0.00 (n=8) |
| tight_loop | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| vint_intro | +6.40 (n=8) | +7.45 (n=8) | +7.21 (n=8) | +4.21 (n=8) | +7.85 (n=8) | +8.07 (n=8) | +6.40 (n=8) | +4.70 (n=4) | +0.00 (n=8) | +0.01 (n=6) | +0.01 (n=6) |
| vint_multi | +8.45 (n=8) | +8.65 (n=8) | +8.45 (n=8) | +3.93 (n=8) | +8.59 (n=8) | +8.60 (n=8) | +8.45 (n=8) | +6.50 (n=2) | +0.00 (n=8) | +0.00 (n=4) | +0.11 (n=6) |

### By machine pair (tight_loop excluded)

| group | cheap_portfolio | pso | sa | cpsat | cpsat:warm | cpsat:warmbest | heft_edf | heft | decomposed | greedy_periodic | greedy_reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gempair | +5.15 (n=19) | +5.60 (n=19) | +5.39 (n=19) | +6.62 (n=19) | +5.07 (n=19) | +6.68 (n=19) | -1.81 (n=19) | -10.70 (n=9) | -7.84 (n=18) | +0.30 (n=12) | -0.80 (n=14) |
| hetero | +7.22 (n=20) | +10.44 (n=20) | +9.53 (n=20) | +7.42 (n=20) | +9.05 (n=20) | +11.29 (n=20) | -1.17 (n=20) | -7.64 (n=13) | -8.77 (n=20) | +0.34 (n=14) | +0.40 (n=14) |
| quad | +8.18 (n=20) | +9.15 (n=20) | +9.00 (n=20) | -0.41 (n=20) | +7.61 (n=20) | +9.37 (n=20) | +6.63 (n=20) | +5.12 (n=18) | -0.75 (n=20) | +0.78 (n=15) | +0.20 (n=15) |
| rvvpair | +9.96 (n=19) | +11.14 (n=19) | +11.01 (n=19) | +10.73 (n=19) | +11.46 (n=19) | +11.58 (n=19) | +5.79 (n=19) | +2.06 (n=10) | -2.51 (n=19) | +2.55 (n=15) | -0.21 (n=14) |

### By arm (tight_loop excluded)

| group | cheap_portfolio | pso | sa | cpsat | cpsat:warm | cpsat:warmbest | heft_edf | heft | decomposed | greedy_periodic | greedy_reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wl_sweep | +8.83 (n=39) | +10.29 (n=39) | +9.94 (n=39) | +10.18 (n=39) | +11.07 (n=39) | +11.08 (n=39) | +6.46 (n=39) | +3.88 (n=29) | -2.46 (n=39) | +0.33 (n=28) | -0.47 (n=29) |
| wl_sweep_shard | +6.44 (n=39) | +7.91 (n=39) | +7.56 (n=39) | +1.87 (n=39) | +5.52 (n=39) | +8.42 (n=39) | -1.72 (n=39) | -9.29 (n=21) | -7.45 (n=38) | +1.75 (n=28) | +0.28 (n=28) |

### Paired head-to-head (tight_loop excluded, both feasible)

Positive = row is FASTER than column, as a % of the column's makespan.

| | cpsat:warmbest | pso | sa | cpsat:warm | cheap_portfolio | cpsat | heft_edf | greedy |
|---|---|---|---|---|---|---|---|---|
| **cpsat:warmbest** | -- | +0.97 / +0.02 | +1.44 / +0.05 | +1.19 / +0.00 | +2.60 / +0.23 | +3.17 / +0.00 | +5.50 / +0.23 | +9.75 / +9.06 |
| **pso** | -1.05 / -0.02 | -- | +0.50 / +0.00 | +0.12 / -0.00 | +1.66 / +0.00 | +2.10 / +0.01 | +4.59 / +0.00 | +9.10 / +8.63 |
| **sa** | -1.62 / -0.05 | -0.53 / +0.00 | -- | -0.44 / -0.01 | +1.17 / +0.00 | +1.55 / +0.00 | +4.07 / +0.00 | +8.75 / +8.55 |
| **cpsat:warm** | -1.57 / +0.00 | -0.62 / +0.00 | -0.14 / +0.01 | -- | +1.11 / +0.12 | +1.95 / +0.00 | +4.58 / +0.22 | +8.30 / +8.58 |
| **cheap_portfolio** | -2.89 / -0.23 | -1.80 / +0.00 | -1.26 / +0.00 | -1.64 / -0.12 | -- | +0.39 / +0.00 | +3.04 / +0.00 | +7.63 / +6.59 |
| **cpsat** | -4.30 / +0.00 | -3.35 / -0.01 | -2.86 / -0.00 | -2.72 / +0.00 | -1.55 / -0.00 | -- | +1.92 / +0.01 | +6.03 / +7.42 |
| **heft_edf** | -9.10 / -0.23 | -7.95 / +0.00 | -7.46 / +0.00 | -6.97 / -0.22 | -5.93 / +0.00 | -4.99 / -0.01 | -- | +2.37 / +6.29 |
| **greedy** | -11.90 / -9.96 | -10.96 / -9.45 | -10.48 / -9.34 | -10.52 / -9.39 | -9.08 / -7.05 | -7.98 / -8.01 | -5.45 / -6.72 | -- |

(mean / median. A median of 0.00 with a non-zero mean means the two solvers agree on most workloads and differ only in a hard tail.)

### Cheap portfolio composition

Which of the six sub-second heuristics actually supplied the portfolio's answer, over the 80 non-tight_loop workload-arms:

- `heft`: 33
- `heft_edf`: 31
- `greedy`: 6
- `greedy_periodic`: 6
- `decomposed`: 4

Cost of running all six: mean 0.771 s, median 0.496 s, max 3.215 s.
