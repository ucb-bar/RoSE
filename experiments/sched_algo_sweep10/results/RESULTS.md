# Ten-solver scheduler bench: wl_sweep x {base, shard}

88 workload-arms x 12 solvers = 1056 solves.

## Headline (feasible-first, tight_loop excluded)

`usable` = workload-arms where the solver returned a schedule with ZERO missed periodic windows (and greedy also had zero, so the comparison is defined). `mean/median` = %% makespan improvement over greedy on those.

| solver | usable / of | usable %% | mean impr %% | median impr %% | p25 | p75 | worst | best | mean wall s |
|---|---|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 78/78 | 100.0 | 9.747 | 9.059 | 2.437 | 13.374 | -0.03 | 37.262 | 27.509 |
| pso | 78/78 | 100.0 | 9.102 | 8.632 | 2.441 | 13.239 | 0.0 | 36.456 | 12.17 |
| sa | 78/78 | 100.0 | 8.747 | 8.545 | 1.532 | 13.11 | 0.0 | 36.787 | 15.268 |
| cpsat:warm | 78/78 | 100.0 | 8.298 | 8.581 | 0.548 | 12.888 | -28.109 | 36.708 | 29.163 |
| cheap_portfolio | 78/78 | 100.0 | 7.631 | 6.59 | 1.037 | 10.125 | 0.0 | 36.324 | 0.771 |
| cpsat | 78/78 | 100.0 | 6.025 | 7.419 | -0.0 | 11.724 | -38.896 | 32.087 | 34.406 |
| heft_edf | 78/78 | 100.0 | 2.369 | 6.293 | 0.079 | 9.653 | -119.065 | 26.598 | 0.464 |
| greedy | 78/78 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.056 |
| decomposed | 77/78 | 98.7 | -4.924 | 0.0 | 0.0 | 0.002 | -119.067 | 4.792 | 0.073 |
| greedy_reserved | 57/78 | 73.1 | -0.098 | 0.0 | 0.0 | 0.0 | -8.389 | 8.514 | 0.066 |
| greedy_periodic | 56/78 | 71.8 | 1.041 | 0.0 | 0.0 | 0.0 | -5.73 | 36.324 | 0.061 |
| heft | 50/78 | 64.1 | -1.656 | 3.604 | 0.004 | 7.148 | -119.065 | 26.598 | 0.053 |

## Per-solver summary (all 88 workload-arms)

| solver | feasible/88 | wins | wins (ex tight_loop) | total misses | misses ex tight_loop | mean impr vs greedy %% | median wall s | max wall s |
|---|---|---|---|---|---|---|---|---|
| greedy | 78/88 | 7 | 6 | 938 | 42 | 0.0 | 0.04 | 0.195 |
| greedy_periodic | 56/88 | 0 | 0 | 3181 | 2285 | 2.502 | 0.047 | 0.309 |
| greedy_reserved | 57/88 | 0 | 0 | 3294 | 2398 | 0.819 | 0.05 | 0.322 |
| decomposed | 79/88 | 2 | 2 | 2370 | 130 | -4.702 | 0.057 | 0.196 |
| heft | 50/88 | 13 | 13 | 3920 | 2930 | 3.814 | 0.036 | 0.208 |
| heft_edf | 80/88 | 7 | 7 | 896 | 0 | 2.45 | 0.246 | 2.477 |
| pso | 80/88 | 13 | 7 | 896 | 0 | 9.137 | 12.027 | 22.319 |
| sa | 80/88 | 4 | 3 | 896 | 0 | 8.69 | 20.003 | 20.108 |
| cpsat | 80/88 | 25 | 25 | 0 | 0 | 6.372 | 30.998 | 61.394 |
| cpsat:warm | 80/88 | 7 | 7 | 0 | 0 | 8.581 | 19.797 | 62.564 |
| cheap_portfolio | 80/88 | 0 | 0 | 896 | 0 | 7.601 | 0.496 | 3.215 |
| cpsat:warmbest | 80/88 | 10 | 10 | 0 | 0 | 9.983 | 14.274 | 63.276 |

## Per workload-arm: full table


### wl_sweep / networks_bimodal_gempair  (379 ops, 224 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 3607.959 | 3607.959 | 0 | 13.89 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3607.959 | 3607.959 | 0 | 53.97 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3607.959 | 3607.959 | 0 | 11.82 | OPTIMAL | 0.0 | 0/0 |
| sa | 3608.022 | 3608.022 | 0 | 20.00 |  |  | 0/0 |
| heft | 3608.081 | 3608.081 | 0 | 0.05 |  |  | 0/0 |
| heft_edf | 3608.081 | 3608.081 | 0 | 0.13 |  |  | 0/0 |
| pso | 3608.081 | 3608.081 | 0 | 7.75 |  |  | 0/0 |
| cheap_portfolio | 3608.081 | 3608.081 | 0 | 0.35 |  |  | 0/0 |
| greedy | 3627.607 | 3627.607 | 0 | 0.04 |  |  | 0/0 |
| greedy_periodic | 3627.607 | 3627.607 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 3627.607 | 3627.607 | 0 | 0.05 |  |  | 0/0 |
| greedy_reserved | 3628.557 | 3628.557 | 0 | 0.04 |  |  | 0/0 |

### wl_sweep / networks_bimodal_hetero  (379 ops, 224 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 234.474 | 234.474 | 0 | 18.07 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 234.474 | 234.474 | 0 | 8.80 | OPTIMAL | 0.0 | 0/0 |
| cpsat | 234.519 | 234.519 | 0 | 12.74 | OPTIMAL | 0.0 | 0/0 |
| pso | 236.974 | 236.974 | 0 | 14.81 |  |  | 0/0 |
| sa | 238.104 | 238.104 | 0 | 20.00 |  |  | 0/0 |
| heft | 245.304 | 245.304 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 245.304 | 245.304 | 0 | 0.05 |  |  | 0/0 |
| cheap_portfolio | 245.304 | 245.304 | 0 | 0.20 |  |  | 0/0 |
| greedy | 263.720 | 263.720 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 263.720 | 263.720 | 0 | 0.03 |  |  | 0/0 |
| decomposed | 263.720 | 263.720 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 263.721 | 263.721 | 0 | 0.04 |  |  | 0/0 |

### wl_sweep / networks_bimodal_quad  (379 ops, 224 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 213.668 | 213.668 | 0 | 26.61 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 213.668 | 213.668 | 0 | 16.12 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 213.668 | 213.668 | 0 | 19.48 | OPTIMAL | 0.0 | 0/0 |
| sa | 214.668 | 214.668 | 0 | 20.02 |  |  | 0/0 |
| pso | 214.862 | 214.862 | 0 | 20.67 |  |  | 0/0 |
| heft | 218.183 | 218.183 | 0 | 0.03 |  |  | 0/0 |
| heft_edf | 218.183 | 218.183 | 0 | 0.09 |  |  | 0/0 |
| cheap_portfolio | 218.183 | 218.183 | 0 | 0.37 |  |  | 0/0 |
| greedy | 235.370 | 235.370 | 0 | 0.06 |  |  | 0/0 |
| greedy_periodic | 235.370 | 235.370 | 0 | 0.06 |  |  | 0/0 |
| decomposed | 235.370 | 235.370 | 0 | 0.07 |  |  | 0/0 |
| greedy_reserved | 236.372 | 236.372 | 0 | 0.07 |  |  | 0/0 |

### wl_sweep / networks_bimodal_rvvpair  (379 ops, 224 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 430.478 | 430.478 | 0 | 0.03 |  |  | 0/0 |
| heft_edf | 430.478 | 430.478 | 0 | 0.12 |  |  | 0/0 |
| pso | 430.478 | 430.478 | 0 | 8.90 |  |  | 0/0 |
| sa | 430.478 | 430.478 | 0 | 18.80 |  |  | 0/0 |
| cheap_portfolio | 430.478 | 430.478 | 0 | 0.29 |  |  | 0/0 |
| cpsat | 430.535 | 430.535 | 0 | 14.01 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 430.535 | 430.535 | 0 | 20.79 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 430.535 | 430.535 | 0 | 12.59 | OPTIMAL | 0.0 | 0/0 |
| greedy | 498.932 | 498.932 | 0 | 0.04 |  |  | 0/0 |
| greedy_periodic | 498.932 | 498.932 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 498.932 | 498.932 | 0 | 0.05 |  |  | 0/0 |
| greedy_reserved | 499.935 | 499.935 | 0 | 0.02 |  |  | 0/0 |

### wl_sweep / networks_control_mix_gempair  (295 ops, 140 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 51.809 | 79.513 | 0 | 9.18 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 51.809 | 79.390 | 0 | 21.79 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 51.809 | 79.403 | 0 | 11.13 | OPTIMAL | 0.0 | 0/0 |
| heft_edf | 54.074 | 79.370 | 0 | 0.21 |  |  | 0/0 |
| pso | 54.074 | 79.370 | 0 | 5.34 |  |  | 0/0 |
| sa | 54.074 | 79.370 | 0 | 10.20 |  |  | 0/0 |
| cheap_portfolio | 54.074 | 79.370 | 0 | 0.35 |  |  | 0/0 |
| decomposed | 62.398 | 79.370 | 0 | 0.03 |  |  | 0/0 |
| greedy | 62.770 | 79.370 | 0 | 0.02 |  |  | 0/0 |
| heft | 54.074 | 79.370 | 14 | 0.03 |  |  | 0/0 |
| greedy_periodic | 58.321 | 79.370 | 58 | 0.03 |  |  | 0/0 |
| greedy_reserved | 60.682 | 79.748 | 91 | 0.04 |  |  | 0/0 |

### wl_sweep / networks_control_mix_hetero  (295 ops, 140 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 36.795 | 121.179 | 0 | 29.90 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 36.795 | 121.174 | 0 | 39.22 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 36.795 | 121.174 | 0 | 26.21 | OPTIMAL | 0.0 | 0/0 |
| pso | 37.506 | 121.278 | 0 | 9.20 |  |  | 0/0 |
| heft_edf | 38.910 | 121.164 | 0 | 0.10 |  |  | 0/0 |
| sa | 38.910 | 121.164 | 0 | 4.08 |  |  | 0/0 |
| cheap_portfolio | 38.910 | 121.164 | 0 | 0.19 |  |  | 0/0 |
| decomposed | 46.037 | 121.165 | 0 | 0.02 |  |  | 0/0 |
| greedy | 47.986 | 121.278 | 0 | 0.02 |  |  | 0/0 |
| heft | 38.174 | 121.164 | 14 | 0.01 |  |  | 0/0 |
| greedy_reserved | 42.121 | 121.523 | 37 | 0.02 |  |  | 0/0 |
| greedy_periodic | 43.455 | 121.278 | 56 | 0.02 |  |  | 0/0 |

### wl_sweep / networks_control_mix_quad  (295 ops, 140 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 33.430 | 123.621 | 0 | 30.38 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 33.430 | 121.172 | 0 | 20.75 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 33.430 | 121.172 | 0 | 17.85 | OPTIMAL | 0.0 | 0/0 |
| sa | 33.559 | 121.162 | 0 | 12.15 |  |  | 0/0 |
| heft | 33.635 | 121.162 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 33.635 | 121.162 | 0 | 0.17 |  |  | 0/0 |
| pso | 33.635 | 121.162 | 0 | 4.74 |  |  | 0/0 |
| cheap_portfolio | 33.635 | 121.162 | 0 | 0.37 |  |  | 0/0 |
| decomposed | 37.283 | 121.162 | 0 | 0.03 |  |  | 0/0 |
| greedy | 39.160 | 121.162 | 0 | 0.04 |  |  | 0/0 |
| greedy_periodic | 37.039 | 121.162 | 56 | 0.05 |  |  | 0/0 |
| greedy_reserved | 37.513 | 121.278 | 56 | 0.06 |  |  | 0/0 |

### wl_sweep / networks_control_mix_rvvpair  (295 ops, 140 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 82.220 | 126.538 | 0 | 15.11 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 82.221 | 126.154 | 0 | 10.68 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 82.221 | 126.154 | 0 | 8.36 | OPTIMAL | 0.0 | 0/0 |
| pso | 85.625 | 126.146 | 0 | 5.00 |  |  | 0/0 |
| heft_edf | 85.670 | 126.146 | 0 | 0.17 |  |  | 0/0 |
| sa | 85.670 | 126.146 | 0 | 10.51 |  |  | 0/0 |
| cheap_portfolio | 85.670 | 126.146 | 0 | 0.31 |  |  | 0/0 |
| decomposed | 109.604 | 126.146 | 0 | 0.03 |  |  | 0/0 |
| greedy | 111.151 | 126.146 | 0 | 0.02 |  |  | 0/0 |
| heft | 85.670 | 126.146 | 21 | 0.03 |  |  | 0/0 |
| greedy_periodic | 104.595 | 126.146 | 81 | 0.03 |  |  | 0/0 |
| greedy_reserved | 111.509 | 126.527 | 88 | 0.04 |  |  | 0/0 |

### wl_sweep / networks_depth_chain_gempair  (270 ops, 270 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 592.416 | 592.416 | 0 | 0.02 |  |  | 0/0 |
| greedy_periodic | 592.416 | 592.416 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 592.416 | 592.416 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 592.416 | 592.416 | 0 | 0.02 |  |  | 0/0 |
| heft | 592.416 | 592.416 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 592.416 | 592.416 | 0 | 0.12 |  |  | 0/0 |
| pso | 592.416 | 592.416 | 0 | 3.18 |  |  | 0/0 |
| sa | 592.416 | 592.416 | 0 | 7.90 |  |  | 0/0 |
| cheap_portfolio | 592.416 | 592.416 | 0 | 0.20 |  |  | 0/0 |
| cpsat | 592.480 | 592.480 | 0 | 1.41 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 592.480 | 592.480 | 0 | 1.22 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 592.480 | 592.480 | 0 | 0.60 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep / networks_depth_chain_hetero  (270 ops, 270 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 430.586 | 430.586 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 430.586 | 430.586 | 0 | 0.07 |  |  | 0/0 |
| pso | 430.586 | 430.586 | 0 | 1.61 |  |  | 0/0 |
| sa | 430.586 | 430.586 | 0 | 4.84 |  |  | 0/0 |
| cheap_portfolio | 430.586 | 430.586 | 0 | 0.14 |  |  | 0/0 |
| decomposed | 430.587 | 430.587 | 0 | 0.02 |  |  | 0/0 |
| cpsat | 430.648 | 430.648 | 0 | 1.95 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 430.648 | 430.648 | 0 | 1.75 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 430.648 | 430.648 | 0 | 0.92 | OPTIMAL | 0.0 | 0/0 |
| greedy | 430.728 | 430.728 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 430.728 | 430.728 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 430.728 | 430.728 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_depth_chain_quad  (270 ops, 270 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 430.585 | 430.585 | 0 | 0.02 |  |  | 0/0 |
| greedy_periodic | 430.585 | 430.585 | 0 | 0.02 |  |  | 0/0 |
| greedy_reserved | 430.585 | 430.585 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 430.585 | 430.585 | 0 | 0.02 |  |  | 0/0 |
| heft | 430.585 | 430.585 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 430.585 | 430.585 | 0 | 0.14 |  |  | 0/0 |
| pso | 430.585 | 430.585 | 0 | 3.72 |  |  | 0/0 |
| sa | 430.585 | 430.585 | 0 | 9.20 |  |  | 0/0 |
| cheap_portfolio | 430.585 | 430.585 | 0 | 0.24 |  |  | 0/0 |
| cpsat:warm | 430.647 | 430.647 | 0 | 1.78 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 430.647 | 430.647 | 0 | 1.56 | OPTIMAL | 0.0 | 0/0 |
| cpsat | 430.647 | 430.647 | 0 | 3.77 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep / networks_depth_chain_rvvpair  (270 ops, 270 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 222.846 | 222.846 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 222.846 | 222.846 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 222.846 | 222.846 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 222.846 | 222.846 | 0 | 0.02 |  |  | 0/0 |
| heft | 222.846 | 222.846 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 222.846 | 222.846 | 0 | 0.07 |  |  | 0/0 |
| pso | 222.846 | 222.846 | 0 | 3.23 |  |  | 0/0 |
| sa | 222.846 | 222.846 | 0 | 7.79 |  |  | 0/0 |
| cheap_portfolio | 222.846 | 222.846 | 0 | 0.15 |  |  | 0/0 |
| cpsat | 222.912 | 222.912 | 0 | 1.26 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 222.912 | 222.912 | 0 | 0.78 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 222.912 | 222.912 | 0 | 0.60 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep / networks_depth_contended_gempair  (425 ops, 270 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 51.809 | 592.972 | 0 | 16.57 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 51.809 | 592.480 | 0 | 18.84 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 51.810 | 592.480 | 0 | 8.38 | OPTIMAL | 0.0 | 0/0 |
| pso | 51.862 | 592.416 | 0 | 9.34 |  |  | 0/0 |
| sa | 51.903 | 592.416 | 0 | 19.22 |  |  | 0/0 |
| greedy_periodic | 53.985 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| greedy_reserved | 53.985 | 592.905 | 0 | 0.04 |  |  | 0/0 |
| cheap_portfolio | 53.985 | 592.416 | 0 | 0.40 |  |  | 0/0 |
| greedy | 54.515 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 59.733 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| heft | 59.733 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 59.733 | 592.416 | 0 | 0.21 |  |  | 0/0 |

### wl_sweep / networks_depth_contended_hetero  (425 ops, 270 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 36.795 | 432.557 | 0 | 60.78 | FEASIBLE | 0.0518 | 0/0 |
| cpsat:warm | 36.795 | 430.648 | 0 | 49.88 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 36.795 | 430.791 | 0 | 50.30 | OPTIMAL | 0.0 | 0/0 |
| sa | 37.780 | 430.677 | 0 | 20.02 |  |  | 0/0 |
| pso | 38.203 | 430.943 | 0 | 12.12 |  |  | 0/0 |
| greedy_periodic | 40.848 | 430.728 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 40.848 | 438.174 | 0 | 0.03 |  |  | 0/0 |
| cheap_portfolio | 40.848 | 430.728 | 0 | 0.28 |  |  | 0/0 |
| greedy | 44.082 | 430.728 | 0 | 0.03 |  |  | 0/0 |
| heft | 76.831 | 430.586 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 76.831 | 430.586 | 0 | 0.14 |  |  | 0/0 |
| decomposed | 79.306 | 430.587 | 0 | 0.03 |  |  | 0/0 |

### wl_sweep / networks_depth_contended_quad  (425 ops, 270 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 33.430 | 542.523 | 0 | 49.11 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 33.430 | 430.647 | 0 | 31.03 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 33.430 | 430.647 | 0 | 25.20 | OPTIMAL | 0.0 | 0/0 |
| pso | 33.813 | 430.585 | 0 | 20.39 |  |  | 0/0 |
| sa | 34.371 | 430.585 | 0 | 20.00 |  |  | 0/0 |
| heft | 34.404 | 430.585 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 34.404 | 430.585 | 0 | 0.27 |  |  | 0/0 |
| cheap_portfolio | 34.404 | 430.585 | 0 | 0.50 |  |  | 0/0 |
| greedy_periodic | 36.594 | 430.585 | 0 | 0.05 |  |  | 0/0 |
| greedy_reserved | 36.594 | 430.728 | 0 | 0.05 |  |  | 0/0 |
| decomposed | 36.938 | 430.585 | 0 | 0.04 |  |  | 0/0 |
| greedy | 37.789 | 430.585 | 0 | 0.05 |  |  | 0/0 |

### wl_sweep / networks_depth_contended_rvvpair  (425 ops, 270 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 82.220 | 222.912 | 0 | 12.68 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 82.221 | 222.912 | 0 | 11.25 | OPTIMAL | 0.0 | 0/0 |
| cpsat | 82.221 | 223.327 | 0 | 60.61 | FEASIBLE | 0.1346 | 0/0 |
| sa | 82.335 | 222.846 | 0 | 20.01 |  |  | 0/0 |
| pso | 82.447 | 222.846 | 0 | 17.52 |  |  | 0/0 |
| heft | 99.242 | 222.846 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 99.242 | 222.846 | 0 | 0.27 |  |  | 0/0 |
| cheap_portfolio | 99.242 | 222.846 | 0 | 0.47 |  |  | 0/0 |
| greedy | 102.792 | 222.846 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 111.698 | 222.846 | 0 | 0.04 |  |  | 0/0 |
| greedy_periodic | 94.853 | 233.405 | 35 | 0.04 |  |  | 0/0 |
| greedy_reserved | 94.853 | 263.334 | 35 | 0.04 |  |  | 0/0 |

### wl_sweep / networks_depth_nav_gempair  (382 ops, 382 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 592.416 | 592.416 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 592.416 | 592.416 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 592.416 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 592.416 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| heft | 592.416 | 592.416 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 592.416 | 592.416 | 0 | 0.28 |  |  | 0/0 |
| pso | 592.416 | 592.416 | 0 | 6.97 |  |  | 0/0 |
| sa | 592.416 | 592.416 | 0 | 17.18 |  |  | 0/0 |
| cheap_portfolio | 592.416 | 592.416 | 0 | 0.46 |  |  | 0/0 |
| cpsat | 592.480 | 592.480 | 0 | 1.71 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 592.480 | 592.480 | 0 | 1.67 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 592.480 | 592.480 | 0 | 1.72 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep / networks_depth_nav_hetero  (382 ops, 382 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 430.586 | 430.586 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 430.586 | 430.586 | 0 | 0.14 |  |  | 0/0 |
| pso | 430.586 | 430.586 | 0 | 3.39 |  |  | 0/0 |
| sa | 430.586 | 430.586 | 0 | 8.19 |  |  | 0/0 |
| cheap_portfolio | 430.586 | 430.586 | 0 | 0.28 |  |  | 0/0 |
| decomposed | 430.587 | 430.587 | 0 | 0.03 |  |  | 0/0 |
| cpsat | 430.648 | 430.648 | 0 | 1.97 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 430.648 | 430.648 | 0 | 1.27 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 430.648 | 430.648 | 0 | 1.90 | OPTIMAL | 0.0 | 0/0 |
| greedy | 430.728 | 430.728 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 430.728 | 430.728 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 430.728 | 430.728 | 0 | 0.03 |  |  | 0/0 |

### wl_sweep / networks_depth_nav_quad  (382 ops, 382 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 430.585 | 430.585 | 0 | 0.05 |  |  | 0/0 |
| greedy_periodic | 430.585 | 430.585 | 0 | 0.05 |  |  | 0/0 |
| greedy_reserved | 430.585 | 430.585 | 0 | 0.06 |  |  | 0/0 |
| decomposed | 430.585 | 430.585 | 0 | 0.05 |  |  | 0/0 |
| heft | 430.585 | 430.585 | 0 | 0.03 |  |  | 0/0 |
| heft_edf | 430.585 | 430.585 | 0 | 0.19 |  |  | 0/0 |
| pso | 430.585 | 430.585 | 0 | 5.86 |  |  | 0/0 |
| sa | 430.585 | 430.585 | 0 | 16.74 |  |  | 0/0 |
| cheap_portfolio | 430.585 | 430.585 | 0 | 0.43 |  |  | 0/0 |
| cpsat | 430.647 | 430.647 | 0 | 3.82 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 430.647 | 430.647 | 0 | 2.84 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 430.647 | 430.647 | 0 | 2.73 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep / networks_depth_nav_rvvpair  (382 ops, 382 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 222.846 | 222.846 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 222.846 | 222.846 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 222.846 | 222.846 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 222.846 | 222.846 | 0 | 0.04 |  |  | 0/0 |
| heft | 222.846 | 222.846 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 222.846 | 222.846 | 0 | 0.27 |  |  | 0/0 |
| pso | 222.846 | 222.846 | 0 | 6.68 |  |  | 0/0 |
| sa | 222.846 | 222.846 | 0 | 14.97 |  |  | 0/0 |
| cheap_portfolio | 222.846 | 222.846 | 0 | 0.44 |  |  | 0/0 |
| cpsat | 222.912 | 222.912 | 0 | 1.19 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 222.912 | 222.912 | 0 | 1.04 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 222.912 | 222.912 | 0 | 1.46 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep / networks_perception_heavy_gempair  (183 ops, 28 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 155.085 | 155.085 | 0 | 8.33 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 155.085 | 155.085 | 0 | 52.10 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 155.085 | 155.085 | 0 | 5.04 | OPTIMAL | 0.0 | 0/0 |
| pso | 155.237 | 155.237 | 0 | 2.21 |  |  | 0/0 |
| heft | 155.256 | 155.256 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 155.256 | 155.256 | 0 | 0.03 |  |  | 0/0 |
| sa | 155.256 | 155.256 | 0 | 4.02 |  |  | 0/0 |
| cheap_portfolio | 155.256 | 155.256 | 0 | 0.09 |  |  | 0/0 |
| greedy | 162.590 | 162.590 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 162.590 | 162.590 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 162.590 | 162.590 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 176.230 | 176.230 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_perception_heavy_hetero  (183 ops, 28 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 112.334 | 112.334 | 0 | 14.16 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 112.334 | 112.334 | 0 | 12.89 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 112.382 | 112.382 | 0 | 11.15 | OPTIMAL | 0.0 | 0/0 |
| pso | 113.266 | 113.266 | 0 | 2.65 |  |  | 0/0 |
| sa | 113.753 | 113.753 | 0 | 2.92 |  |  | 0/0 |
| heft | 118.296 | 118.296 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 118.296 | 118.296 | 0 | 0.02 |  |  | 0/0 |
| cheap_portfolio | 118.296 | 118.296 | 0 | 0.06 |  |  | 0/0 |
| greedy | 130.993 | 130.993 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 130.993 | 130.993 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 133.979 | 133.979 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 133.979 | 133.979 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_perception_heavy_quad  (183 ops, 28 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 101.509 | 101.509 | 0 | 24.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 101.509 | 101.509 | 0 | 21.43 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 101.509 | 101.509 | 0 | 16.40 | OPTIMAL | 0.0 | 0/0 |
| pso | 102.129 | 102.129 | 0 | 2.47 |  |  | 0/0 |
| sa | 102.202 | 102.202 | 0 | 5.84 |  |  | 0/0 |
| heft | 104.555 | 104.555 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 104.555 | 104.555 | 0 | 0.03 |  |  | 0/0 |
| cheap_portfolio | 104.555 | 104.555 | 0 | 0.11 |  |  | 0/0 |
| greedy | 110.698 | 110.698 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 110.698 | 110.698 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 110.698 | 110.698 | 0 | 0.02 |  |  | 0/0 |
| greedy_reserved | 119.912 | 119.912 | 0 | 0.02 |  |  | 0/0 |

### wl_sweep / networks_perception_heavy_rvvpair  (183 ops, 28 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| sa | 232.845 | 232.845 | 0 | 5.31 |  |  | 0/0 |
| heft | 232.863 | 232.863 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 232.863 | 232.863 | 0 | 0.03 |  |  | 0/0 |
| pso | 232.863 | 232.863 | 0 | 1.67 |  |  | 0/0 |
| cheap_portfolio | 232.863 | 232.863 | 0 | 0.09 |  |  | 0/0 |
| cpsat | 232.895 | 232.895 | 0 | 60.83 | FEASIBLE | 0.1095 | 0/0 |
| cpsat:warmbest | 232.896 | 232.896 | 0 | 60.72 | FEASIBLE | 0.1093 | 0/0 |
| cpsat:warm | 232.896 | 232.896 | 0 | 3.42 | OPTIMAL | 0.0 | 0/0 |
| greedy | 267.514 | 267.514 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 267.514 | 267.514 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 267.514 | 267.514 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 281.966 | 281.966 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_saturation_gempair  (393 ops, 238 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 130.305 | 130.305 | 0 | 60.61 | FEASIBLE | 0.189 | 0/0 |
| cpsat:warm | 130.398 | 130.398 | 0 | 60.97 | FEASIBLE | 0.1855 | 0/0 |
| cpsat:warmbest | 130.943 | 130.943 | 0 | 61.12 | FEASIBLE | 0.1817 | 0/0 |
| pso | 144.991 | 144.991 | 0 | 20.22 |  |  | 0/0 |
| heft_edf | 154.639 | 154.639 | 0 | 0.46 |  |  | 0/0 |
| sa | 154.639 | 154.639 | 0 | 17.00 |  |  | 0/0 |
| cheap_portfolio | 154.639 | 154.639 | 0 | 0.70 |  |  | 0/0 |
| decomposed | 169.927 | 169.927 | 0 | 0.05 |  |  | 0/0 |
| greedy | 170.994 | 170.994 | 8 | 0.04 |  |  | 0/0 |
| greedy_reserved | 214.500 | 214.500 | 125 | 0.05 |  |  | 0/0 |
| heft | 112.157 | 112.157 | 154 | 0.05 |  |  | 0/0 |
| greedy_periodic | 160.728 | 160.728 | 158 | 0.05 |  |  | 0/0 |

### wl_sweep / networks_saturation_hetero  (393 ops, 238 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 81.825 | 105.849 | 0 | 61.03 | FEASIBLE | 0.0855 | 0/0 |
| cpsat | 82.043 | 104.446 | 0 | 60.79 | FEASIBLE | 0.0734 | 0/0 |
| cpsat:warmbest | 82.083 | 105.851 | 0 | 61.30 | FEASIBLE | 0.0887 | 0/0 |
| pso | 96.259 | 100.013 | 0 | 7.12 |  |  | 0/0 |
| heft_edf | 104.251 | 104.251 | 0 | 0.23 |  |  | 0/0 |
| sa | 104.251 | 104.251 | 0 | 9.26 |  |  | 0/0 |
| cheap_portfolio | 104.251 | 104.251 | 0 | 0.40 |  |  | 0/0 |
| greedy | 120.807 | 120.807 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 136.658 | 136.658 | 0 | 0.03 |  |  | 0/0 |
| heft | 86.938 | 99.746 | 44 | 0.02 |  |  | 0/0 |
| greedy_reserved | 106.839 | 110.445 | 146 | 0.05 |  |  | 0/0 |
| greedy_periodic | 110.141 | 111.908 | 164 | 0.04 |  |  | 0/0 |

### wl_sweep / networks_saturation_quad  (393 ops, 238 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 73.568 | 99.751 | 0 | 38.00 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 73.568 | 99.844 | 0 | 33.74 | OPTIMAL | 0.0 | 0/0 |
| cpsat | 73.605 | 104.517 | 0 | 61.00 | FEASIBLE | 0.0181 | 0/0 |
| pso | 73.889 | 99.745 | 0 | 20.55 |  |  | 0/0 |
| sa | 74.206 | 99.745 | 0 | 20.04 |  |  | 0/0 |
| heft | 75.544 | 99.745 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 75.544 | 99.745 | 0 | 0.29 |  |  | 0/0 |
| cheap_portfolio | 75.544 | 99.745 | 0 | 0.67 |  |  | 0/0 |
| decomposed | 90.562 | 99.745 | 0 | 0.06 |  |  | 0/0 |
| greedy | 92.440 | 99.745 | 0 | 0.06 |  |  | 0/0 |
| greedy_periodic | 92.758 | 99.745 | 189 | 0.10 |  |  | 0/0 |
| greedy_reserved | 99.081 | 99.888 | 189 | 0.12 |  |  | 0/0 |

### wl_sweep / networks_saturation_rvvpair  (393 ops, 238 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 170.980 | 170.980 | 0 | 34.37 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 170.980 | 170.980 | 0 | 61.40 | FEASIBLE | 0.1261 | 0/0 |
| cpsat:warmbest | 170.980 | 170.980 | 0 | 22.95 | OPTIMAL | 0.0 | 0/0 |
| pso | 174.694 | 174.694 | 0 | 18.97 |  |  | 0/0 |
| sa | 179.911 | 179.911 | 0 | 20.01 |  |  | 0/0 |
| heft_edf | 182.981 | 182.981 | 0 | 0.56 |  |  | 0/0 |
| cheap_portfolio | 182.981 | 182.981 | 0 | 0.81 |  |  | 0/0 |
| greedy | 248.067 | 248.067 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 248.067 | 248.067 | 0 | 0.04 |  |  | 0/0 |
| heft | 170.960 | 170.960 | 47 | 0.05 |  |  | 0/0 |
| greedy_periodic | 221.599 | 221.599 | 132 | 0.05 |  |  | 0/0 |
| greedy_reserved | 264.833 | 264.833 | 231 | 0.07 |  |  | 0/0 |

### wl_sweep / networks_scale_ladder_gempair  (126 ops, 0 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 98.589 | 98.589 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 98.589 | 98.589 | 0 | 0.01 |  |  | 0/0 |
| pso | 98.589 | 98.589 | 0 | 1.42 |  |  | 0/0 |
| sa | 98.589 | 98.589 | 0 | 3.50 |  |  | 0/0 |
| cheap_portfolio | 98.589 | 98.589 | 0 | 0.05 |  |  | 0/0 |
| cpsat | 98.598 | 98.598 | 0 | 1.08 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 98.598 | 98.598 | 0 | 0.59 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 98.598 | 98.598 | 0 | 0.59 | OPTIMAL | 0.0 | 0/0 |
| greedy | 128.846 | 128.846 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 128.846 | 128.846 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 128.846 | 128.846 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 128.846 | 128.846 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_scale_ladder_hetero  (126 ops, 0 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 97.164 | 97.164 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 97.164 | 97.164 | 0 | 0.01 |  |  | 0/0 |
| pso | 97.164 | 97.164 | 0 | 0.73 |  |  | 0/0 |
| sa | 97.164 | 97.164 | 0 | 1.56 |  |  | 0/0 |
| cheap_portfolio | 97.164 | 97.164 | 0 | 0.03 |  |  | 0/0 |
| cpsat | 97.173 | 97.173 | 0 | 0.74 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 97.173 | 97.173 | 0 | 0.76 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 97.173 | 97.173 | 0 | 0.83 | OPTIMAL | 0.0 | 0/0 |
| greedy | 111.349 | 111.349 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 111.349 | 111.349 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 111.349 | 111.349 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 111.349 | 111.349 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_scale_ladder_quad  (126 ops, 0 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 97.163 | 97.163 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 97.163 | 97.163 | 0 | 0.01 |  |  | 0/0 |
| pso | 97.163 | 97.163 | 0 | 1.37 |  |  | 0/0 |
| sa | 97.163 | 97.163 | 0 | 3.65 |  |  | 0/0 |
| cheap_portfolio | 97.163 | 97.163 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 97.173 | 97.173 | 0 | 1.50 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 97.173 | 97.173 | 0 | 0.85 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 97.173 | 97.173 | 0 | 1.41 | OPTIMAL | 0.0 | 0/0 |
| greedy | 109.635 | 109.635 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 109.635 | 109.635 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 109.635 | 109.635 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 109.635 | 109.635 | 0 | 0.02 |  |  | 0/0 |

### wl_sweep / networks_scale_ladder_rvvpair  (126 ops, 0 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 97.370 | 97.370 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 97.370 | 97.370 | 0 | 0.01 |  |  | 0/0 |
| pso | 97.370 | 97.370 | 0 | 1.43 |  |  | 0/0 |
| sa | 97.370 | 97.370 | 0 | 3.43 |  |  | 0/0 |
| cheap_portfolio | 97.370 | 97.370 | 0 | 0.04 |  |  | 0/0 |
| cpsat | 97.378 | 97.378 | 0 | 0.77 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 97.378 | 97.378 | 0 | 0.58 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 97.378 | 97.378 | 0 | 0.63 | OPTIMAL | 0.0 | 0/0 |
| greedy | 132.654 | 132.654 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 132.654 | 132.654 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 132.654 | 132.654 | 0 | 0.00 |  |  | 0/0 |
| decomposed | 132.654 | 132.654 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep / networks_tight_loop_gempair  (252 ops, 252 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 78.551 | 78.551 | 84 | 7.30 |  |  | 0/0 |
| sa | 80.499 | 80.499 | 84 | 20.02 |  |  | 0/0 |
| heft_edf | 92.204 | 92.204 | 84 | 0.22 |  |  | 0/0 |
| cheap_portfolio | 92.204 | 92.204 | 84 | 0.37 |  |  | 0/0 |
| greedy | 98.403 | 98.403 | 84 | 0.02 |  |  | 0/0 |
| greedy_periodic | 98.403 | 98.403 | 84 | 0.02 |  |  | 0/0 |
| greedy_reserved | 98.403 | 98.403 | 84 | 0.03 |  |  | 0/0 |
| heft | 75.580 | 75.580 | 119 | 0.02 |  |  | 0/0 |
| decomposed | 93.883 | 93.883 | 252 | 0.06 |  |  | 0/0 |
| cpsat | FAILED | | | 0.523 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.662 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 0.773 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep / networks_tight_loop_hetero  (252 ops, 252 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 76.085 | 76.085 | 84 | 5.82 |  |  | 0/0 |
| sa | 77.629 | 77.629 | 84 | 15.98 |  |  | 0/0 |
| heft_edf | 100.815 | 100.815 | 84 | 0.11 |  |  | 0/0 |
| cheap_portfolio | 100.815 | 100.815 | 84 | 0.21 |  |  | 0/0 |
| greedy | 102.037 | 102.037 | 84 | 0.02 |  |  | 0/0 |
| greedy_periodic | 102.037 | 102.037 | 84 | 0.02 |  |  | 0/0 |
| greedy_reserved | 102.037 | 102.037 | 84 | 0.02 |  |  | 0/0 |
| heft | 76.051 | 76.051 | 108 | 0.01 |  |  | 0/0 |
| decomposed | 106.925 | 106.925 | 252 | 0.04 |  |  | 0/0 |
| cpsat | FAILED | | | 0.544 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.617 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 0.829 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep / networks_tight_loop_quad  (252 ops, 252 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| sa | 45.671 | 45.671 | 84 | 10.84 |  |  | 0/0 |
| pso | 45.686 | 45.686 | 84 | 5.38 |  |  | 0/0 |
| heft | 45.916 | 45.916 | 84 | 0.02 |  |  | 0/0 |
| heft_edf | 45.916 | 45.916 | 84 | 0.22 |  |  | 0/0 |
| cheap_portfolio | 45.916 | 45.916 | 84 | 0.45 |  |  | 0/0 |
| greedy | 84.338 | 84.338 | 84 | 0.04 |  |  | 0/0 |
| greedy_periodic | 84.338 | 84.338 | 84 | 0.04 |  |  | 0/0 |
| greedy_reserved | 84.338 | 84.338 | 84 | 0.04 |  |  | 0/0 |
| decomposed | 64.268 | 64.268 | 252 | 0.09 |  |  | 0/0 |
| cpsat | FAILED | | | 0.609 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.715 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 1.262 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep / networks_tight_loop_rvvpair  (252 ops, 252 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 75.372 | 75.372 | 84 | 10.79 |  |  | 0/0 |
| sa | 75.394 | 75.394 | 84 | 10.65 |  |  | 0/0 |
| heft | 75.598 | 75.598 | 84 | 0.02 |  |  | 0/0 |
| heft_edf | 75.598 | 75.598 | 84 | 0.23 |  |  | 0/0 |
| cheap_portfolio | 75.598 | 75.598 | 84 | 0.37 |  |  | 0/0 |
| greedy | 89.921 | 89.921 | 84 | 0.02 |  |  | 0/0 |
| greedy_periodic | 89.921 | 89.921 | 84 | 0.02 |  |  | 0/0 |
| greedy_reserved | 89.921 | 89.921 | 84 | 0.03 |  |  | 0/0 |
| decomposed | 95.507 | 95.507 | 252 | 0.05 |  |  | 0/0 |
| cpsat | FAILED | | | 0.539 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.625 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 0.936 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep / networks_vint_intro_gempair  (661 ops, 56 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 3754.051 | 3754.051 | 0 | 0.51 |  |  | 0/0 |
| pso | 3754.051 | 3754.051 | 0 | 20.87 |  |  | 0/0 |
| sa | 3754.051 | 3754.051 | 0 | 20.05 |  |  | 0/0 |
| cheap_portfolio | 3754.051 | 3754.051 | 0 | 1.01 |  |  | 0/0 |
| cpsat | 3754.235 | 3754.235 | 0 | 4.42 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3754.235 | 3754.235 | 0 | 1.49 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3754.235 | 3754.235 | 0 | 2.49 | OPTIMAL | 0.0 | 0/0 |
| decomposed | 4074.368 | 4074.368 | 0 | 0.12 |  |  | 0/0 |
| greedy | 4074.944 | 4074.944 | 0 | 0.08 |  |  | 0/0 |
| greedy_reserved | 4072.363 | 4072.363 | 5 | 0.08 |  |  | 0/0 |
| greedy_periodic | 4074.670 | 4074.670 | 5 | 0.07 |  |  | 0/0 |
| heft | 3754.051 | 3754.051 | 56 | 0.14 |  |  | 0/0 |

### wl_sweep / networks_vint_intro_hetero  (661 ops, 56 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 3940.536 | 3940.536 | 0 | 61.80 | FEASIBLE | 0.0454 | 0/0 |
| cpsat | 3947.610 | 3947.610 | 0 | 61.10 | FEASIBLE | 0.0472 | 0/0 |
| cpsat:warm | 3949.901 | 3949.901 | 0 | 61.19 | FEASIBLE | 0.0477 | 0/0 |
| pso | 3980.438 | 3980.438 | 0 | 21.19 |  |  | 0/0 |
| sa | 4017.042 | 4017.042 | 0 | 20.05 |  |  | 0/0 |
| heft | 4160.689 | 4160.689 | 0 | 0.05 |  |  | 0/0 |
| heft_edf | 4160.689 | 4160.689 | 0 | 0.14 |  |  | 0/0 |
| cheap_portfolio | 4160.689 | 4160.689 | 0 | 0.49 |  |  | 0/0 |
| greedy | 4472.274 | 4472.274 | 0 | 0.07 |  |  | 0/0 |
| greedy_reserved | 4472.274 | 4472.274 | 0 | 0.07 |  |  | 0/0 |
| decomposed | 4472.274 | 4472.274 | 0 | 0.09 |  |  | 0/0 |
| greedy_periodic | 4472.374 | 4472.374 | 0 | 0.07 |  |  | 0/0 |

### wl_sweep / networks_vint_intro_quad  (661 ops, 56 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 3750.825 | 3750.825 | 0 | 0.11 |  |  | 0/0 |
| heft_edf | 3750.825 | 3750.825 | 0 | 0.33 |  |  | 0/0 |
| pso | 3750.825 | 3750.825 | 0 | 15.63 |  |  | 0/0 |
| sa | 3750.825 | 3750.825 | 0 | 20.02 |  |  | 0/0 |
| cheap_portfolio | 3750.825 | 3750.825 | 0 | 0.87 |  |  | 0/0 |
| cpsat:warm | 3751.009 | 3751.009 | 0 | 14.64 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3751.009 | 3751.009 | 0 | 13.78 | OPTIMAL | 0.0 | 0/0 |
| greedy | 3844.686 | 3844.686 | 0 | 0.10 |  |  | 0/0 |
| greedy_periodic | 3844.686 | 3844.686 | 0 | 0.10 |  |  | 0/0 |
| greedy_reserved | 3844.686 | 3844.686 | 0 | 0.10 |  |  | 0/0 |
| decomposed | 3844.686 | 3844.686 | 0 | 0.13 |  |  | 0/0 |
| cpsat | 4394.914 | 4394.914 | 0 | 61.27 | FEASIBLE | 0.1465 | 0/0 |

### wl_sweep / networks_vint_intro_rvvpair  (661 ops, 56 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 11068.078 | 11068.078 | 0 | 0.35 |  |  | 0/0 |
| pso | 11068.078 | 11068.078 | 0 | 21.29 |  |  | 0/0 |
| sa | 11068.078 | 11068.078 | 0 | 20.08 |  |  | 0/0 |
| cheap_portfolio | 11068.078 | 11068.078 | 0 | 0.82 |  |  | 0/0 |
| cpsat | 11068.233 | 11068.233 | 0 | 5.60 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 11068.233 | 11068.233 | 0 | 2.20 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 11068.233 | 11068.233 | 0 | 2.72 | OPTIMAL | 0.0 | 0/0 |
| greedy_periodic | 12141.930 | 12141.930 | 0 | 0.06 |  |  | 0/0 |
| greedy | 12149.205 | 12149.205 | 0 | 0.08 |  |  | 0/0 |
| decomposed | 12149.205 | 12149.205 | 0 | 0.13 |  |  | 0/0 |
| greedy_reserved | 12187.950 | 12187.950 | 0 | 0.08 |  |  | 0/0 |
| heft | 11068.078 | 11068.078 | 56 | 0.13 |  |  | 0/0 |

### wl_sweep / networks_vint_multi_gempair  (745 ops, 140 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 3754.235 | 3754.235 | 0 | 4.88 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3754.235 | 3754.235 | 0 | 8.23 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3754.235 | 3754.235 | 0 | 7.43 | OPTIMAL | 0.0 | 0/0 |
| pso | 3755.207 | 3755.207 | 0 | 21.62 |  |  | 0/0 |
| heft_edf | 3757.463 | 3757.463 | 0 | 1.27 |  |  | 0/0 |
| sa | 3757.463 | 3757.463 | 0 | 20.05 |  |  | 0/0 |
| cheap_portfolio | 3757.463 | 3757.463 | 0 | 1.88 |  |  | 0/0 |
| greedy | 4107.657 | 4107.657 | 0 | 0.09 |  |  | 0/0 |
| decomposed | 4107.657 | 4107.657 | 0 | 0.14 |  |  | 0/0 |
| greedy_reserved | 4108.035 | 4108.035 | 0 | 0.10 |  |  | 0/0 |
| greedy_periodic | 4107.657 | 4107.657 | 19 | 0.10 |  |  | 0/0 |
| heft | 3754.051 | 3754.051 | 140 | 0.18 |  |  | 0/0 |

### wl_sweep / networks_vint_multi_hetero  (745 ops, 140 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 4150.600 | 4150.600 | 0 | 61.50 | FEASIBLE | 0.0937 | 0/0 |
| cpsat:warm | 4156.214 | 4156.214 | 0 | 61.33 | FEASIBLE | 0.095 | 0/0 |
| heft_edf | 4160.689 | 4160.689 | 0 | 0.75 |  |  | 0/0 |
| pso | 4160.689 | 4160.689 | 0 | 20.22 |  |  | 0/0 |
| sa | 4160.689 | 4160.689 | 0 | 20.05 |  |  | 0/0 |
| cheap_portfolio | 4160.689 | 4160.689 | 0 | 1.18 |  |  | 0/0 |
| cpsat | 4320.503 | 4320.503 | 0 | 61.13 | FEASIBLE | 0.1294 | 0/0 |
| decomposed | 4546.275 | 4546.275 | 0 | 0.10 |  |  | 0/0 |
| greedy | 4546.388 | 4546.388 | 0 | 0.09 |  |  | 0/0 |
| heft | 4160.689 | 4160.689 | 33 | 0.06 |  |  | 0/0 |
| greedy_periodic | 4546.388 | 4546.388 | 63 | 0.09 |  |  | 0/0 |
| greedy_reserved | 4546.633 | 4546.633 | 63 | 0.09 |  |  | 0/0 |

### wl_sweep / networks_vint_multi_quad  (745 ops, 140 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 3750.825 | 3750.825 | 0 | 0.14 |  |  | 0/0 |
| heft_edf | 3750.825 | 3750.825 | 0 | 1.12 |  |  | 0/0 |
| pso | 3750.825 | 3750.825 | 0 | 20.64 |  |  | 0/0 |
| sa | 3750.825 | 3750.825 | 0 | 20.11 |  |  | 0/0 |
| cheap_portfolio | 3750.825 | 3750.825 | 0 | 1.78 |  |  | 0/0 |
| cpsat:warm | 3751.009 | 3751.009 | 0 | 18.36 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3751.009 | 3751.009 | 0 | 25.89 | OPTIMAL | 0.0 | 0/0 |
| greedy_reserved | 4011.504 | 4011.504 | 0 | 0.14 |  |  | 0/0 |
| decomposed | 4012.786 | 4012.786 | 0 | 0.13 |  |  | 0/0 |
| greedy | 4012.814 | 4012.814 | 0 | 0.13 |  |  | 0/0 |
| greedy_periodic | 4012.814 | 4012.814 | 0 | 0.12 |  |  | 0/0 |
| cpsat | 4327.992 | 4327.992 | 0 | 60.94 | FEASIBLE | 0.1333 | 0/0 |

### wl_sweep / networks_vint_multi_rvvpair  (745 ops, 140 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 11068.233 | 11068.233 | 0 | 29.19 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 11068.233 | 11068.233 | 0 | 4.44 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 11068.233 | 11068.233 | 0 | 4.97 | OPTIMAL | 0.0 | 0/0 |
| heft_edf | 11069.591 | 11069.591 | 0 | 1.31 |  |  | 0/0 |
| pso | 11069.591 | 11069.591 | 0 | 22.32 |  |  | 0/0 |
| sa | 11069.591 | 11069.591 | 0 | 20.05 |  |  | 0/0 |
| cheap_portfolio | 11069.591 | 11069.591 | 0 | 1.92 |  |  | 0/0 |
| greedy_reserved | 12290.346 | 12290.346 | 0 | 0.11 |  |  | 0/0 |
| greedy | 12337.001 | 12337.001 | 0 | 0.10 |  |  | 0/0 |
| greedy_periodic | 12337.001 | 12337.001 | 0 | 0.10 |  |  | 0/0 |
| decomposed | 12337.001 | 12337.001 | 0 | 0.15 |  |  | 0/0 |
| heft | 11068.078 | 11068.078 | 140 | 0.16 |  |  | 0/0 |

### wl_sweep_shard / networks_bimodal_gempair  (442 ops, 224 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 253.500 | 253.500 | 0 | 60.90 | FEASIBLE | 0.1566 | 0/0 |
| cpsat:warmbest | 253.501 | 253.501 | 0 | 60.93 | FEASIBLE | 0.153 | 0/0 |
| cpsat:warm | 253.502 | 253.502 | 0 | 60.75 | FEASIBLE | 0.1568 | 0/0 |
| heft_edf | 253.885 | 253.885 | 0 | 0.23 |  |  | 0/0 |
| pso | 253.885 | 253.885 | 0 | 20.32 |  |  | 0/0 |
| sa | 253.885 | 253.885 | 0 | 20.03 |  |  | 0/0 |
| cheap_portfolio | 253.885 | 253.885 | 0 | 0.55 |  |  | 0/0 |
| greedy | 263.200 | 263.200 | 0 | 0.05 |  |  | 0/0 |
| greedy_periodic | 263.200 | 263.200 | 0 | 0.06 |  |  | 0/0 |
| greedy_reserved | 263.200 | 263.200 | 0 | 0.06 |  |  | 0/0 |
| decomposed | 263.200 | 263.200 | 0 | 0.08 |  |  | 0/0 |
| heft | 252.965 | 252.965 | 224 | 0.07 |  |  | 0/0 |

### wl_sweep_shard / networks_bimodal_hetero  (442 ops, 224 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 224.856 | 224.856 | 0 | 60.96 | FEASIBLE | 0.3197 | 0/0 |
| cpsat:warmbest | 227.182 | 227.182 | 0 | 60.79 | FEASIBLE | 0.3267 | 0/0 |
| pso | 228.106 | 228.106 | 0 | 18.88 |  |  | 0/0 |
| cpsat:warm | 228.318 | 228.318 | 0 | 60.95 | FEASIBLE | 0.3301 | 0/0 |
| sa | 228.918 | 228.918 | 0 | 20.01 |  |  | 0/0 |
| heft_edf | 232.702 | 232.702 | 0 | 0.10 |  |  | 0/0 |
| cheap_portfolio | 232.702 | 232.702 | 0 | 0.32 |  |  | 0/0 |
| greedy_reserved | 251.613 | 251.613 | 0 | 0.05 |  |  | 0/0 |
| greedy | 252.616 | 252.616 | 0 | 0.05 |  |  | 0/0 |
| greedy_periodic | 252.616 | 252.616 | 0 | 0.05 |  |  | 0/0 |
| decomposed | 252.616 | 252.616 | 0 | 0.05 |  |  | 0/0 |
| heft | 231.689 | 231.689 | 224 | 0.03 |  |  | 0/0 |

### wl_sweep_shard / networks_bimodal_quad  (442 ops, 224 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 146.992 | 146.992 | 0 | 61.10 | FEASIBLE | 0.1051 | 0/0 |
| cpsat:warm | 148.284 | 148.284 | 0 | 61.40 | FEASIBLE | 0.113 | 0/0 |
| pso | 148.378 | 148.378 | 0 | 20.45 |  |  | 0/0 |
| sa | 148.899 | 148.899 | 0 | 20.02 |  |  | 0/0 |
| heft | 149.617 | 149.617 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 149.617 | 149.617 | 0 | 0.14 |  |  | 0/0 |
| cheap_portfolio | 149.617 | 149.617 | 0 | 0.53 |  |  | 0/0 |
| cpsat | 150.486 | 150.486 | 0 | 61.09 | FEASIBLE | 0.126 | 0/0 |
| greedy | 161.231 | 161.231 | 0 | 0.08 |  |  | 0/0 |
| greedy_periodic | 161.231 | 161.231 | 0 | 0.08 |  |  | 0/0 |
| greedy_reserved | 161.231 | 161.231 | 0 | 0.09 |  |  | 0/0 |
| decomposed | 161.231 | 161.231 | 0 | 0.09 |  |  | 0/0 |

### wl_sweep_shard / networks_bimodal_rvvpair  (442 ops, 224 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 279.155 | 279.155 | 0 | 60.96 | FEASIBLE | 0.059 | 0/0 |
| cpsat:warm | 279.155 | 279.155 | 0 | 61.18 | FEASIBLE | 0.059 | 0/0 |
| cpsat:warmbest | 279.155 | 279.155 | 0 | 61.00 | FEASIBLE | 0.059 | 0/0 |
| heft_edf | 279.772 | 279.772 | 0 | 0.18 |  |  | 0/0 |
| pso | 279.772 | 279.772 | 0 | 16.43 |  |  | 0/0 |
| sa | 279.772 | 279.772 | 0 | 20.03 |  |  | 0/0 |
| cheap_portfolio | 279.772 | 279.772 | 0 | 0.50 |  |  | 0/0 |
| greedy | 280.910 | 280.910 | 0 | 0.06 |  |  | 0/0 |
| greedy_periodic | 280.910 | 280.910 | 0 | 0.06 |  |  | 0/0 |
| greedy_reserved | 280.910 | 280.910 | 0 | 0.06 |  |  | 0/0 |
| decomposed | 280.910 | 280.910 | 0 | 0.08 |  |  | 0/0 |
| heft | 278.786 | 278.786 | 224 | 0.07 |  |  | 0/0 |

### wl_sweep_shard / networks_control_mix_gempair  (477 ops, 196 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 42.484 | 78.131 | 0 | 61.92 | FEASIBLE | 0.0932 | 0/0 |
| cpsat | 42.912 | 78.992 | 0 | 61.11 | FEASIBLE | 0.1023 | 0/0 |
| cpsat:warmbest | 43.127 | 78.321 | 0 | 62.06 | FEASIBLE | 0.1067 | 0/0 |
| heft_edf | 43.840 | 77.961 | 0 | 0.85 |  |  | 0/0 |
| pso | 43.840 | 77.961 | 0 | 20.09 |  |  | 0/0 |
| sa | 43.840 | 77.961 | 0 | 20.04 |  |  | 0/0 |
| cheap_portfolio | 43.840 | 77.961 | 0 | 1.25 |  |  | 0/0 |
| greedy | 48.009 | 78.278 | 0 | 0.07 |  |  | 0/0 |
| decomposed | 54.260 | 78.283 | 0 | 0.09 |  |  | 0/0 |
| greedy_periodic | 42.579 | 78.278 | 51 | 0.08 |  |  | 0/0 |
| greedy_reserved | 40.117 | 80.638 | 71 | 0.09 |  |  | 0/0 |
| heft | 42.235 | 77.961 | 84 | 0.06 |  |  | 0/0 |

### wl_sweep_shard / networks_control_mix_hetero  (477 ops, 196 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 38.429 | 121.587 | 0 | 20.24 |  |  | 0/0 |
| sa | 39.018 | 121.788 | 0 | 14.99 |  |  | 0/0 |
| cpsat:warmbest | 39.172 | 121.488 | 0 | 61.15 | FEASIBLE | 0.3245 | 0/0 |
| cpsat:warm | 39.563 | 121.483 | 0 | 61.35 | FEASIBLE | 0.3311 | 0/0 |
| heft_edf | 40.429 | 121.469 | 0 | 0.51 |  |  | 0/0 |
| cheap_portfolio | 40.429 | 121.469 | 0 | 0.80 |  |  | 0/0 |
| cpsat | 45.690 | 124.262 | 0 | 60.82 | FEASIBLE | 0.4205 | 0/0 |
| decomposed | 46.447 | 121.499 | 0 | 0.06 |  |  | 0/0 |
| greedy | 47.479 | 121.773 | 0 | 0.06 |  |  | 0/0 |
| heft | 38.359 | 121.469 | 19 | 0.03 |  |  | 0/0 |
| greedy_periodic | 45.546 | 121.773 | 56 | 0.07 |  |  | 0/0 |
| greedy_reserved | 43.928 | 122.911 | 62 | 0.07 |  |  | 0/0 |

### wl_sweep_shard / networks_control_mix_quad  (477 ops, 196 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 23.843 | 120.416 | 0 | 61.68 | FEASIBLE | 0.1026 | 0/0 |
| cpsat:warm | 24.001 | 120.416 | 0 | 61.42 | FEASIBLE | 0.1085 | 0/0 |
| heft_edf | 24.117 | 120.401 | 0 | 0.87 |  |  | 0/0 |
| pso | 24.117 | 120.401 | 0 | 20.66 |  |  | 0/0 |
| sa | 24.117 | 120.401 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 24.117 | 120.401 | 0 | 1.45 |  |  | 0/0 |
| decomposed | 28.289 | 120.314 | 0 | 0.12 |  |  | 0/0 |
| greedy | 29.709 | 120.405 | 0 | 0.12 |  |  | 0/0 |
| cpsat | 34.476 | 124.036 | 0 | 61.20 | FEASIBLE | 0.3785 | 0/0 |
| heft | 23.960 | 120.314 | 5 | 0.06 |  |  | 0/0 |
| greedy_periodic | 26.904 | 120.405 | 25 | 0.13 |  |  | 0/0 |
| greedy_reserved | 27.339 | 121.773 | 38 | 0.15 |  |  | 0/0 |

### wl_sweep_shard / networks_control_mix_rvvpair  (477 ops, 196 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 51.558 | 125.390 | 0 | 60.84 | FEASIBLE | 0.1318 | 0/0 |
| cpsat:warm | 51.886 | 126.196 | 0 | 61.83 | FEASIBLE | 0.1373 | 0/0 |
| cpsat:warmbest | 52.196 | 124.714 | 0 | 61.48 | FEASIBLE | 0.1424 | 0/0 |
| pso | 57.819 | 124.081 | 0 | 20.02 |  |  | 0/0 |
| decomposed | 59.343 | 124.045 | 0 | 0.09 |  |  | 0/0 |
| sa | 59.343 | 124.045 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 59.343 | 124.045 | 0 | 1.25 |  |  | 0/0 |
| heft_edf | 60.410 | 124.036 | 0 | 0.84 |  |  | 0/0 |
| greedy | 61.384 | 124.045 | 34 | 0.08 |  |  | 0/0 |
| greedy_periodic | 53.020 | 124.045 | 46 | 0.08 |  |  | 0/0 |
| greedy_reserved | 58.589 | 129.323 | 59 | 0.09 |  |  | 0/0 |
| heft | 56.710 | 124.041 | 66 | 0.06 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_chain_gempair  (414 ops, 414 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 582.550 | 582.550 | 0 | 0.04 |  |  | 0/0 |
| heft_edf | 582.550 | 582.550 | 0 | 0.53 |  |  | 0/0 |
| pso | 582.550 | 582.550 | 0 | 8.13 |  |  | 0/0 |
| sa | 582.550 | 582.550 | 0 | 18.98 |  |  | 0/0 |
| cheap_portfolio | 582.550 | 582.550 | 0 | 0.74 |  |  | 0/0 |
| cpsat | 582.619 | 582.619 | 0 | 4.00 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 582.619 | 582.619 | 0 | 1.92 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 582.619 | 582.619 | 0 | 1.30 | OPTIMAL | 0.0 | 0/0 |
| greedy | 582.965 | 582.965 | 0 | 0.04 |  |  | 0/0 |
| greedy_periodic | 582.965 | 582.965 | 0 | 0.04 |  |  | 0/0 |
| greedy_reserved | 582.965 | 582.965 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 582.973 | 582.973 | 0 | 0.05 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_chain_hetero  (414 ops, 414 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 435.791 | 435.791 | 0 | 0.26 |  |  | 0/0 |
| pso | 435.791 | 435.791 | 0 | 4.39 |  |  | 0/0 |
| sa | 435.791 | 435.791 | 0 | 11.01 |  |  | 0/0 |
| cheap_portfolio | 435.791 | 435.791 | 0 | 0.43 |  |  | 0/0 |
| heft | 435.791 | 435.791 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 435.871 | 435.871 | 0 | 0.04 |  |  | 0/0 |
| cpsat:warm | 435.877 | 435.877 | 0 | 29.42 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 435.877 | 435.877 | 0 | 20.01 | OPTIMAL | 0.0 | 0/0 |
| cpsat | 436.074 | 436.074 | 0 | 60.96 | FEASIBLE | 0.0098 | 0/0 |
| greedy | 436.235 | 436.235 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 436.235 | 436.235 | 0 | 0.04 |  |  | 0/0 |
| greedy_reserved | 436.235 | 436.235 | 0 | 0.04 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_chain_quad  (414 ops, 414 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| decomposed | 425.976 | 425.976 | 0 | 0.07 |  |  | 0/0 |
| heft | 425.976 | 425.976 | 0 | 0.05 |  |  | 0/0 |
| heft_edf | 425.976 | 425.976 | 0 | 0.59 |  |  | 0/0 |
| pso | 425.976 | 425.976 | 0 | 9.58 |  |  | 0/0 |
| sa | 425.976 | 425.976 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 425.976 | 425.976 | 0 | 0.87 |  |  | 0/0 |
| greedy | 426.016 | 426.016 | 0 | 0.06 |  |  | 0/0 |
| greedy_periodic | 426.016 | 426.016 | 0 | 0.06 |  |  | 0/0 |
| greedy_reserved | 426.016 | 426.016 | 0 | 0.06 |  |  | 0/0 |
| cpsat:warm | 426.044 | 426.044 | 0 | 7.89 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 426.044 | 426.044 | 0 | 9.61 | OPTIMAL | 0.0 | 0/0 |
| cpsat | 426.044 | 426.044 | 0 | 31.62 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep_shard / networks_depth_chain_rvvpair  (414 ops, 414 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 213.894 | 213.894 | 0 | 14.56 |  |  | 0/0 |
| heft_edf | 213.899 | 213.899 | 0 | 0.52 |  |  | 0/0 |
| sa | 213.899 | 213.899 | 0 | 17.97 |  |  | 0/0 |
| cheap_portfolio | 213.899 | 213.899 | 0 | 0.73 |  |  | 0/0 |
| heft | 213.905 | 213.905 | 0 | 0.04 |  |  | 0/0 |
| greedy | 213.911 | 213.911 | 0 | 0.04 |  |  | 0/0 |
| greedy_periodic | 213.911 | 213.911 | 0 | 0.04 |  |  | 0/0 |
| greedy_reserved | 213.911 | 213.911 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 213.911 | 213.911 | 0 | 0.05 |  |  | 0/0 |
| cpsat | 213.958 | 213.958 | 0 | 19.15 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 213.958 | 213.958 | 0 | 2.46 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 213.958 | 213.958 | 0 | 3.26 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep_shard / networks_depth_contended_gempair  (695 ops, 414 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 35.331 | 583.035 | 0 | 62.92 | FEASIBLE | 0.0505 | 0/0 |
| cpsat | 35.360 | 591.321 | 0 | 60.87 | FEASIBLE | 0.0513 | 0/0 |
| sa | 35.595 | 582.550 | 0 | 20.03 |  |  | 0/0 |
| pso | 35.707 | 582.550 | 0 | 20.38 |  |  | 0/0 |
| greedy_periodic | 35.813 | 582.965 | 0 | 0.09 |  |  | 0/0 |
| greedy_reserved | 35.813 | 600.756 | 0 | 0.11 |  |  | 0/0 |
| cheap_portfolio | 35.813 | 582.965 | 0 | 2.00 |  |  | 0/0 |
| greedy | 36.783 | 582.965 | 0 | 0.11 |  |  | 0/0 |
| cpsat:warm | 47.122 | 590.137 | 0 | 61.52 | FEASIBLE | 0.2881 | 0/0 |
| heft | 80.578 | 582.550 | 0 | 0.11 |  |  | 0/0 |
| heft_edf | 80.578 | 582.550 | 0 | 1.43 |  |  | 0/0 |
| decomposed | 80.579 | 582.973 | 0 | 0.14 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_contended_hetero  (695 ops, 414 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| sa | 38.214 | 435.981 | 0 | 20.00 |  |  | 0/0 |
| pso | 39.064 | 436.160 | 0 | 9.79 |  |  | 0/0 |
| cpsat:warmbest | 39.518 | 438.224 | 0 | 61.97 | FEASIBLE | 0.3427 | 0/0 |
| greedy_periodic | 41.834 | 436.235 | 0 | 0.09 |  |  | 0/0 |
| greedy_reserved | 41.834 | 442.023 | 0 | 0.10 |  |  | 0/0 |
| cheap_portfolio | 41.834 | 436.235 | 0 | 1.07 |  |  | 0/0 |
| greedy | 44.254 | 436.235 | 0 | 0.10 |  |  | 0/0 |
| cpsat | 53.974 | 444.645 | 0 | 60.87 | FEASIBLE | 0.5183 | 0/0 |
| cpsat:warm | 54.656 | 444.649 | 0 | 61.25 | FEASIBLE | 0.5248 | 0/0 |
| heft_edf | 79.914 | 435.797 | 0 | 0.64 |  |  | 0/0 |
| heft | 80.071 | 435.791 | 0 | 0.05 |  |  | 0/0 |
| decomposed | 82.413 | 435.871 | 0 | 0.09 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_contended_quad  (695 ops, 414 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 24.383 | 426.208 | 0 | 21.14 |  |  | 0/0 |
| sa | 24.518 | 426.014 | 0 | 20.04 |  |  | 0/0 |
| cpsat:warmbest | 24.819 | 426.084 | 0 | 62.14 | FEASIBLE | 0.1349 | 0/0 |
| greedy_periodic | 26.876 | 426.016 | 0 | 0.15 |  |  | 0/0 |
| greedy_reserved | 26.876 | 436.235 | 0 | 0.16 |  |  | 0/0 |
| cheap_portfolio | 26.876 | 426.016 | 0 | 2.07 |  |  | 0/0 |
| greedy | 29.378 | 426.016 | 0 | 0.16 |  |  | 0/0 |
| cpsat:warm | 34.395 | 426.083 | 0 | 62.55 | FEASIBLE | 0.3781 | 0/0 |
| heft | 36.023 | 425.976 | 0 | 0.11 |  |  | 0/0 |
| heft_edf | 36.023 | 426.015 | 0 | 1.34 |  |  | 0/0 |
| decomposed | 37.851 | 425.976 | 0 | 0.16 |  |  | 0/0 |
| cpsat | 40.804 | 502.312 | 0 | 61.20 | FEASIBLE | 0.4756 | 0/0 |

### wl_sweep_shard / networks_depth_contended_rvvpair  (695 ops, 414 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 46.105 | 213.975 | 0 | 61.80 | FEASIBLE | 0.0422 | 0/0 |
| sa | 46.455 | 213.943 | 0 | 20.01 |  |  | 0/0 |
| cpsat:warm | 46.512 | 213.965 | 0 | 61.67 | FEASIBLE | 0.0506 | 0/0 |
| pso | 46.698 | 214.103 | 0 | 20.20 |  |  | 0/0 |
| greedy_periodic | 46.795 | 213.911 | 0 | 0.11 |  |  | 0/0 |
| cheap_portfolio | 46.795 | 213.911 | 0 | 1.73 |  |  | 0/0 |
| cpsat | 58.421 | 219.765 | 0 | 61.04 | FEASIBLE | 0.2441 | 0/0 |
| greedy | 73.489 | 213.911 | 0 | 0.12 |  |  | 0/0 |
| heft_edf | 104.055 | 213.899 | 0 | 1.16 |  |  | 0/0 |
| heft | 104.208 | 213.905 | 0 | 0.10 |  |  | 0/0 |
| decomposed | 104.767 | 213.911 | 0 | 0.12 |  |  | 0/0 |
| greedy_reserved | 46.795 | 258.156 | 37 | 0.12 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_nav_gempair  (526 ops, 526 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 582.550 | 582.550 | 0 | 0.92 |  |  | 0/0 |
| pso | 582.550 | 582.550 | 0 | 11.56 |  |  | 0/0 |
| sa | 582.550 | 582.550 | 0 | 20.00 |  |  | 0/0 |
| cheap_portfolio | 582.550 | 582.550 | 0 | 1.26 |  |  | 0/0 |
| cpsat | 582.619 | 582.619 | 0 | 4.19 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 582.619 | 582.619 | 0 | 1.90 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 582.619 | 582.619 | 0 | 1.69 | OPTIMAL | 0.0 | 0/0 |
| greedy | 582.965 | 582.965 | 0 | 0.06 |  |  | 0/0 |
| greedy_periodic | 582.965 | 582.965 | 0 | 0.06 |  |  | 0/0 |
| greedy_reserved | 582.965 | 582.965 | 0 | 0.07 |  |  | 0/0 |
| decomposed | 582.973 | 582.973 | 0 | 0.07 |  |  | 0/0 |
| heft | 582.550 | 582.550 | 7 | 0.07 |  |  | 0/0 |

### wl_sweep_shard / networks_depth_nav_hetero  (526 ops, 526 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 435.791 | 435.791 | 0 | 0.42 |  |  | 0/0 |
| pso | 435.791 | 435.791 | 0 | 6.37 |  |  | 0/0 |
| sa | 435.791 | 435.791 | 0 | 15.53 |  |  | 0/0 |
| cheap_portfolio | 435.791 | 435.791 | 0 | 0.69 |  |  | 0/0 |
| heft | 435.791 | 435.791 | 0 | 0.03 |  |  | 0/0 |
| decomposed | 435.871 | 435.871 | 0 | 0.07 |  |  | 0/0 |
| cpsat:warm | 435.877 | 435.877 | 0 | 17.44 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 435.877 | 435.877 | 0 | 10.26 | OPTIMAL | 0.0 | 0/0 |
| greedy | 436.235 | 436.235 | 0 | 0.06 |  |  | 0/0 |
| greedy_periodic | 436.235 | 436.235 | 0 | 0.06 |  |  | 0/0 |
| greedy_reserved | 436.235 | 436.235 | 0 | 0.06 |  |  | 0/0 |
| cpsat | 436.560 | 436.560 | 0 | 60.91 | FEASIBLE | 0.0129 | 0/0 |

### wl_sweep_shard / networks_depth_nav_quad  (526 ops, 526 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| decomposed | 425.976 | 425.976 | 0 | 0.10 |  |  | 0/0 |
| heft | 425.976 | 425.976 | 0 | 0.07 |  |  | 0/0 |
| heft_edf | 425.976 | 425.976 | 0 | 0.85 |  |  | 0/0 |
| pso | 425.976 | 425.976 | 0 | 12.84 |  |  | 0/0 |
| sa | 425.976 | 425.976 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 425.976 | 425.976 | 0 | 1.31 |  |  | 0/0 |
| greedy | 426.016 | 426.016 | 0 | 0.10 |  |  | 0/0 |
| greedy_periodic | 426.016 | 426.016 | 0 | 0.10 |  |  | 0/0 |
| greedy_reserved | 426.016 | 426.016 | 0 | 0.11 |  |  | 0/0 |
| cpsat | 426.044 | 426.044 | 0 | 40.19 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 426.044 | 426.044 | 0 | 6.91 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 426.044 | 426.044 | 0 | 12.76 | OPTIMAL | 0.0 | 0/0 |

### wl_sweep_shard / networks_depth_nav_rvvpair  (526 ops, 526 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 213.899 | 213.899 | 0 | 14.91 |  |  | 0/0 |
| heft_edf | 213.899 | 213.899 | 0 | 0.94 |  |  | 0/0 |
| sa | 213.899 | 213.899 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 213.899 | 213.899 | 0 | 1.29 |  |  | 0/0 |
| greedy | 213.911 | 213.911 | 0 | 0.07 |  |  | 0/0 |
| greedy_periodic | 213.911 | 213.911 | 0 | 0.07 |  |  | 0/0 |
| greedy_reserved | 213.911 | 213.911 | 0 | 0.07 |  |  | 0/0 |
| decomposed | 213.911 | 213.911 | 0 | 0.07 |  |  | 0/0 |
| cpsat | 213.958 | 213.958 | 0 | 22.65 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 213.958 | 213.958 | 0 | 4.25 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 213.958 | 213.958 | 0 | 3.56 | OPTIMAL | 0.0 | 0/0 |
| heft | 213.905 | 213.905 | 5 | 0.07 |  |  | 0/0 |

### wl_sweep_shard / networks_perception_heavy_gempair  (228 ops, 28 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 122.857 | 122.857 | 0 | 60.93 | FEASIBLE | 0.1307 | 0/0 |
| cpsat:warm | 122.858 | 122.858 | 0 | 61.02 | FEASIBLE | 0.1337 | 0/0 |
| cpsat:warmbest | 122.858 | 122.858 | 0 | 60.77 | FEASIBLE | 0.1298 | 0/0 |
| pso | 127.614 | 127.614 | 0 | 7.58 |  |  | 0/0 |
| heft_edf | 128.172 | 128.172 | 0 | 0.08 |  |  | 0/0 |
| sa | 128.172 | 128.172 | 0 | 7.71 |  |  | 0/0 |
| cheap_portfolio | 128.172 | 128.172 | 0 | 0.18 |  |  | 0/0 |
| greedy | 132.730 | 132.730 | 0 | 0.02 |  |  | 0/0 |
| greedy_periodic | 132.730 | 132.730 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 132.730 | 132.730 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 141.055 | 141.055 | 0 | 0.02 |  |  | 0/0 |
| heft | 122.857 | 122.857 | 12 | 0.02 |  |  | 0/0 |

### wl_sweep_shard / networks_perception_heavy_hetero  (228 ops, 28 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 108.360 | 108.360 | 0 | 60.93 | FEASIBLE | 0.2694 | 0/0 |
| cpsat | 108.502 | 108.502 | 0 | 61.02 | FEASIBLE | 0.3141 | 0/0 |
| cpsat:warmbest | 109.590 | 109.590 | 0 | 60.64 | FEASIBLE | 0.1631 | 0/0 |
| sa | 111.403 | 111.403 | 0 | 14.14 |  |  | 0/0 |
| pso | 111.551 | 111.551 | 0 | 4.07 |  |  | 0/0 |
| heft | 115.882 | 115.882 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 115.882 | 115.882 | 0 | 0.03 |  |  | 0/0 |
| cheap_portfolio | 115.882 | 115.882 | 0 | 0.09 |  |  | 0/0 |
| greedy | 122.883 | 122.883 | 0 | 0.01 |  |  | 0/0 |
| decomposed | 122.883 | 122.883 | 0 | 0.02 |  |  | 0/0 |
| greedy_reserved | 129.413 | 129.413 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 129.925 | 129.925 | 0 | 0.01 |  |  | 0/0 |

### wl_sweep_shard / networks_perception_heavy_quad  (228 ops, 28 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 74.397 | 74.397 | 0 | 60.81 | FEASIBLE | 0.0761 | 0/0 |
| pso | 75.006 | 75.006 | 0 | 11.93 |  |  | 0/0 |
| cpsat:warm | 75.242 | 75.242 | 0 | 60.96 | FEASIBLE | 0.0865 | 0/0 |
| cpsat | 75.381 | 75.381 | 0 | 61.05 | FEASIBLE | 0.0878 | 0/0 |
| sa | 75.529 | 75.529 | 0 | 20.00 |  |  | 0/0 |
| heft | 77.581 | 77.581 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 77.581 | 77.581 | 0 | 0.05 |  |  | 0/0 |
| cheap_portfolio | 77.581 | 77.581 | 0 | 0.18 |  |  | 0/0 |
| greedy | 83.015 | 83.015 | 0 | 0.03 |  |  | 0/0 |
| decomposed | 83.015 | 83.015 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 82.013 | 82.013 | 19 | 0.03 |  |  | 0/0 |
| greedy_reserved | 82.013 | 82.013 | 19 | 0.03 |  |  | 0/0 |

### wl_sweep_shard / networks_perception_heavy_rvvpair  (228 ops, 28 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 172.032 | 172.032 | 0 | 16.34 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 172.032 | 172.032 | 0 | 21.65 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 172.032 | 172.032 | 0 | 28.21 | OPTIMAL | 0.0 | 0/0 |
| sa | 172.055 | 172.055 | 0 | 18.25 |  |  | 0/0 |
| heft | 172.243 | 172.243 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 172.243 | 172.243 | 0 | 0.04 |  |  | 0/0 |
| pso | 172.243 | 172.243 | 0 | 3.29 |  |  | 0/0 |
| cheap_portfolio | 172.243 | 172.243 | 0 | 0.13 |  |  | 0/0 |
| greedy_periodic | 178.879 | 178.879 | 0 | 0.02 |  |  | 0/0 |
| greedy_reserved | 178.881 | 178.881 | 0 | 0.02 |  |  | 0/0 |
| greedy | 182.216 | 182.216 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 182.216 | 182.216 | 0 | 0.03 |  |  | 0/0 |

### wl_sweep_shard / networks_saturation_gempair  (619 ops, 338 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warm | 120.153 | 120.153 | 0 | 62.37 | FEASIBLE | 0.0753 | 0/0 |
| cpsat:warmbest | 120.266 | 120.266 | 0 | 63.01 | FEASIBLE | 0.0761 | 0/0 |
| cpsat | 122.435 | 122.435 | 0 | 61.01 | FEASIBLE | 0.0925 | 0/0 |
| pso | 132.444 | 132.444 | 0 | 20.13 |  |  | 0/0 |
| heft_edf | 137.135 | 137.135 | 0 | 0.87 |  |  | 0/0 |
| sa | 137.135 | 137.135 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 137.135 | 137.135 | 0 | 1.85 |  |  | 0/0 |
| greedy | 138.842 | 138.842 | 0 | 0.11 |  |  | 0/0 |
| decomposed | 140.133 | 140.133 | 130 | 0.14 |  |  | 0/0 |
| greedy_periodic | 123.892 | 123.892 | 291 | 0.31 |  |  | 0/0 |
| greedy_reserved | 161.819 | 161.819 | 292 | 0.31 |  |  | 0/0 |
| heft | 116.272 | 116.678 | 326 | 0.11 |  |  | 0/0 |

### wl_sweep_shard / networks_saturation_hetero  (619 ops, 338 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 95.876 | 100.349 | 0 | 20.38 |  |  | 0/0 |
| cpsat:warmbest | 100.158 | 104.870 | 0 | 61.35 | FEASIBLE | 0.3713 | 0/0 |
| sa | 102.324 | 102.324 | 0 | 20.01 |  |  | 0/0 |
| cpsat:warm | 109.761 | 111.636 | 0 | 61.77 | FEASIBLE | 0.4263 | 0/0 |
| heft_edf | 111.488 | 111.488 | 0 | 0.86 |  |  | 0/0 |
| cheap_portfolio | 111.488 | 111.488 | 0 | 1.34 |  |  | 0/0 |
| greedy | 115.252 | 115.252 | 0 | 0.10 |  |  | 0/0 |
| decomposed | 118.352 | 118.352 | 0 | 0.09 |  |  | 0/0 |
| cpsat | 125.553 | 125.553 | 0 | 60.99 | FEASIBLE | 0.4984 | 0/0 |
| heft | 94.525 | 100.025 | 175 | 0.05 |  |  | 0/0 |
| greedy_periodic | 106.060 | 111.385 | 227 | 0.12 |  |  | 0/0 |
| greedy_reserved | 93.987 | 153.993 | 241 | 0.12 |  |  | 0/0 |

### wl_sweep_shard / networks_saturation_quad  (619 ops, 338 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 53.285 | 98.676 | 0 | 63.28 | FEASIBLE | 0.1707 | 0/0 |
| cpsat:warm | 53.465 | 98.676 | 0 | 62.56 | FEASIBLE | 0.1735 | 0/0 |
| heft_edf | 53.663 | 98.668 | 0 | 1.34 |  |  | 0/0 |
| pso | 53.663 | 98.668 | 0 | 20.68 |  |  | 0/0 |
| sa | 53.663 | 98.668 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 53.663 | 98.668 | 0 | 2.42 |  |  | 0/0 |
| decomposed | 68.341 | 98.629 | 0 | 0.18 |  |  | 0/0 |
| greedy | 68.386 | 98.669 | 0 | 0.20 |  |  | 0/0 |
| cpsat | 82.492 | 103.575 | 0 | 61.39 | FEASIBLE | 0.4643 | 0/0 |
| heft | 52.478 | 98.629 | 10 | 0.09 |  |  | 0/0 |
| greedy_periodic | 59.533 | 98.669 | 196 | 0.29 |  |  | 0/0 |
| greedy_reserved | 61.020 | 100.436 | 196 | 0.32 |  |  | 0/0 |

### wl_sweep_shard / networks_saturation_rvvpair  (619 ops, 338 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 149.780 | 149.780 | 0 | 60.73 | FEASIBLE | 0.1858 | 0/0 |
| cpsat:warmbest | 150.922 | 150.922 | 0 | 62.99 | FEASIBLE | 0.1919 | 0/0 |
| cpsat:warm | 153.794 | 153.794 | 0 | 62.41 | FEASIBLE | 0.207 | 0/0 |
| pso | 154.930 | 154.930 | 0 | 20.93 |  |  | 0/0 |
| sa | 156.365 | 156.365 | 0 | 20.01 |  |  | 0/0 |
| decomposed | 158.370 | 158.370 | 0 | 0.14 |  |  | 0/0 |
| cheap_portfolio | 158.370 | 158.370 | 0 | 2.26 |  |  | 0/0 |
| heft_edf | 160.684 | 160.684 | 0 | 1.57 |  |  | 0/0 |
| greedy | 161.748 | 161.748 | 0 | 0.12 |  |  | 0/0 |
| greedy_reserved | 186.350 | 186.350 | 235 | 0.17 |  |  | 0/0 |
| greedy_periodic | 145.845 | 145.845 | 257 | 0.15 |  |  | 0/0 |
| heft | 142.541 | 142.574 | 275 | 0.11 |  |  | 0/0 |

### wl_sweep_shard / networks_scale_ladder_gempair  (211 ops, 0 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 70.775 | 70.775 | 0 | 6.53 |  |  | 0/0 |
| cpsat | 70.812 | 70.812 | 0 | 2.79 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 70.812 | 70.812 | 0 | 3.44 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 70.812 | 70.812 | 0 | 5.49 | OPTIMAL | 0.0 | 0/0 |
| sa | 70.817 | 70.817 | 0 | 7.93 |  |  | 0/0 |
| heft | 70.962 | 70.962 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 70.962 | 70.962 | 0 | 0.02 |  |  | 0/0 |
| cheap_portfolio | 70.962 | 70.962 | 0 | 0.14 |  |  | 0/0 |
| greedy | 73.730 | 73.730 | 0 | 0.02 |  |  | 0/0 |
| greedy_periodic | 73.730 | 73.730 | 0 | 0.02 |  |  | 0/0 |
| greedy_reserved | 73.730 | 73.730 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 73.730 | 73.730 | 0 | 0.05 |  |  | 0/0 |

### wl_sweep_shard / networks_scale_ladder_hetero  (211 ops, 0 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 61.505 | 61.505 | 0 | 60.56 | FEASIBLE | 0.0149 | 0/0 |
| cpsat:warm | 61.778 | 61.778 | 0 | 60.62 | FEASIBLE | 0.0193 | 0/0 |
| pso | 61.932 | 61.932 | 0 | 4.37 |  |  | 0/0 |
| sa | 62.328 | 62.328 | 0 | 6.38 |  |  | 0/0 |
| cpsat | 62.648 | 62.648 | 0 | 60.87 | FEASIBLE | 0.0342 | 0/0 |
| heft | 63.982 | 63.982 | 0 | 0.01 |  |  | 0/0 |
| heft_edf | 63.982 | 63.982 | 0 | 0.01 |  |  | 0/0 |
| cheap_portfolio | 63.982 | 63.982 | 0 | 0.10 |  |  | 0/0 |
| greedy | 68.529 | 68.529 | 0 | 0.01 |  |  | 0/0 |
| greedy_periodic | 68.529 | 68.529 | 0 | 0.01 |  |  | 0/0 |
| greedy_reserved | 68.529 | 68.529 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 68.529 | 68.529 | 0 | 0.03 |  |  | 0/0 |

### wl_sweep_shard / networks_scale_ladder_quad  (211 ops, 0 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 46.663 | 46.663 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 46.663 | 46.663 | 0 | 0.02 |  |  | 0/0 |
| pso | 46.663 | 46.663 | 0 | 4.04 |  |  | 0/0 |
| sa | 46.663 | 46.663 | 0 | 8.52 |  |  | 0/0 |
| cheap_portfolio | 46.663 | 46.663 | 0 | 0.21 |  |  | 0/0 |
| cpsat | 46.672 | 46.672 | 0 | 11.71 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 46.672 | 46.672 | 0 | 3.02 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 46.672 | 46.672 | 0 | 4.64 | OPTIMAL | 0.0 | 0/0 |
| greedy | 56.632 | 56.632 | 0 | 0.03 |  |  | 0/0 |
| greedy_periodic | 56.632 | 56.632 | 0 | 0.03 |  |  | 0/0 |
| greedy_reserved | 56.632 | 56.632 | 0 | 0.04 |  |  | 0/0 |
| decomposed | 56.632 | 56.632 | 0 | 0.06 |  |  | 0/0 |

### wl_sweep_shard / networks_scale_ladder_rvvpair  (211 ops, 0 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| sa | 74.365 | 74.365 | 0 | 11.84 |  |  | 0/0 |
| cpsat | 74.368 | 74.368 | 0 | 2.80 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 74.368 | 74.368 | 0 | 3.54 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 74.368 | 74.368 | 0 | 1.96 | OPTIMAL | 0.0 | 0/0 |
| pso | 74.458 | 74.458 | 0 | 3.85 |  |  | 0/0 |
| heft | 74.673 | 74.673 | 0 | 0.02 |  |  | 0/0 |
| heft_edf | 74.673 | 74.673 | 0 | 0.02 |  |  | 0/0 |
| cheap_portfolio | 74.673 | 74.673 | 0 | 0.14 |  |  | 0/0 |
| greedy | 74.793 | 74.793 | 0 | 0.02 |  |  | 0/0 |
| greedy_periodic | 74.793 | 74.793 | 0 | 0.02 |  |  | 0/0 |
| greedy_reserved | 74.793 | 74.793 | 0 | 0.02 |  |  | 0/0 |
| decomposed | 74.793 | 74.793 | 0 | 0.05 |  |  | 0/0 |

### wl_sweep_shard / networks_tight_loop_gempair  (308 ops, 308 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 71.382 | 71.382 | 140 | 12.67 |  |  | 0/0 |
| greedy | 71.890 | 71.890 | 140 | 0.04 |  |  | 0/0 |
| greedy_periodic | 71.890 | 71.890 | 140 | 0.04 |  |  | 0/0 |
| greedy_reserved | 71.890 | 71.890 | 140 | 0.03 |  |  | 0/0 |
| sa | 71.890 | 71.890 | 140 | 9.91 |  |  | 0/0 |
| cheap_portfolio | 71.890 | 71.890 | 140 | 0.75 |  |  | 0/0 |
| heft_edf | 72.101 | 72.101 | 140 | 0.47 |  |  | 0/0 |
| heft | 72.248 | 72.248 | 175 | 0.03 |  |  | 0/0 |
| decomposed | 71.058 | 71.058 | 308 | 0.14 |  |  | 0/0 |
| cpsat | FAILED | | | 0.623 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 1.067 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 1.528 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep_shard / networks_tight_loop_hetero  (308 ops, 308 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 71.448 | 71.448 | 140 | 12.41 |  |  | 0/0 |
| heft_edf | 72.554 | 72.554 | 140 | 0.18 |  |  | 0/0 |
| sa | 72.554 | 72.554 | 140 | 5.57 |  |  | 0/0 |
| cheap_portfolio | 72.554 | 72.554 | 140 | 0.38 |  |  | 0/0 |
| heft | 75.160 | 75.160 | 140 | 0.01 |  |  | 0/0 |
| greedy | 77.071 | 77.071 | 140 | 0.03 |  |  | 0/0 |
| greedy_periodic | 77.071 | 77.071 | 140 | 0.03 |  |  | 0/0 |
| greedy_reserved | 77.071 | 77.071 | 140 | 0.04 |  |  | 0/0 |
| decomposed | 71.081 | 71.081 | 308 | 0.08 |  |  | 0/0 |
| cpsat | FAILED | | | 0.583 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.688 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 1.038 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep_shard / networks_tight_loop_quad  (308 ops, 308 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 37.676 | 37.676 | 140 | 7.46 |  |  | 0/0 |
| sa | 37.833 | 37.833 | 140 | 9.82 |  |  | 0/0 |
| heft | 38.024 | 38.024 | 140 | 0.03 |  |  | 0/0 |
| heft_edf | 38.024 | 38.024 | 140 | 0.42 |  |  | 0/0 |
| cheap_portfolio | 38.024 | 38.024 | 140 | 0.86 |  |  | 0/0 |
| greedy | 43.366 | 43.366 | 140 | 0.07 |  |  | 0/0 |
| greedy_periodic | 43.366 | 43.366 | 140 | 0.07 |  |  | 0/0 |
| greedy_reserved | 43.366 | 43.366 | 140 | 0.08 |  |  | 0/0 |
| decomposed | 46.593 | 46.593 | 308 | 0.20 |  |  | 0/0 |
| cpsat | FAILED | | | 0.659 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.894 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 1.714 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep_shard / networks_tight_loop_rvvpair  (308 ops, 308 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| greedy | 71.529 | 71.529 | 140 | 0.04 |  |  | 0/0 |
| greedy_periodic | 71.529 | 71.529 | 140 | 0.04 |  |  | 0/0 |
| greedy_reserved | 71.529 | 71.529 | 140 | 0.04 |  |  | 0/0 |
| pso | 71.529 | 71.529 | 140 | 8.47 |  |  | 0/0 |
| sa | 71.529 | 71.529 | 140 | 8.12 |  |  | 0/0 |
| cheap_portfolio | 71.529 | 71.529 | 140 | 0.72 |  |  | 0/0 |
| heft_edf | 71.734 | 71.734 | 140 | 0.46 |  |  | 0/0 |
| heft | 75.103 | 75.103 | 140 | 0.03 |  |  | 0/0 |
| decomposed | 70.551 | 70.551 | 308 | 0.11 |  |  | 0/0 |
| cpsat | FAILED | | | 0.595 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warm | FAILED | | | 0.789 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |
| cpsat:warmbest | FAILED | | | 1.034 | | | RuntimeError: cpsat returned INFEASIBLE with no solution |

### wl_sweep_shard / networks_vint_intro_gempair  (669 ops, 64 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 3754.235 | 3754.235 | 0 | 2.82 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3754.235 | 3754.235 | 0 | 3.95 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3754.235 | 3754.235 | 0 | 5.31 | OPTIMAL | 0.0 | 0/0 |
| pso | 3824.579 | 3824.579 | 0 | 16.33 |  |  | 0/0 |
| heft_edf | 3837.018 | 3837.018 | 0 | 0.68 |  |  | 0/0 |
| sa | 3837.018 | 3837.018 | 0 | 20.05 |  |  | 0/0 |
| cheap_portfolio | 3837.018 | 3837.018 | 0 | 1.19 |  |  | 0/0 |
| greedy | 4086.604 | 4086.604 | 0 | 0.08 |  |  | 0/0 |
| decomposed | 4086.604 | 4086.604 | 0 | 0.13 |  |  | 0/0 |
| greedy_reserved | 4072.363 | 4072.363 | 5 | 0.08 |  |  | 0/0 |
| greedy_periodic | 4086.604 | 4086.604 | 5 | 0.06 |  |  | 0/0 |
| heft | 3754.051 | 3754.051 | 64 | 0.14 |  |  | 0/0 |

### wl_sweep_shard / networks_vint_intro_hetero  (669 ops, 64 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat:warmbest | 3874.853 | 3874.853 | 0 | 61.39 | FEASIBLE | 0.0293 | 0/0 |
| cpsat | 3925.954 | 3925.954 | 0 | 60.80 | FEASIBLE | 0.0419 | 0/0 |
| cpsat:warm | 3942.295 | 3942.295 | 0 | 60.86 | FEASIBLE | 0.0459 | 0/0 |
| pso | 3979.525 | 3979.525 | 0 | 20.36 |  |  | 0/0 |
| sa | 4016.719 | 4016.719 | 0 | 20.02 |  |  | 0/0 |
| heft | 4160.689 | 4160.689 | 0 | 0.05 |  |  | 0/0 |
| heft_edf | 4160.689 | 4160.689 | 0 | 0.20 |  |  | 0/0 |
| cheap_portfolio | 4160.689 | 4160.689 | 0 | 0.58 |  |  | 0/0 |
| greedy | 4472.274 | 4472.274 | 0 | 0.08 |  |  | 0/0 |
| greedy_reserved | 4472.274 | 4472.274 | 0 | 0.08 |  |  | 0/0 |
| decomposed | 4472.274 | 4472.274 | 0 | 0.10 |  |  | 0/0 |
| greedy_periodic | 4472.390 | 4472.390 | 0 | 0.08 |  |  | 0/0 |

### wl_sweep_shard / networks_vint_intro_quad  (669 ops, 64 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 3750.825 | 3750.825 | 0 | 0.12 |  |  | 0/0 |
| heft_edf | 3750.825 | 3750.825 | 0 | 0.47 |  |  | 0/0 |
| pso | 3750.825 | 3750.825 | 0 | 13.71 |  |  | 0/0 |
| sa | 3750.825 | 3750.825 | 0 | 20.04 |  |  | 0/0 |
| cheap_portfolio | 3750.825 | 3750.825 | 0 | 1.04 |  |  | 0/0 |
| cpsat:warm | 3751.009 | 3751.009 | 0 | 17.88 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3751.009 | 3751.009 | 0 | 14.77 | OPTIMAL | 0.0 | 0/0 |
| greedy | 3844.686 | 3844.686 | 0 | 0.10 |  |  | 0/0 |
| greedy_periodic | 3844.686 | 3844.686 | 0 | 0.11 |  |  | 0/0 |
| greedy_reserved | 3844.686 | 3844.686 | 0 | 0.11 |  |  | 0/0 |
| decomposed | 3844.686 | 3844.686 | 0 | 0.14 |  |  | 0/0 |
| cpsat | 4243.634 | 4243.634 | 0 | 61.00 | FEASIBLE | 0.1161 | 0/0 |

### wl_sweep_shard / networks_vint_intro_rvvpair  (669 ops, 64 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 11068.078 | 11068.078 | 0 | 0.62 |  |  | 0/0 |
| pso | 11068.078 | 11068.078 | 0 | 17.90 |  |  | 0/0 |
| sa | 11068.078 | 11068.078 | 0 | 20.03 |  |  | 0/0 |
| cheap_portfolio | 11068.078 | 11068.078 | 0 | 1.14 |  |  | 0/0 |
| cpsat | 11068.233 | 11068.233 | 0 | 4.93 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 11068.233 | 11068.233 | 0 | 2.12 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 11068.233 | 11068.233 | 0 | 3.04 | OPTIMAL | 0.0 | 0/0 |
| greedy_reserved | 12187.946 | 12187.946 | 0 | 0.09 |  |  | 0/0 |
| greedy | 12234.593 | 12234.593 | 0 | 0.08 |  |  | 0/0 |
| greedy_periodic | 12234.593 | 12234.593 | 0 | 0.08 |  |  | 0/0 |
| decomposed | 12234.593 | 12234.593 | 0 | 0.13 |  |  | 0/0 |
| heft | 11068.078 | 11068.078 | 64 | 0.13 |  |  | 0/0 |

### wl_sweep_shard / networks_vint_multi_gempair  (801 ops, 196 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| cpsat | 3754.235 | 3754.235 | 0 | 8.47 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 3754.235 | 3754.235 | 0 | 8.26 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3754.235 | 3754.235 | 0 | 8.21 | OPTIMAL | 0.0 | 0/0 |
| heft_edf | 3754.817 | 3754.817 | 0 | 2.48 |  |  | 0/0 |
| pso | 3754.817 | 3754.817 | 0 | 21.71 |  |  | 0/0 |
| sa | 3754.817 | 3754.817 | 0 | 20.01 |  |  | 0/0 |
| cheap_portfolio | 3754.817 | 3754.817 | 0 | 3.21 |  |  | 0/0 |
| greedy | 4106.565 | 4106.565 | 0 | 0.12 |  |  | 0/0 |
| decomposed | 4106.570 | 4106.570 | 0 | 0.17 |  |  | 0/0 |
| greedy_reserved | 4108.925 | 4108.925 | 0 | 0.12 |  |  | 0/0 |
| greedy_periodic | 4106.565 | 4106.565 | 19 | 0.12 |  |  | 0/0 |
| heft | 3754.051 | 3754.051 | 196 | 0.21 |  |  | 0/0 |

### wl_sweep_shard / networks_vint_multi_hetero  (801 ops, 196 periodic, 2 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| pso | 4089.552 | 4089.552 | 0 | 20.26 |  |  | 0/0 |
| cpsat:warm | 4120.099 | 4120.099 | 0 | 61.21 | FEASIBLE | 0.087 | 0/0 |
| cpsat:warmbest | 4122.112 | 4122.112 | 0 | 62.52 | FEASIBLE | 0.0875 | 0/0 |
| heft_edf | 4160.689 | 4160.689 | 0 | 1.08 |  |  | 0/0 |
| sa | 4160.689 | 4160.689 | 0 | 20.00 |  |  | 0/0 |
| cheap_portfolio | 4160.689 | 4160.689 | 0 | 1.60 |  |  | 0/0 |
| cpsat | 4300.858 | 4300.858 | 0 | 60.92 | FEASIBLE | 0.1254 | 0/0 |
| decomposed | 4546.672 | 4546.672 | 0 | 0.12 |  |  | 0/0 |
| greedy | 4546.947 | 4546.947 | 0 | 0.11 |  |  | 0/0 |
| heft | 4160.689 | 4160.689 | 35 | 0.06 |  |  | 0/0 |
| greedy_periodic | 4546.947 | 4546.947 | 77 | 0.11 |  |  | 0/0 |
| greedy_reserved | 4548.022 | 4548.022 | 77 | 0.11 |  |  | 0/0 |

### wl_sweep_shard / networks_vint_multi_quad  (801 ops, 196 periodic, 6 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft | 3750.825 | 3750.825 | 0 | 0.15 |  |  | 0/0 |
| heft_edf | 3750.825 | 3750.825 | 0 | 1.75 |  |  | 0/0 |
| pso | 3750.825 | 3750.825 | 0 | 17.46 |  |  | 0/0 |
| sa | 3750.825 | 3750.825 | 0 | 20.07 |  |  | 0/0 |
| cheap_portfolio | 3750.825 | 3750.825 | 0 | 2.56 |  |  | 0/0 |
| cpsat:warm | 3751.009 | 3751.009 | 0 | 29.62 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 3751.009 | 3751.009 | 0 | 18.25 | OPTIMAL | 0.0 | 0/0 |
| decomposed | 4010.540 | 4010.540 | 0 | 0.18 |  |  | 0/0 |
| greedy | 4010.631 | 4010.631 | 0 | 0.16 |  |  | 0/0 |
| greedy_periodic | 4010.631 | 4010.631 | 0 | 0.16 |  |  | 0/0 |
| greedy_reserved | 4011.999 | 4011.999 | 0 | 0.17 |  |  | 0/0 |
| cpsat | 4365.088 | 4365.088 | 0 | 60.77 | FEASIBLE | 0.1407 | 0/0 |

### wl_sweep_shard / networks_vint_multi_rvvpair  (801 ops, 196 periodic, 3 combos, lanes gemmini_q31+V256D128_rvv)

| solver | objective ms | all-ops ms | misses | wall s | cpsat status | gap | prec/overlap viol |
|---|---|---|---|---|---|---|---|
| heft_edf | 11068.226 | 11068.226 | 0 | 2.31 |  |  | 0/0 |
| pso | 11068.226 | 11068.226 | 0 | 19.16 |  |  | 0/0 |
| sa | 11068.226 | 11068.226 | 0 | 20.02 |  |  | 0/0 |
| cheap_portfolio | 11068.226 | 11068.226 | 0 | 3.04 |  |  | 0/0 |
| cpsat | 11068.233 | 11068.233 | 0 | 22.55 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warm | 11068.233 | 11068.233 | 0 | 6.89 | OPTIMAL | 0.0 | 0/0 |
| cpsat:warmbest | 11068.233 | 11068.233 | 0 | 8.64 | OPTIMAL | 0.0 | 0/0 |
| greedy_reserved | 12293.142 | 12293.142 | 0 | 0.13 |  |  | 0/0 |
| greedy | 12334.900 | 12334.900 | 0 | 0.12 |  |  | 0/0 |
| greedy_periodic | 12334.900 | 12334.900 | 0 | 0.12 |  |  | 0/0 |
| decomposed | 12334.900 | 12334.900 | 0 | 0.17 |  |  | 0/0 |
| heft | 11068.078 | 11068.078 | 196 | 0.19 |  |  | 0/0 |
