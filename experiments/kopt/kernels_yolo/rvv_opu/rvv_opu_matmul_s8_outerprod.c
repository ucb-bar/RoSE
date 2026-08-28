/* source: curated */
/* algorithm: outerprod */
/* origin: Saturn OPU outer-product i8 matmul. Ported from
 *   hw/chipyard/generators/saturn/benchmarks/opu-gemm/kernel.h
 *   (branch origin/opu-fp8), function `i8_mm_bme_sq`, by Miles Rusch.
 *
 * Upstream's `i8_mm_bme_sq` iterates over mlmax-sized tiles of (i, j),
 * with a per-tile MAC loop that VOPACC's K i8 inputs into m1. We
 * preserve that tile-walking shape faithfully here, with three
 * adaptations for the modelblaster-flow matmul_s8 signature:
 *
 *   1. Layout normalization. Upstream takes `at` laid out as [K, M]
 *      (i.e., a's transpose, so unit-stride loads can sweep K rows of
 *      M lanes). Our `a` is [M, K] row-major. We transpose into a
 *      stack scratch of size K_TILE × mlmax bytes (well-sized because
 *      we only need one mlmax-tile of M at a time, not the full M).
 *      For transpose_b=1, b is [N, K] and we do the same flip into a
 *      K_TILE × mlmax scratch.
 *
 *   2. Requantize tail. The matmul_s8 ABI demands i8 output with a
 *      float-scale rescale (scale_a*scale_b/(scale_out*scale_div)).
 *      We drain each m1 tile into i32 scratch, then loop over the
 *      tile computing roundf(acc * total) + clamp + i8 store.
 *
 *   3. Edge handling. Upstream's narrow-N fallback uses a per-row
 *      variant (`i8_loop_k_general`). We don't replicate that — the
 *      tile loop here always uses square mlmax tiles, falling back
 *      to the embedded scalar reference for (M, N) shapes that
 *      aren't multiples of mlmax. Easier to read; perf-equivalent
 *      for the modelblaster-flow's typical shapes (M, N either small <= 8
 *      or pure multiples like 64).
 *
 * MAC body itself (VOPACC inner loop with two-way unroll) is byte-
 * for-byte upstream.
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include "saturn_opu.h"

/* Tile bounds. mlmax = VLEN/8 at runtime; we cap at OPU_MAX_TILE=64
 * (matches V512). K_TILE is the per-tile K reach; we transpose only
 * one mlmax row-strip of a (and optionally of b) at a time, so the
 * scratch is K_TILE × mlmax = 1024 × 64 = 64 KiB per buffer. That fits
 * in stack with margin; for deeper K we sub-tile the K loop. */
#define OPU_MAX_TILE   64
#define OPU_MAX_K     1024

/* Core OPU MAC + drain into i32 scratch.
 *
 * IMPORTANT: every tile here runs at FULL mlmax × mlmax, even when the
 * logical M_tile or N_tile is smaller. The caller zero-pads partial
 * tiles so that the extra rows/cols contribute zero MAC. Two reasons:
 *
 *   1. The OPU's outer-product unit derives its row/col counts from
 *      the vsetvli configured at each operand load — but spike's
 *      vector model has a single global vl CSR, so changing vsetvli
 *      between vs1 and vs2 loads doesn't propagate to VOPACC as
 *      "rows = ml, cols = vl". Always running at full mlmax sidesteps
 *      that and matches what real HW does cleanly: padded operands
 *      with one vsetvli per tile.
 *
 *   2. Upstream `i8_sq_loop_k` always uses ml=mlmax for the square
 *      tile path; partial handling went through a separate path that
 *      we don't replicate. Padding once per tile is cheap (a memset
 *      on the partial border rows of at_tile / b_tile) and removes
 *      the partial-VL complexity entirely.
 *
 * Caller supplies `at_tile` of [K, MLMAX] and `b_tile` of [K, MLMAX],
 * both zero-padded past the logical (M_tile, N_tile). Result drained
 * into `c_out_tile` as [MLMAX, MLMAX] i32; caller picks out the valid
 * M_tile × N_tile sub-block for the requantize tail.
 */
static inline void opu_tile_mac(int32_t *c_out_tile,
                                const int8_t *at_tile, const int8_t *b_tile,
                                size_t MLMAX, size_t K) {
    /* Seed m1 to zero. */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(MLMAX));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST(m1, v0);

    /* Two-way unrolled VOPACC over K, always at full mlmax. */
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(MLMAX));
    size_t k = 0;
    while (k + 2 <= K) {
        asm volatile("vle8.v v16, (%0)" : : "r"(&at_tile[k * MLMAX]));
        asm volatile("vle8.v v18, (%0)" : : "r"(&b_tile[k * MLMAX]));
        VOPACC(m1, v18, v16);
        k++;
        asm volatile("vle8.v v20, (%0)" : : "r"(&at_tile[k * MLMAX]));
        asm volatile("vle8.v v22, (%0)" : : "r"(&b_tile[k * MLMAX]));
        VOPACC(m1, v22, v20);
        k++;
    }
    if (k < K) {
        asm volatile("vle8.v v16, (%0)" : : "r"(&at_tile[k * MLMAX]));
        asm volatile("vle8.v v18, (%0)" : : "r"(&b_tile[k * MLMAX]));
        VOPACC(m1, v18, v16);
    }

    /* Drain MLMAX rows of MLMAX i32 elements each. */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(MLMAX));
    for (size_t r = 0; r < MLMAX; r++) {
        VMV_VR(v0, r, m1);
        asm volatile("vse32.v v0, (%0)" : : "r"(&c_out_tile[r * MLMAX]));
    }
}

static void matmul_s8_scalar_fallback(const int8_t *a, const int8_t *b,
                                      int8_t *output,
                                      int M, int K, int N,
                                      float scale_a, float scale_b,
                                      float scale_out,
                                      int transpose_b, float scale_div,
                                      int activation_min, int activation_max) {
    float total = (scale_a * scale_b) / (scale_out * scale_div);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t av = a[i * K + k];
                int8_t bv = transpose_b ? b[j * K + k] : b[k * N + j];
                acc += (int32_t)av * (int32_t)bv;
            }
            int32_t v = (int32_t)roundf((float)acc * total);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i * N + j] = (int8_t)v;
        }
    }
}

void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output,
                      int M, int K, int N,
                      float scale_a, float scale_b, float scale_out,
                      int transpose_b, float scale_div,
                      int activation_min, int activation_max) {
    /* Eligibility: just K within scratch. Partial tiles at the M and
     * N edges are handled by tiling with min(remaining, mlmax). */
    size_t mlmax;
    asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(mlmax));
    if (K > OPU_MAX_K || (int)mlmax > OPU_MAX_TILE) {
        matmul_s8_scalar_fallback(a, b, output, M, K, N,
                                  scale_a, scale_b, scale_out,
                                  transpose_b, scale_div,
                                  activation_min, activation_max);
        return;
    }
    const int MLMAX = (int)mlmax;
    const float total = (scale_a * scale_b) / (scale_out * scale_div);

    /* Per-tile padded scratch — always K × MLMAX rows so the inner MAC
     * stays at full mlmax. Partial M / N tiles fill the trailing rows
     * / cols with zero, which contribute no MAC to the result. The
     * c_tile drain reads MLMAX × MLMAX i32; we only requantize the
     * (M_tile, N_tile) sub-block back to output. */
    int8_t  at_tile[OPU_MAX_K * OPU_MAX_TILE];    /* [K, MLMAX] */
    int8_t  b_tile [OPU_MAX_K * OPU_MAX_TILE];    /* [K, MLMAX] */
    int32_t c_tile [OPU_MAX_TILE * OPU_MAX_TILE]; /* [MLMAX, MLMAX] */

    for (int i0 = 0; i0 < M; i0 += MLMAX) {
        const int M_tile = (M - i0 < MLMAX) ? (M - i0) : MLMAX;

        /* Transpose + pad a's row strip into at_tile [K, MLMAX].
         * Rows 0..M_tile from a; rows M_tile..MLMAX-1 zero-pad. */
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < M_tile; r++) {
                at_tile[k * MLMAX + r] = a[(i0 + r) * K + k];
            }
            for (int r = M_tile; r < MLMAX; r++) {
                at_tile[k * MLMAX + r] = 0;
            }
        }

        for (int j0 = 0; j0 < N; j0 += MLMAX) {
            const int N_tile = (N - j0 < MLMAX) ? (N - j0) : MLMAX;

            /* Normalize + pad b strip into b_tile [K, MLMAX]. */
            if (transpose_b) {
                /* b is [N, K]; pull column j for j in [j0, j0+N_tile). */
                for (int k = 0; k < K; k++) {
                    for (int c = 0; c < N_tile; c++) {
                        b_tile[k * MLMAX + c] = b[(j0 + c) * K + k];
                    }
                    for (int c = N_tile; c < MLMAX; c++) {
                        b_tile[k * MLMAX + c] = 0;
                    }
                }
            } else {
                /* b is [K, N]; copy a column window contiguous in K. */
                for (int k = 0; k < K; k++) {
                    for (int c = 0; c < N_tile; c++) {
                        b_tile[k * MLMAX + c] = b[k * N + (j0 + c)];
                    }
                    for (int c = N_tile; c < MLMAX; c++) {
                        b_tile[k * MLMAX + c] = 0;
                    }
                }
            }

            /* Full mlmax × mlmax OPU MAC. */
            opu_tile_mac(c_tile, at_tile, b_tile,
                         (size_t)MLMAX, (size_t)K);

            /* Requantize the valid M_tile × N_tile sub-block. */
            for (int r = 0; r < M_tile; r++) {
                int8_t *out_row = &output[(i0 + r) * N + j0];
                int32_t *acc_row = &c_tile[r * MLMAX];
                for (int c = 0; c < N_tile; c++) {
                    int32_t v = (int32_t)roundf((float)acc_row[c] * total);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    out_row[c] = (int8_t)v;
                }
            }
        }
    }
}
