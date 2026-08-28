/* source: curated */
/* algorithm: outerprod */
/* origin: Saturn OPU i8 outer-product MAC, adapted for the modelblaster-flow
 *   linear_s8 signature. Ported from
 *     hw/chipyard/generators/saturn/benchmarks/opu-gemm/kernel.h
 *   (branch origin/opu-fp8), function `i8_mm_bme_sq` + the bias-aware
 *   `i8_loop_k_general` variant (Miles Rusch).
 *
 * Differences from the rvv_opu_matmul_s8_outerprod.c port:
 *   1. STRIDED loads (vlse8.v) for both `input` and `weight` instead
 *      of pre-transposing into stack scratch. This keeps the kernel
 *      scratch-free (works for arbitrary K, including dronet's
 *      K=2048 FC) at the cost of slower per-vector loads on Saturn
 *      V256+ (~2× vs unit-stride). For OPU shapes where K is the
 *      reduction dim and M, N are small, the strided-load overhead
 *      is amortized over the K iterations.
 *   2. Bias-broadcast seed: OPMVINBCAST loads the i32 bias vector
 *      [N] into all M rows of m1, replacing the upstream pattern of
 *      manually seeding c_out + later add.
 *   3. Q0.31 requantize tail (output_multiplier, output_shift,
 *      output_offset, activation_min/max) on the i32 drain, matching
 *      the CMSIS-NN / muRISCV-NN linear_s8 reference exactly.
 *   4. Symmetric quant only (input_offset == filter_offset == 0):
 *      the asymmetric path needs sum(x), sum(w) precomputation
 *      which adds complexity; modelblaster-flow extract emits symmetric
 *      quant today so falling back is fine until that changes.
 *
 * The OPU MAC body (the inner VOPACC loop) is unchanged from upstream.
 */

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
#include "saturn_opu.h"

/* Q0.31 fixed-point requantize — matches the linear_s8 reference impl in
 * modelblaster/pipeline/reference_kernels.py::LINEAR_S8 (CMSIS-NN canonical
 * round-multiply-and-shift). */
static inline int32_t q31_requantize(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return ((int32_t)prod + round) >> shift;
    }
    return (int32_t)prod << -shift;
}

static void linear_s8_scalar_fallback(const int8_t *input, const int8_t *weight,
                                      const int32_t *bias, int8_t *output,
                                      int M, int K, int N,
                                      int input_offset, int filter_offset,
                                      int output_offset,
                                      int output_multiplier, int output_shift,
                                      int activation_min, int activation_max) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;
            for (int k = 0; k < K; k++) {
                int32_t iv = (int32_t)input[m * K + k]  + input_offset;
                int32_t wv = (int32_t)weight[n * K + k] + filter_offset;
                acc += iv * wv;
            }
            int32_t v = q31_requantize(acc, output_multiplier, output_shift);
            v += output_offset;
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[m * N + n] = (int8_t)v;
        }
    }
}

void kernel_linear_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int M, int K, int N,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    /* OPU eligibility check. Same constraints as the matmul curation:
     * M, N must fit in one mlmax (= VLEN/8) tile. Asymmetric quant is
     * deferred — the simple OPU MAC computes raw sum, and the
     * offset-correction cross-terms (sum(x), sum(w)) would need a
     * separate vector reduction pass. */
    size_t mlmax;
    asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(mlmax));
    if ((size_t)M > mlmax || (size_t)N > mlmax ||
        input_offset != 0 || filter_offset != 0) {
        linear_s8_scalar_fallback(input, weight, bias, output, M, K, N,
                                  input_offset, filter_offset, output_offset,
                                  output_multiplier, output_shift,
                                  activation_min, activation_max);
        return;
    }

    /* Seed m1 with bias broadcast (or zero if bias == NULL). Vector
     * width is N i32 lanes; OPMVINBCAST replicates to all rows. */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)N));
    if (bias) {
        asm volatile("vle32.v v0, (%0)" : : "r"(bias));
    } else {
        asm volatile("vmv.v.i v0, 0");
    }
    OPMVINBCAST(m1, v0);

    /* Per-K MAC loop with strided loads:
     *   - input[m, k] for m in [0, M): base=&input[k], stride=K bytes.
     *   - weight[n, k] for n in [0, N): base=&weight[k], stride=K bytes.
     * (input is [M, K] row-major; element [m, k] sits at offset m*K+k.
     * Reading M consecutive m values for fixed k means base=&input[k]
     * stepping by K bytes per lane.) */
    const ptrdiff_t in_stride = (ptrdiff_t)K * sizeof(int8_t);
    const ptrdiff_t w_stride  = (ptrdiff_t)K * sizeof(int8_t);
    for (int k = 0; k < K; k++) {
        /* Load M lanes of input column k. */
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)M));
        asm volatile("vlse8.v v16, (%0), %1" :
                     : "r"(&input[k]),  "r"((unsigned long)in_stride));
        /* Load N lanes of weight column k. */
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)N));
        asm volatile("vlse8.v v18, (%0), %1" :
                     : "r"(&weight[k]), "r"((unsigned long)w_stride));
        /* m1[r,c] += input[r,k] * weight[c,k] */
        VOPACC(m1, v18, v16);
    }

    /* Drain m1 rows to an i32 stack buffer, then requantize per element. */
    int32_t row_i32[64];  /* mlmax <= 64 for V512 */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)N));
    for (int m = 0; m < M; m++) {
        VMV_VR(v0, m, m1);
        asm volatile("vse32.v v0, (%0)" : : "r"(row_i32));
        for (int n = 0; n < N; n++) {
            int32_t v = q31_requantize(row_i32[n],
                                       output_multiplier, output_shift);
            v += output_offset;
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[m * N + n] = (int8_t)v;
        }
    }
}
