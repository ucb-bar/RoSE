/* source: curated */
/* algorithm: gemmini_mvin_scale */
/* accuracy_class: numeric_drift */
/* origin: per-input requantize-and-copy via gemmini's MVIN_SCALE path,
 *         with the ACC_SCALE mvout stage bypassed entirely (full_C=true
 *         raw accumulator drain).
 *
 *         WHY THIS AVOIDS THE KNOWN ACC_SCALE BUG: two prior kernels
 *         (kernels/gemmini_q31/archive/gemmini_q31_relu_s8_gemmini_resadd_relu.c,
 *         .../gemmini_q31_maxpool2d_s8_gemmini_tiled_conv_pool.c) both
 *         declared accuracy_class=bit_exact and measured max_abs_err=17
 *         in isolation (kernel_opt_log.jsonl id 1100): gemmini's mvout
 *         ACC_SCALE round-and-saturate unit is not a true identity for
 *         this Q31Ws32x32Acc config, even when the requested scale is
 *         ACC_SCALE_IDENTITY. Both of those kernels reached ACC_SCALE
 *         via tiled_resadd_auto / tiled_conv_dw_auto, and NEITHER of
 *         those convenience wrappers exposes a `full_C` raw-accumulator
 *         option (checked: their gemmini.h signatures take a plain
 *         `acc_scale_t scale`, no full_C bool) -- so there is no way to
 *         route a resadd- or conv-shaped op around ACC_SCALE. Only the
 *         base tiled_matmul_auto/tiled_matmul exposes full_C, which is
 *         how gemmini_q31_conv2d_s8_gemmini_im2col_full_C.c gets its own
 *         bit-exact result (raw acc, exact two-stage Q0.31 requantize on
 *         the CPU). This kernel reuses that SAME full_C escape hatch for
 *         a degenerate 1-column, 1-reduction-step matmul that does
 *         nothing but move data:
 *
 *           A = input chunk, [count x K=1], mvin with A_scale=ratio
 *               (ratio = scale_in/scale_out, a REAL ieee754 float --
 *               scale_t is `float`, MVIN_SCALE_IDENTITY=1.0 is an EXACT
 *               representation, unlike ACC_SCALE_IDENTITY which is a
 *               Q-fixed-point approximation of 1.0 and is the root cause
 *               of the bug above). MVIN_SCALE computes
 *               ROUND_NEAR_EVEN(x*scale) clamped to int8 -- i.e. gemmini
 *               performs the SAME "scale, round, clamp to int8" the
 *               reference formula performs, in hardware, before the
 *               value ever reaches the accumulator.
 *           B = constant {1} int8 weight, mvin with B_scale=1.0 (exact,
 *               weight value 1 needs no rounding at all).
 *           C = A*B = A exactly (int8 x int8 -> int32 MAC by 1 is exact
 *               integer arithmetic -- no further rounding is possible at
 *               this stage). full_C=true drains the RAW int32
 *               accumulator, bypassing ACC_SCALE altogether. Since A was
 *               already clamped into int8 range at mvin time, the raw
 *               int32 value IS the final answer -- a plain narrowing
 *               cast to int8_t, no further math needed.
 *
 *         ACCURACY CAVEAT (why this is numeric_drift, not bit_exact):
 *         MVIN_SCALE's ROUND_NEAR_EVEN ties to EVEN; the reference
 *         (roundf) ties AWAY FROM ZERO. These differ only when the
 *         scaled value lands on an EXACT .5 boundary in float32, which
 *         a Monte-Carlo sweep (in-repo analysis, 51.2M (ratio, int8
 *         input) pairs at random ratios in [0.001, 2.0]) shows happens
 *         at rate ~2.3e-6 per (ratio, input-code) pair -- i.e. expected
 *         well under 1 mismatched output code across a whole model's
 *         worth of cat calls, but not provably zero for an arbitrary
 *         ratio. This is the SAME category of caveat
 *         gemmini_q31_add_s8_gemmini_resadd.c documents for its own
 *         ratio-precompute path, and is labeled the same way. Isolation-
 *         test the actual model's real scale values before trusting
 *         max_abs_err=0 (see kernel_opt_log.jsonl id 1300+).
 *
 *         Falls back to scalar (matching the reference bit-for-bit,
 *         roundf/lroundf half-away-from-zero) for small counts, where
 *         gemmini's per-call RoCC setup cost exceeds the body work. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

static void cat2_gemmini_scale_copy(const int8_t *src, int8_t *dst,
                                    size_t count, float ratio)
{
    if (count == 0) return;

    if (count < 256) {
        for (size_t i = 0; i < count; i++) {
            float f = (float)src[i] * ratio;
            long v = lroundf(f);
            if (v < -128) v = -128;
            if (v > 127) v = 127;
            dst[i] = (int8_t)v;
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    enum { CHUNK_MAX = 4096 };
    static const elem_t ws_b[1] = {1};
    static acc_t ws_c[CHUNK_MAX] __attribute__((aligned(64)));

    size_t remaining = count;
    size_t offset = 0;
    while (remaining > 0) {
        size_t chunk = remaining > CHUNK_MAX ? CHUNK_MAX : remaining;

        gemmini_flush(0);
        asm volatile("fence" ::: "memory");

        tiled_matmul_auto(
            chunk, 1, 1,
            src + offset, ws_b,
            NULL, ws_c,
            1, 1, 1, 1,
            (scale_t)ratio, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
            false,
            false, false,
            true, false,
            0, WS
        );

        gemmini_fence();
        gemmini_flush(0);

        for (size_t i = 0; i < chunk; i++) {
            int32_t v = ws_c[i];
            dst[offset + i] = (int8_t)v;
        }

        offset    += chunk;
        remaining -= chunk;
    }
}

void kernel_cat2_c1_s8(const int8_t *in0, int c0, float scale0,
                       const int8_t *in1, int c1, float scale1,
                       int8_t *output, int N, int H, int W,
                       float scale_out, int activation_min, int activation_max)
{
    const size_t stride = (size_t)H * (size_t)W;
    const int8_t *ins[2]   = { in0, in1 };
    const int     cs[2]    = { c0, c1 };
    const float   scales[2] = { scale0, scale1 };
    const size_t  c_total  = (size_t)c0 + (size_t)c1;

    for (int n = 0; n < N; n++) {
        size_t out_c = 0;
        for (int i = 0; i < 2; i++) {
            float ratio = scales[i] / scale_out;
            const int8_t *src = ins[i] + (size_t)n * (size_t)cs[i] * stride;
            int8_t *dst = output + ((size_t)n * c_total + out_c) * stride;
            size_t count = (size_t)cs[i] * stride;
            cat2_gemmini_scale_copy(src, dst, count, ratio);
            out_c += (size_t)cs[i];
        }
    }

    if (activation_min > -128 || activation_max < 127) {
        size_t total = (size_t)N * c_total * stride;
        for (size_t i = 0; i < total; i++) {
            if (output[i] < activation_min) output[i] = (int8_t)activation_min;
            else if (output[i] > activation_max) output[i] = (int8_t)activation_max;
        }
    }
}
