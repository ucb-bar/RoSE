/* source: curated */
/* algorithm: gemmini_resadd */
/* accuracy_class: numeric_drift */
/* origin: gemmini tiled_resadd where the mvin scale ratios are in range,
 *         fixed-point scalar otherwise.
 *
 * WHY THE SCALAR PATH MATTERS: the gemmini path is gated on both operand
 * scale ratios landing in [0.5, 2.0] (outside that the int8 mvin scale
 * loses too much).  Every add in dronet fails that gate --
 *   add   : 0.066793/0.339678 = 0.197
 *   add_1 : 0.163921/0.360274 = 0.455
 *   add_2 : 0.230922/0.707875 = 0.326
 * -- so 100% of dronet's add_s8 work runs the fallback.  Measured on the
 * F2 Q0.31 Gemmini bitstream the old float fallback cost 925,683 cycles
 * for 11,456 elements (81 cyc/element): per element it did two int->float
 * converts, an fdiv by scale_out, and an out-of-line roundf() call.
 *
 * The fallback here folds (scale_a/scale_out, scale_b/scale_out) into two
 * Q(S) integer multipliers once at entry and runs the element loop in
 * pure int64 arithmetic (native on rv64), rounding half-away-from-zero to
 * match roundf().  No fdiv, no libm call, no float in the loop. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max)
{
    float a_ratio = scale_a / scale_out;
    float b_ratio = scale_b / scale_out;
    float a_abs   = a_ratio < 0 ? -a_ratio : a_ratio;
    float b_abs   = b_ratio < 0 ? -b_ratio : b_ratio;
    bool scales_ok = (a_abs >= 0.5f && a_abs <= 2.0f
                      && b_abs >= 0.5f && b_abs <= 2.0f);

    if (n <= 0) return;

    if (n < 256 || !scales_ok) {
        /* Fixed-point scalar path. S is picked so ratio*2^S stays exactly
         * representable in float (|ratio|*2^S < 2^24) and the int64 product
         * cannot overflow. */
        int S = 24;
        float mx = a_abs > b_abs ? a_abs : b_abs;
        while (S > 0 && mx * (float)((uint32_t)1 << S) >= 8388608.0f) S--;

        int64_t ma = (int64_t)lrintf(a_ratio * (float)((uint32_t)1 << S));
        int64_t mb = (int64_t)lrintf(b_ratio * (float)((uint32_t)1 << S));
        int64_t rnd = (int64_t)1 << (S - 1);

        /* Both operands are int8, so there are only 256 possible products
         * per side. Tabulate them once (2 x 1 KB, L1-resident) and the
         * element loop becomes two table loads, an add, a shift and a
         * clamp -- no multiplier in the loop at all. On a small in-order
         * Rocket the 64-bit multiplier is the loop's critical resource;
         * the tables trade it for L1 hits. */
        static int64_t ta[256], tb[256];
        for (int v = 0; v < 256; v++) {
            int sv = v - 128;                 /* ta/tb are indexed by v+128 */
            ta[v] = (int64_t)sv * ma;
            tb[v] = (int64_t)sv * mb;
        }

        for (int i = 0; i < n; i++) {
            int64_t acc = ta[(int)a[i] + 128] + tb[(int)b[i] + 128];
            int32_t v = (acc >= 0)
                      ? (int32_t)((acc + rnd) >> S)
                      : -(int32_t)(((-acc) + rnd) >> S);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i] = (int8_t)v;
        }
        return;
    }

    bool fused_relu = (activation_min == 0 && activation_max == 127);
    bool need_post_clamp = !(activation_min == -128 && activation_max == 127)
                            && !fused_relu;

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    scale_t a_scale = (scale_t)a_ratio;
    scale_t b_scale = (scale_t)b_ratio;

    enum { ADD_CHUNK_MAX = 6272 };
    int remaining = n;
    int offset = 0;
    while (remaining > 0) {
        int chunk = remaining > ADD_CHUNK_MAX ? ADD_CHUNK_MAX : remaining;
        gemmini_flush(0);
        asm volatile("fence" ::: "memory");
        tiled_resadd_auto(
            /* I = */ 1, /* J = */ (size_t)chunk,
            a_scale, b_scale, ACC_SCALE_IDENTITY,
            a + offset, b + offset, output + offset,
            /* relu = */ fused_relu,
            WS
        );
        gemmini_fence();
        gemmini_flush(0);
        offset    += chunk;
        remaining -= chunk;
    }

    if (need_post_clamp) {
        for (int i = 0; i < n; i++) {
            int v = output[i];
            if (v < activation_min) output[i] = (int8_t)activation_min;
            else if (v > activation_max) output[i] = (int8_t)activation_max;
        }
    }
}
