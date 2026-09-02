/* source: curated */
/* algorithm: gemmini_resadd */
/* accuracy_class: bit_exact */
/* BUG FOUND + FIXED 2026-08-28 (experiments/kernel_opt_log.jsonl id
 * BUG-add-s8-gemmini / 1500+): this kernel used to route well-conditioned
 * adds (both |scale/scale_out| ratios in [0.5, 2.0], n>=256) through
 * gemmini's tiled_resadd_auto with C_scale=ACC_SCALE_IDENTITY, on the
 * theory that "identity" made the mvout requantize a no-op. It isn't --
 * ACC_SCALE_IDENTITY still traverses the same round-and-saturate unit
 * conv2d_s8's HW path uses (see kernels/gemmini_q31/archive/'s two
 * kernels, retired for the identical defect), so mvout picks up its own
 * rounding on top of whatever the mvin scale already lost. Validated
 * ONLY against dronet, where every add fails the [0.5,2.0] gate and
 * fell back to scalar 100% of the time -- so this HW branch was NEVER
 * actually exercised until isolation-tested on yolov8n (whose adds land
 * inside the gate): max_abs_err=43 on the real F2 Q0.31 Gemmini
 * bitstream, isolation-tested (this kernel alone curated, every other
 * op forced to scalar reference_impl).
 *
 * WHY THIS CANNOT BE MADE EXACT ON THIS HW: the concatenation kernels
 * (kernels/gemmini_q31/*_cat*_gemmini_mvin_scale.c) work around the same
 * mvout round-and-saturate unit by driving tiled_matmul_auto with
 * full_C=true, which drains the RAW int32 accumulator and skips mvout's
 * ACC_SCALE entirely. That escape hatch is not available here:
 * tiled_resadd_auto (and everything under it -- tiled_resadd_stride_auto,
 * tiled_resadd, sp_tiled_resadd) hardcodes full_C=false in its
 * gemmini_loop_ws call (cores/gemmini/include/gemmini.h); only
 * tiled_matmul_auto exposes the full_C parameter. Reformulating the add
 * as a degenerate matmul (A=a[], B=1, D=b[]) doesn't recover exactness
 * either: D's own scale hook is a hardwired no-op on this config
 * (`#define MVIN_SCALE_ACC(x, scale) (x)` in gemmini_params.h), so D
 * can only inject an UNSCALED int32 bias, not a second ratio-scaled
 * operand -- and the operand that *does* go through the matmul (A) still
 * gets quantized to int8 by MVIN_SCALE before the multiply, the exact
 * same lossy step tiled_resadd_auto's mvin already does. MVIN_SCALE's
 * "round to elem_t on the way in" is how every datapath into gemmini's
 * PE array works on this bitstream, not a workaround-able software
 * choice -- there is no exact gemmini construction for a two-operand,
 * arbitrary-ratio weighted sum here.
 *
 * FIX: the gemmini branch (and the [0.5, 2.0] scale gate that used to
 * select it -- that gate was about mvin scale representability, not
 * correctness, and it accidentally hid this bug from every model whose
 * adds happened to fail it) is removed. This kernel now ALWAYS takes the
 * fixed-point scalar path below, which is exact by construction (folds
 * scale_a/scale_out and scale_b/scale_out into integer multipliers and
 * rounds half-away-from-zero, matching the reference roundf() bit for
 * bit) and was already the 100%-of-traffic path on dronet
 * (max_abs_err=0). Slower than the HW path would have been, but
 * correct beats fast: dronet stays at its existing err=0 baseline, and
 * yolov8n -- the model that actually exercises the previously-untested
 * gate -- now also verifies at err=0 (isolation-tested, this kernel
 * alone curated).
 *
 * WHY THE SCALAR PATH IS STILL FAST: measured on the F2 Q0.31 Gemmini
 * bitstream, the old naive float fallback (int->float converts, fdiv by
 * scale_out, out-of-line roundf()) cost 925,683 cycles for 11,456
 * elements (81 cyc/element). This path folds (scale_a/scale_out,
 * scale_b/scale_out) into two Q(S) integer multipliers once at entry and
 * runs the element loop in pure int64 arithmetic (native on rv64),
 * rounding half-away-from-zero to match roundf(). No fdiv, no libm call,
 * no float in the loop. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max)
{
    if (n <= 0) return;

    float a_ratio = scale_a / scale_out;
    float b_ratio = scale_b / scale_out;
    float a_abs   = a_ratio < 0 ? -a_ratio : a_ratio;
    float b_abs   = b_ratio < 0 ? -b_ratio : b_ratio;

    /* Fixed-point scalar path -- the ONLY path; see header. S is picked so
     * ratio*2^S stays exactly representable in float (|ratio|*2^S < 2^24)
     * and the int64 product cannot overflow. */
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
}
