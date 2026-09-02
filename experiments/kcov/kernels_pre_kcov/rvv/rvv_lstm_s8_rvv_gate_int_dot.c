/* source: curated */
/* algorithm: rvv_gate_int_dot */
/* accuracy_class: bit_exact */
/* origin: hand-written. The gate reductions of lstm_s8, moved off the scalar
 *         double-precision FPU and onto the integer vector unit; the cell
 *         update itself is left exactly as the reference writes it.
 *
 *   WHY THIS FILE EXISTS. Curated kernels are looked up by EXACT op name
 *   (kernels/<backend>/<backend>_<op>_<algorithm>.c), and the RVV library
 *   had no lstm_s8 entry at all -- nor did the spec carry an RVV-affined
 *   AlgorithmCandidate for the probe to look for. So both of VitFly's LSTM
 *   layers resolved to the scalar reference inside a build labelled
 *   rvv_x60: 27.6 ms of a 28.2 ms run, 97.9%, which is why that model
 *   measured 1.04x against its own scalar build.
 *
 *   WHERE THE TIME GOES. Per timestep the op reduces
 *   4H*(input_size + H) products -- 1.67 M for VitFly's 660->395 layer.
 *   The reference computes each one as two int->double converts and two
 *   double multiplies, then accumulates on a serial fadd chain. That chain
 *   is the kernel: the FPU is never the width that matters, the dependency
 *   is.
 *
 *   WHAT CHANGES. The products are integers. x[k] and w[k] are int8, so
 *   sum(x[k]*w[k]) is exact in int32 (|x*w| <= 16384, and the longest
 *   reduction here is 660 terms -- eight orders of magnitude of headroom),
 *   and the two per-tensor scales factor straight out of the sum:
 *
 *       sum_k (x_k*s_in) * (w_k*s_w)  ==  (sum_k x_k*w_k) * (s_in*s_w)
 *
 *   as REALS. So the reduction becomes a widening int8 dot product --
 *   vwmul.vv into i16, vwadd.wv into an i32 accumulator, vredsum at the
 *   end -- and the float math shrinks to one multiply per gate.
 *
 *   BIT-EXACTNESS, STATED HONESTLY. This is not bit-exact by construction,
 *   and it is the one kernel here where that is true. The reference rounds
 *   every one of the 1.67 M products to double and then rounds again at
 *   every step of a serial sum; this computes the same real quantity with
 *   ONE rounding. The two differ by ~1e-13 relative -- the reference's own
 *   accumulated error, which this kernel does not have -- and agreement at
 *   the int8 output is therefore a measured property, not a proven one. It
 *   is measured: max_abs_err=0 over both VitFly layers and both lstm_tiny
 *   layers, in two data regimes, on the board
 *   (scripts/k1_verify_curated_rvv.py), and the end-to-end golden compare is
 *   unchanged. The margin is large -- for a disagreement, h/scale_h would
 *   have to land within ~1e-11 of a .5 tie -- but it is a margin, not an
 *   identity, so it is written down rather than claimed away.
 *
 *   Note the direction: the integer sum is EXACT. Where the two disagree,
 *   this kernel is the more accurate one.
 *
 *   WHAT DOES NOT CHANGE. sigmoid, tanh, the cell update, the requantize
 *   and the clamps are the reference's own expressions, in double, on the
 *   scalar unit, in the same order. They are ~5 libm calls per hidden unit
 *   against ~4200 MACs, so there is nothing to win there, and vectorising
 *   them would mean matching a rounding mode vfcvt does not have.
 *
 *   THE AVL IS THE ELEMENT COUNT, NEVER A PREVIOUS vsetvl'S RESULT, AND
 *   THAT IS NOT STYLE. Written as `__riscv_vsetvl_e32m4(vl8)` -- chaining
 *   one vsetvl on another's return value -- GCC 14.3 substitutes an
 *   unrelated register for the AVL. Measured in the avgpool kernel, which
 *   had the same shape: its second vsetvl was issued with the OUTER LOOP
 *   BOUND as its AVL, vl came out 5 where the row is 11
 *   wide, the `vsetvli zero,zero` forms carried that 5 down to the store,
 *   and six of every eleven outputs were never written at all. Not a
 *   rounding difference: max_abs_err=68 against the reference, and silent.
 *
 *   It was correct under GCC 13.2 -- the compiler these kernels were
 *   verified on, and still what the default `CROSS` points at. 13.2 has its
 *   own bug (it reorders a vsetvl across a widening op and the binary
 *   SIGILLs), which is why 14.3 is mandatory. So these kernels moved from a
 *   compiler that crashes loudly to one that answers wrongly, and the only
 *   form that is correct under both is the element count, every time.
 *
 *   VTYPE. The vector body is integer-only (i8m1 loads, i16m2 products,
 *   i32m4 accumulator) and every width transition is named by its own
 *   intrinsic. The reduction runs at VLMAX and the ragged tail is handled
 *   scalar, so no partial-vl write ever leaves an accumulator tail whose
 *   contents depend on the tail policy. Checked with
 *   scripts/check_rvv_vtype.py.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

/* Upper bound on hidden_size, so the h_prev snapshot is a fixed stack buffer
   rather than a VLA -- same bound and same reason as the reference. */
#ifndef MERLIN_LSTM_MAX_HIDDEN
#define MERLIN_LSTM_MAX_HIDDEN 1024
#endif

/* Exact int8 dot product.
 *
 * Two independent i32m4 accumulators so the vwadd.wv chain is not the
 * critical path, and VLMAX-sized bodies only.
 *
 * The ragged tail is the interesting part. Folding it into an accumulator
 * at a short vl would write only part of that register and leave the rest
 * at the mercy of the tail policy -- vta=1 permits the hardware to fill it,
 * and the vredsum that follows reads the whole register. So the tail
 * instead widens straight to i32 (vwcvt, no accumulator involved) and is
 * reduced AT ITS OWN vl into the same scalar the main accumulators reduce
 * into: vredsum reads exactly vl elements, so nothing above the tail is
 * ever looked at and no policy question arises.
 *
 * That is worth doing rather than dropping to a scalar remainder loop.
 * VitFly's recurrent reduction is 395 long -- 11 elements past the last
 * full 32-wide body, once per gate, 1580 times per timestep. Measured on
 * the board at n=395: 0.373 ms scalar remainder vs 0.308 ms this way, same
 * sums (integer arithmetic is order-independent, so "same" here is
 * identical, not close). */
static inline int32_t mb_lstm_dot_s8(const int8_t *a, const int8_t *b, int n)
{
    const size_t vlmax8 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax32 = __riscv_vsetvlmax_e32m4();
    vint32m4_t acc0 = __riscv_vmv_v_x_i32m4(0, vlmax32);
    vint32m4_t acc1 = __riscv_vmv_v_x_i32m4(0, vlmax32);
    const int step = (int)vlmax8;
    int k = 0;

    for (; k + 2 * step <= n; k += 2 * step) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)step);
        vint8m1_t va0 = __riscv_vle8_v_i8m1(a + k, vl);
        vint8m1_t vb0 = __riscv_vle8_v_i8m1(b + k, vl);
        vint8m1_t va1 = __riscv_vle8_v_i8m1(a + k + step, vl);
        vint8m1_t vb1 = __riscv_vle8_v_i8m1(b + k + step, vl);
        vint16m2_t p0 = __riscv_vwmul_vv_i16m2(va0, vb0, vl);
        vint16m2_t p1 = __riscv_vwmul_vv_i16m2(va1, vb1, vl);
        acc0 = __riscv_vwadd_wv_i32m4(acc0, p0, vl);
        acc1 = __riscv_vwadd_wv_i32m4(acc1, p1, vl);
    }
    for (; k + step <= n; k += step) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)step);
        vint8m1_t va0 = __riscv_vle8_v_i8m1(a + k, vl);
        vint8m1_t vb0 = __riscv_vle8_v_i8m1(b + k, vl);
        vint16m2_t p0 = __riscv_vwmul_vv_i16m2(va0, vb0, vl);
        acc0 = __riscv_vwadd_wv_i32m4(acc0, p0, vl);
    }

    acc0 = __riscv_vadd_vv_i32m4(acc0, acc1, vlmax32);
    vint32m1_t red = __riscv_vmv_s_x_i32m1(0, 1);
    red = __riscv_vredsum_vs_i32m4_i32m1(acc0, red, vlmax32);

    if (k < n) {
        const size_t n_tail = (size_t)(n - k);
        size_t vl = __riscv_vsetvl_e8m1(n_tail);
        vint8m1_t va0 = __riscv_vle8_v_i8m1(a + k, vl);
        vint8m1_t vb0 = __riscv_vle8_v_i8m1(b + k, vl);
        vint16m2_t p0 = __riscv_vwmul_vv_i16m2(va0, vb0, vl);
        /* Name the 32-bit domain before the widen, and reduce at the tail's
         * own vl -- vredsum reads exactly vl elements. */
        size_t vl32 = __riscv_vsetvl_e32m4(n_tail);
        red = __riscv_vredsum_vs_i32m4_i32m1(
            __riscv_vwcvt_x_x_v_i32m4(p0, vl32), red, vl32);
    }
    return __riscv_vmv_x_s_i32m1_i32(red);
}

void kernel_lstm_s8(const int8_t *input, const int8_t *w_ih,
                    const int8_t *w_hh, const int32_t *b_ih,
                    const int32_t *b_hh,
                    int8_t *h_state, int8_t *c_state, int8_t *output,
                    int input_size, int hidden_size,
                    float scale_in, float scale_w_ih, float scale_w_hh,
                    float scale_b, float scale_h, float scale_c,
                    int has_bias) {
    const int H = hidden_size;
    /* Snapshot h_prev before touching h_state -- every gate reduction reads
     * the WHOLE previous hidden vector while the loop below overwrites
     * h_state[t] as it goes. Dropping this makes the recurrence consume its
     * own output mid-step (the reference measured that at up to 12 LSB). */
    int8_t h_prev[MERLIN_LSTM_MAX_HIDDEN];
    if (H > MERLIN_LSTM_MAX_HIDDEN) {
        /* Refuse rather than silently truncate the state. */
        return;
    }
    for (int k = 0; k < H; k++) h_prev[k] = h_state[k];

    /* Both products are exact in double: each factor is a float (24-bit
     * significand), so the product needs at most 48 and double has 53. The
     * only rounding on this path is the one multiply by the integer sum. */
    const double s_x = (double)scale_in * (double)scale_w_ih;
    const double s_r = (double)scale_h * (double)scale_w_hh;

    for (int t = 0; t < H; t++) {
        double g_pre[4];
        for (int gi = 0; gi < 4; gi++) {
            const int j = gi * H + t;
            const int32_t sx =
                mb_lstm_dot_s8(input, w_ih + (size_t)j * (size_t)input_size,
                               input_size);
            const int32_t sr =
                mb_lstm_dot_s8(h_prev, w_hh + (size_t)j * (size_t)H, H);
            double acc = (double)sx * s_x + (double)sr * s_r;
            if (has_bias) {
                acc += (double)b_ih[j] * (double)scale_b;
                acc += (double)b_hh[j] * (double)scale_b;
            }
            g_pre[gi] = acc;
        }
        /* double, not float, and written as the reference writes it: the
           device and the golden simulator only agree bit-for-bit if both
           land well inside half an int8 LSB, and float32 did not (2 LSB). */
        const double i_g = 1.0 / (1.0 + exp(-g_pre[0]));
        const double f_g = 1.0 / (1.0 + exp(-g_pre[1]));
        const double g_g = tanh(g_pre[2]);
        const double o_g = 1.0 / (1.0 + exp(-g_pre[3]));

        const double c_prev = (double)c_state[t] * (double)scale_c;
        const double c_new = f_g * c_prev + i_g * g_g;
        const double h_new = o_g * tanh(c_new);

        int32_t cq = (int32_t)round(c_new / (double)scale_c);
        if (cq < -128) cq = -128;
        if (cq > 127) cq = 127;
        int32_t hq = (int32_t)round(h_new / (double)scale_h);
        if (hq < -128) hq = -128;
        if (hq > 127) hq = 127;
        c_state[t] = (int8_t)cq;
        h_state[t] = (int8_t)hq;
        output[t] = (int8_t)hq;
    }
}
