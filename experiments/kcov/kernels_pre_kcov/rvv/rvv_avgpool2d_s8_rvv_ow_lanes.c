/* source: curated */
/* algorithm: rvv_ow_lanes */
/* accuracy_class: bit_exact */
/* origin: hand-written. avgpool2d_s8 with the output row in the vector
 *         lanes.
 *
 *   WHY THIS FILE EXISTS. avgpool2d_s8 carried no AlgorithmCandidate, so
 *   the curated probe had no (op, algorithm) pair to look for and the op
 *   ran the scalar reference inside builds labelled rvv_x60 -- 12.8% of
 *   vitfly_frontend, and 0.054 ms of vitfly_lstm's non-LSTM time.
 *
 *   STRUCTURE. One vector = a run of output columns of one (n, c, oh).
 *   Output is contiguous along ow, and input is contiguous along iw with
 *   stride SW, so the window accumulation is KH*KW strided loads into an
 *   i32 accumulator and the whole thing is one pass with no gather.
 *
 *   WHY THE DIVIDE IS A DIVIDE. The reference rounds the mean half away
 *   from zero, written as `(sum + div/2)/div` on the magnitude with the
 *   sign reapplied -- exact integer arithmetic, no float anywhere. So this
 *   is vdiv on the same magnitudes with the same reapplied sign: bit-exact
 *   by construction, and no rounding-mode question to answer at all. The
 *   divisor is loop-invariant but is NOT turned into a reciprocal
 *   multiply; vdiv is slower and correct.
 *
 *   THE PADDED CASE FALLS BACK, DELIBERATELY. With PH=PW=0 every window is
 *   fully in bounds, so `cnt` is KH*KW for every output and the divisor is
 *   the same constant whichever way count_include_pad is set. With padding
 *   it is neither -- cnt varies per output position and only the border
 *   differs -- so the vector path would need a per-lane divisor and a
 *   per-lane in-bounds mask to reproduce a case neither model here has.
 *   The reference loop runs verbatim instead. Getting count_include_pad
 *   backwards is silent (it shifts border values and nothing else), which
 *   is a good reason not to reimplement it twice.
 *
 *   THE AVL IS THE ELEMENT COUNT, NEVER A PREVIOUS vsetvl'S RESULT, AND
 *   THAT IS NOT STYLE. Written as `__riscv_vsetvl_e32m4(vl8)` -- chaining
 *   one vsetvl on another's return value -- GCC 14.3 substitutes an
 *   unrelated register for the AVL. Measured here: the second vsetvl
 *   was issued with the OUTER LOOP BOUND as its AVL, vl came out 5 where
 *   the row is 11 wide, the `vsetvli zero,zero` forms carried that 5 down
 *   to the store, and six of every eleven outputs were never written at all. Not a
 *   rounding difference: max_abs_err=68 against the reference, and silent.
 *
 *   It was correct under GCC 13.2 -- the compiler these kernels were
 *   verified on, and still what the default `CROSS` points at. 13.2 has its
 *   own bug (it reorders a vsetvl across a widening op and the binary
 *   SIGILLs), which is why 14.3 is mandatory. So these kernels moved from a
 *   compiler that crashes loudly to one that answers wrongly, and the only
 *   form that is correct under both is the element count, every time.
 *
 *   VTYPE. Integer only: e8m1 loads, an explicit e32m4 window for the
 *   extend, accumulate and divide, e16m2/e8m1 stepping back down for the
 *   store. Checked with scripts/check_rvv_vtype.py.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_avgpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int count_include_pad)
{
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;

    if (PH != 0 || PW != 0) {
        /* Reference loop, verbatim -- see the header. */
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t sum = 0;
                        int cnt = 0;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh*SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow*SW - PW + kw;
                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    sum += (int32_t)input[((n*C + c)*IH + ih)*IW + iw];
                                    cnt++;
                                }
                            }
                        }
                        int div = count_include_pad ? (KH*KW) : (cnt > 0 ? cnt : 1);
                        int32_t v;
                        if (sum >= 0) v = (sum + div/2) / div;
                        else          v = -(((-sum) + div/2) / div);
                        if (v < -128) v = -128;
                        if (v > 127) v = 127;
                        output[((n*C + c)*OH + oh)*OW + ow] = (int8_t)v;
                    }
                }
            }
        }
        return;
    }

    const int div = KH * KW;          /* every window is fully in bounds */
    const int half = div / 2;
    const ptrdiff_t stride = (ptrdiff_t)SW;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *plane = input + (size_t)(n*C + c) * IH * IW;
            int8_t *oplane = output + (size_t)(n*C + c) * OH * OW;
            for (int oh = 0; oh < OH; oh++) {
                int ow = 0;
                while (ow < OW) {
                    const size_t n_col = (size_t)(OW - ow);
                    size_t vl8 = __riscv_vsetvl_e8m1(n_col);
                    /* The accumulator lives in the 32-bit domain; name that
                     * transition explicitly rather than letting GCC carry
                     * e8m1 into a vsext.vf4 it cannot legally run under. */
                    size_t vl = __riscv_vsetvl_e32m4(n_col);
                    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
                    for (int kh = 0; kh < KH; kh++) {
                        const int8_t *row = plane + (size_t)(oh*SH + kh) * IW;
                        for (int kw = 0; kw < KW; kw++) {
                            const int8_t *src = row + (size_t)ow * SW + kw;
                            /* The vsetvl return values are bound rather than
                             * discarded: they are what tells GCC which width
                             * domain the next intrinsic sits in, and a
                             * dropped result is a dropped vtype change --
                             * which is a SIGILL on the board, not a build
                             * error. */
                            size_t l8 = __riscv_vsetvl_e8m1(n_col);
                            vint8m1_t v8 = (SW == 1)
                                ? __riscv_vle8_v_i8m1(src, l8)
                                : __riscv_vlse8_v_i8m1(src, stride, l8);
                            size_t l32 = __riscv_vsetvl_e32m4(n_col);
                            acc = __riscv_vadd_vv_i32m4(
                                acc, __riscv_vsext_vf4_i32m4(v8, l32), l32);
                        }
                    }
                    /* round(|sum|/div) half away from zero, sign reapplied:
                     * exactly the reference's integer expression. */
                    vbool8_t neg = __riscv_vmslt_vx_i32m4_b8(acc, 0, vl);
                    vint32m4_t mag = __riscv_vrsub_vx_i32m4(acc, 0, vl);
                    mag = __riscv_vmerge_vvm_i32m4(acc, mag, neg, vl);
                    mag = __riscv_vadd_vx_i32m4(mag, half, vl);
                    vint32m4_t q = __riscv_vdiv_vx_i32m4(mag, div, vl);
                    vint32m4_t qn = __riscv_vrsub_vx_i32m4(q, 0, vl);
                    q = __riscv_vmerge_vvm_i32m4(q, qn, neg, vl);
                    q = __riscv_vmax_vx_i32m4(q, -128, vl);
                    q = __riscv_vmin_vx_i32m4(q, 127, vl);

                    size_t vl16 = __riscv_vsetvl_e16m2(n_col);
                    vint16m2_t q16 = __riscv_vncvt_x_x_w_i16m2(q, vl16);
                    size_t vlo8 = __riscv_vsetvl_e8m1(n_col);
                    vint8m1_t q8 = __riscv_vncvt_x_x_w_i8m1(q16, vlo8);
                    __riscv_vse8_v_i8m1(oplane + (size_t)oh * OW + ow, q8, vlo8);

                    ow += (int)vl8;
                }
            }
        }
    }
}
