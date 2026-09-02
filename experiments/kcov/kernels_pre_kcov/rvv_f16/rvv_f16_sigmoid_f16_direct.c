/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: RVV sigmoid_f16 -- vectorized exp, no libm call.
 *
 * The scalar reference is one expf() per element. On ViNT that is 65 ops over
 * 645 K elements at 81.4 cycles each, and every one of those cycles is a
 * scalar libm call on a vector machine.
 *
 * The LUT trick that makes sigmoid_s8 fast does not transfer: int8 has 256
 * distinct inputs, fp16 has 65536, so the table can never amortize. Instead
 * this evaluates exp in the vector unit directly:
 *
 *   t   = clamp(-x, -25, 25)          domain guard, see below
 *   n   = rint(t * log2(e))           integer octave
 *   f   = t - n*ln2_hi - n*ln2_lo     |f| <= ln2/2 = 0.3466, 2-part ln2 so
 *                                     the reduction stays exact in f32
 *   e^f = degree-6 Taylor (Horner)    truncation error < f^7/5040 = 1.2e-7
 *   e^t = e^f * 2^n                   2^n built by injecting n+127 into the
 *                                     f32 exponent field, which is why the
 *                                     clamp matters: |n| <= 37 keeps that
 *                                     normal
 *   out = 1 / (1 + e^t)
 *
 * The clamp is not an approximation at fp16 output width: sigmoid(-25) is
 * 1.4e-11, far below the smallest fp16 subnormal, so everything past it
 * already rounds to 0.0 (and to 1.0 on the other side) in the reference too.
 *
 * Elements are loaded and stored as fp16 (vle16/vse16, eew=16); the exp is
 * evaluated at f32 because the REFERENCE is defined that way -- it computes
 * 1/(1+expf(-(float)v)) and rounds once on the fp16 store -- not because fp16
 * arithmetic is unavailable on this target. LMUL is 2 on the fp16 side (4 on
 * the f32 side) to keep the six live f32 temporaries of the Horner chain
 * inside the register file.
 *
 * accuracy_class is numeric_drift, honestly: a degree-6 minimax-free Taylor
 * plus a reciprocal is accurate to roughly 1e-7 relative in f32, which is
 * three orders of magnitude finer than the fp16 output ulp, so results agree
 * with the reference except where the exact value sits within 1e-7 of an fp16
 * rounding boundary. */

void kernel_sigmoid_f16(const _Float16 *input, _Float16 *output, int n) {
    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e16m2(n - i);

        vfloat16m2_t vh = __riscv_vle16_v_f16m2(input + i, vl);
        vfloat32m4_t vx = __riscv_vfwcvt_f_f_v_f32m4(vh, vl);

        /* t = clamp(-x, -25, 25) */
        vfloat32m4_t vt = __riscv_vfrsub_vf_f32m4(vx, 0.0f, vl);
        vt = __riscv_vfmin_vf_f32m4(vt, 25.0f, vl);
        vt = __riscv_vfmax_vf_f32m4(vt, -25.0f, vl);

        /* n = rint(t * log2e); f = t - n*ln2 (2-part) */
        vfloat32m4_t vnf = __riscv_vfmul_vf_f32m4(vt, 1.44269504088896341f, vl);
        vint32m4_t vni = __riscv_vfcvt_x_f_v_i32m4(vnf, vl);
        vnf = __riscv_vfcvt_f_x_v_f32m4(vni, vl);

        vfloat32m4_t vf = __riscv_vfnmsac_vf_f32m4(vt, 0.693359375f, vnf, vl);
        vf = __riscv_vfnmsac_vf_f32m4(vf, -2.12194440e-4f, vnf, vl);

        /* e^f, degree-6 Horner */
        vfloat32m4_t vp = __riscv_vfmul_vf_f32m4(vf, 1.0f / 720.0f, vl);
        vp = __riscv_vfadd_vf_f32m4(vp, 1.0f / 120.0f, vl);
        vp = __riscv_vfmul_vv_f32m4(vp, vf, vl);
        vp = __riscv_vfadd_vf_f32m4(vp, 1.0f / 24.0f, vl);
        vp = __riscv_vfmul_vv_f32m4(vp, vf, vl);
        vp = __riscv_vfadd_vf_f32m4(vp, 1.0f / 6.0f, vl);
        vp = __riscv_vfmul_vv_f32m4(vp, vf, vl);
        vp = __riscv_vfadd_vf_f32m4(vp, 0.5f, vl);
        vp = __riscv_vfmul_vv_f32m4(vp, vf, vl);
        vp = __riscv_vfadd_vf_f32m4(vp, 1.0f, vl);
        vp = __riscv_vfmul_vv_f32m4(vp, vf, vl);
        vp = __riscv_vfadd_vf_f32m4(vp, 1.0f, vl);

        /* scale by 2^n via exponent injection */
        vint32m4_t vpow = __riscv_vsll_vx_i32m4(
            __riscv_vadd_vx_i32m4(vni, 127, vl), 23, vl);
        vfloat32m4_t v2n = __riscv_vreinterpret_v_i32m4_f32m4(vpow);
        vfloat32m4_t ve = __riscv_vfmul_vv_f32m4(vp, v2n, vl);

        /* 1 / (1 + e^t) */
        vfloat32m4_t vd = __riscv_vfadd_vf_f32m4(ve, 1.0f, vl);
        vfloat32m4_t vs = __riscv_vfrdiv_vf_f32m4(vd, 1.0f, vl);

        __riscv_vse16_v_f16m2(output + i,
                              __riscv_vfncvt_f_f_w_f16m2(vs, vl), vl);
    }
}
