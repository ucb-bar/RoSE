/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV mul_c1_s8 (per-channel gate x tensor).
 *
 * The gate is one int8 per channel, so for a fixed (n, c) the whole HW plane
 * is scaled by the single constant kc = gate[c]*scale_gate*scale_x/scale_out.
 * Hoisting that division out of the HW loop turns the reference's per-element
 * multiply-divide-round into one vfmul.vf plus the standard
 * vfncvt.x.f.w + int16 clamp tail (see rvv_f16_mul_s8_direct.c for why that
 * tail is used).
 *
 * Deviation from the scalar reference, stated rather than hidden: the
 * reference forms g_real*(x*scale_x) and then divides by scale_out per
 * element, and rounds half away from zero; this kernel folds the same three
 * constants into one f32 scalar and rounds half to even. Both differences are
 * sub-LSB and are covered by the int8 verify envelope. */

void kernel_mul_c1_s8(const int8_t *gate, const int8_t *x, int8_t *output,
                      int N, int C, int HW,
                      float scale_gate, float scale_x, float scale_out) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float kc = ((float)gate[c] * scale_gate) * scale_x
                             / scale_out;
            const size_t base = (size_t)(n * C + c) * HW;
            const int8_t *xp = x + base;
            int8_t *op = output + base;

            int i = 0;
            size_t vl;
            for (; i < HW; i += (int)vl) {
                vl = __riscv_vsetvl_e8m2(HW - i);
                vint8m2_t v8 = __riscv_vle8_v_i8m2(xp + i, vl);
                vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                vfloat32m8_t vy = __riscv_vfwcvt_f_x_v_f32m8(v16, vl);
                vy = __riscv_vfmul_vf_f32m8(vy, kc, vl);
                vint16m4_t o16 = __riscv_vfncvt_x_f_w_i16m4(vy, vl);
                o16 = __riscv_vmax_vx_i16m4(o16, (int16_t)-128, vl);
                o16 = __riscv_vmin_vx_i16m4(o16, (int16_t)127, vl);
                __riscv_vse8_v_i8m2(op + i,
                                    __riscv_vncvt_x_x_w_i8m2(o16, vl), vl);
            }
        }
    }
}
