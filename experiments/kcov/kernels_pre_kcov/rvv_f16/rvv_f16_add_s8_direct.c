/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV add_s8.
 *
 * Both operands widen i8 -> i16 -> f32 (vsext.vf2 + vfwcvt, so only the
 * second step runs at 32-bit datapath width), the two per-tensor rescales
 * fold into loop-invariant ka = scale_a/scale_out and kb = scale_b/scale_out
 * so the arithmetic is one vfmul plus one vfmacc, and the requantize tail is
 * a single vfncvt.x.f.w: RVV float -> int conversions round with the current
 * mode and clip to the destination range, so f32 -> i16 saturating narrowing
 * costs one instruction instead of a vfcvt + a narrow, and needs no vnclip
 * (whose vxrm CSR write costs vector-unit sync on this core). The activation
 * clamp then runs in the int16 domain -- 8 lanes/cycle at DLEN=128 instead of
 * 4 for int32. */

void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max) {
    /* Real divisions, computed once, so each constant carries the same
     * rounding the scalar reference's `/ scale_out` would. */
    const float ka = scale_a / scale_out;
    const float kb = scale_b / scale_out;

    const int16_t lo = (int16_t)(activation_min < -128 ? -128
                                 : (activation_min > 127 ? 127
                                    : activation_min));
    const int16_t hi = (int16_t)(activation_max < -128 ? -128
                                 : (activation_max > 127 ? 127
                                    : activation_max));

    int i = 0;
    size_t vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m2(n - i);

        vint8m2_t va8 = __riscv_vle8_v_i8m2(a + i, vl);
        vint16m4_t va16 = __riscv_vsext_vf2_i16m4(va8, vl);
        vfloat32m8_t vy = __riscv_vfwcvt_f_x_v_f32m8(va16, vl);
        vy = __riscv_vfmul_vf_f32m8(vy, ka, vl);

        vint8m2_t vb8 = __riscv_vle8_v_i8m2(b + i, vl);
        vint16m4_t vb16 = __riscv_vsext_vf2_i16m4(vb8, vl);
        vfloat32m8_t vfb = __riscv_vfwcvt_f_x_v_f32m8(vb16, vl);
        vy = __riscv_vfmacc_vf_f32m8(vy, kb, vfb, vl);

        vint16m4_t o16 = __riscv_vfncvt_x_f_w_i16m4(vy, vl);
        o16 = __riscv_vmax_vx_i16m4(o16, lo, vl);
        o16 = __riscv_vmin_vx_i16m4(o16, hi, vl);

        vint8m2_t o8 = __riscv_vncvt_x_x_w_i8m2(o16, vl);
        __riscv_vse8_v_i8m2(output + i, o8, vl);
    }
}
