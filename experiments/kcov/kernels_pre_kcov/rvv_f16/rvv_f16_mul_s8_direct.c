/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV mul_s8.
 *
 * Same shape as the shipped kernels/rvv/rvv_add_s8_direct.c, one operation
 * along: both int8 operands widen i8 -> i16 (vsext.vf2) -> f32 (vfwcvt), the
 * product is one vfmul.vv, and the three per-tensor scales fold into a single
 * loop-invariant k = (scale_a*scale_b)/scale_out so the rescale is one
 * vfmul.vf rather than a per-lane divide.
 *
 * The requantize tail is vfncvt.x.f.w straight from f32 to i16 -- RVV float
 * -> int conversion rounds with the current mode and clips to the destination
 * range, so it saturates for free -- and the activation clamp then runs at
 * int16 width (8 lanes/cycle at DLEN=128 instead of 4 at int32). This is the
 * same tail rvv_add_s8_direct.c uses and it inherits that kernel's one
 * documented deviation from the scalar reference: vfncvt rounds ties to even
 * where roundf() rounds ties away from zero, so an exact .5 product lands one
 * LSB apart. That is inside the int8 verify envelope and is why the accuracy
 * class is left at the default rather than claimed bit_exact. */

void kernel_mul_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max) {
    const float k = (scale_a * scale_b) / scale_out;

    const int16_t lo = (int16_t)(activation_min < -128 ? -128
                                 : (activation_min > 127 ? 127
                                    : activation_min));
    const int16_t hi = (int16_t)(activation_max < -128 ? -128
                                 : (activation_max > 127 ? 127
                                    : activation_max));

    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e8m2(n - i);

        vint8m2_t va8 = __riscv_vle8_v_i8m2(a + i, vl);
        vint16m4_t va16 = __riscv_vsext_vf2_i16m4(va8, vl);
        vfloat32m8_t vfa = __riscv_vfwcvt_f_x_v_f32m8(va16, vl);

        vint8m2_t vb8 = __riscv_vle8_v_i8m2(b + i, vl);
        vint16m4_t vb16 = __riscv_vsext_vf2_i16m4(vb8, vl);
        vfloat32m8_t vfb = __riscv_vfwcvt_f_x_v_f32m8(vb16, vl);

        vfloat32m8_t vy = __riscv_vfmul_vv_f32m8(vfa, vfb, vl);
        vy = __riscv_vfmul_vf_f32m8(vy, k, vl);

        vint16m4_t o16 = __riscv_vfncvt_x_f_w_i16m4(vy, vl);
        o16 = __riscv_vmax_vx_i16m4(o16, lo, vl);
        o16 = __riscv_vmin_vx_i16m4(o16, hi, vl);
        __riscv_vse8_v_i8m2(output + i, __riscv_vncvt_x_x_w_i8m2(o16, vl), vl);
    }
}
