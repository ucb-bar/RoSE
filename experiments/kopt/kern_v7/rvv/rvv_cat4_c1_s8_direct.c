/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV cat4_c1_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_cat4_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2, const int8_t *in3, int c3, float scale3,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    int total_c = c0 + c1 + c2 + c3;

    const int8_t *ins[4] = { in0, in1, in2, in3 };
    int cs[4] = { c0, c1, c2, c3 };
    float scales[4] = { scale0, scale1, scale2, scale3 };

    for (int n = 0; n < N; n++) {
        int out_c = 0;
        for (int i = 0; i < 4; i++) {
            float ratio = scales[i] / scale_out;
            int ci = cs[i];
            const int8_t *in_base = ins[i] + n * ci * stride;
            int8_t *out_base = output + (n * total_c + out_c) * stride;

            /* Check if output channels are contiguous (they are: out_c..out_c+ci-1 are consecutive) */
            /* We can process all ci channels * stride pixels as a flat array */
            int total_elems = ci * stride;
            int hw = 0;
            size_t vl;
            for (; hw < total_elems; hw += vl) {
                vl = __riscv_vsetvl_e8m2(total_elems - hw);
                vint8m2_t v8 = __riscv_vle8_v_i8m2(in_base + hw, vl);
                /* sign-extend i8 -> i32 (4x widen) */
                vint32m8_t v32 = __riscv_vsext_vf4_i32m8(v8, vl);
                /* convert to float */
                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(v32, vl);
                /* multiply by ratio */
                vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl);
                /* round to nearest int32 */
                vint32m8_t vr = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                /* clamp */
                vr = __riscv_vmax_vx_i32m8(vr, activation_min, vl);
                vr = __riscv_vmin_vx_i32m8(vr, activation_max, vl);
                /* narrow i32 -> i16 -> i8 */
                vint16m4_t vr16 = __riscv_vncvt_x_x_w_i16m4(vr, vl);
                vint8m2_t vr8 = __riscv_vncvt_x_x_w_i8m2(vr16, vl);
                __riscv_vse8_v_i8m2(out_base + hw, vr8, vl);
            }

            out_c += ci;
        }
    }
}