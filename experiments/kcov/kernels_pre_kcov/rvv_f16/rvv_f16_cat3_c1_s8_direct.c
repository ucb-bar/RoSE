/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV cat3_c1_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_cat3_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    int c_total = c0 + c1 + c2;
    float ratio0 = scale0 / scale_out;
    float ratio1 = scale1 / scale_out;
    float ratio2 = scale2 / scale_out;

    for (int n = 0; n < N; n++) {
        /* Process input 0 */
        {
            const int8_t *in_base = in0 + n * c0 * stride;
            int8_t *out_base = output + n * c_total * stride;
            for (int c = 0; c < c0; c++) {
                const int8_t *src = in_base + c * stride;
                int8_t *dst = out_base + c * stride;
                int hw = 0;
                size_t vl;
                for (; hw < stride; hw += vl) {
                    vl = __riscv_vsetvl_e8m2(stride - hw);
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src + hw, vl);
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                    vfloat32m8_t vf = __riscv_vfwcvt_f_x_v_f32m8(v16, vl);
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio0, vl);
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                    vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl);
                    vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl);
                    vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl);
                    vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
                    __riscv_vse8_v_i8m2(dst + hw, vi8, vl);
                }
            }
        }
        /* Process input 1 */
        {
            const int8_t *in_base = in1 + n * c1 * stride;
            int8_t *out_base = output + (n * c_total + c0) * stride;
            for (int c = 0; c < c1; c++) {
                const int8_t *src = in_base + c * stride;
                int8_t *dst = out_base + c * stride;
                int hw = 0;
                size_t vl;
                for (; hw < stride; hw += vl) {
                    vl = __riscv_vsetvl_e8m2(stride - hw);
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src + hw, vl);
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                    vfloat32m8_t vf = __riscv_vfwcvt_f_x_v_f32m8(v16, vl);
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio1, vl);
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                    vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl);
                    vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl);
                    vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl);
                    vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
                    __riscv_vse8_v_i8m2(dst + hw, vi8, vl);
                }
            }
        }
        /* Process input 2 */
        {
            const int8_t *in_base = in2 + n * c2 * stride;
            int8_t *out_base = output + (n * c_total + c0 + c1) * stride;
            for (int c = 0; c < c2; c++) {
                const int8_t *src = in_base + c * stride;
                int8_t *dst = out_base + c * stride;
                int hw = 0;
                size_t vl;
                for (; hw < stride; hw += vl) {
                    vl = __riscv_vsetvl_e8m2(stride - hw);
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src + hw, vl);
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                    vfloat32m8_t vf = __riscv_vfwcvt_f_x_v_f32m8(v16, vl);
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio2, vl);
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                    vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl);
                    vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl);
                    vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl);
                    vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vl);
                    __riscv_vse8_v_i8m2(dst + hw, vi8, vl);
                }
            }
        }
    }
}