/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV cat2_c1_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_cat2_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    float ratio0 = scale0 / scale_out;
    float ratio1 = scale1 / scale_out;

    const int8_t *ins[2] = { in0, in1 };
    int cs[2] = { c0, c1 };
    float ratios[2] = { ratio0, ratio1 };
    int8_t amin = (int8_t)activation_min;
    int8_t amax = (int8_t)activation_max;

    for (int n = 0; n < N; n++) {
        int out_c = 0;
        for (int i = 0; i < 2; i++) {
            float ratio = ratios[i];
            int ci = cs[i];
            const int8_t *in = ins[i];
            for (int c = 0; c < ci; c++) {
                const int8_t *src = in + ((n * ci) + c) * stride;
                int8_t *dst = output + ((n * (c0 + c1) + out_c + c) * stride);
                int hw = 0;
                size_t vl;
                for (; hw < stride; hw += (int)vl) {
                    vl = __riscv_vsetvl_e8m2(stride - hw);
                    /* Load int8 */
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src + hw, vl);
                    /* Sign-extend i8 -> i16 */
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                    /* Sign-extend i16 -> i32 */
                    vint32m8_t v32 = __riscv_vsext_vf2_i32m8(v16, vl);
                    /* Convert int32 -> float32 */
                    vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(v32, vl);
                    /* Multiply by ratio */
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl);
                    /* Round to nearest int32 (round-to-nearest-even via vfcvt) */
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                    /* Narrow i32 -> i16 with saturation */
                    vint16m4_t vi16 = __riscv_vnclip_wx_i16m4(vi, 0, __RISCV_VXRM_RDN, vl);
                    /* Narrow i16 -> i8 with saturation */
                    vint8m2_t vi8 = __riscv_vnclip_wx_i8m2(vi16, 0, __RISCV_VXRM_RDN, vl);
                    /* Clamp to activation range */
                    vi8 = __riscv_vmax_vx_i8m2(vi8, amin, vl);
                    vi8 = __riscv_vmin_vx_i8m2(vi8, amax, vl);
                    /* Store */
                    __riscv_vse8_v_i8m2(dst + hw, vi8, vl);
                }
            }
            out_c += ci;
        }
    }
}