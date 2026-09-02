/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV batchnorm2d_f16 -- fp16 loads and stores, fp32 accumulate, via
 * the widening MACC (vfwmacc.vf, f16 x f16 -> f32).
 *
 * This is the ONE fp16 op in this family where widening is not an
 * implementation preference but the definition of the operator. The scalar
 * reference deliberately computes s*x + b in fp32 and rounds once on the
 * store, with a comment recording that a pure-fp16 multiply-add accumulated
 * enough per-channel error to produce 30-50% magnitude drift through the
 * EfficientNet body. A native fp16 vfmacc would round the product before the
 * add and reintroduce exactly that; vfwmacc.vf keeps the product exact and
 * rounds once, which is bit-identical to the reference.
 *
 * The data still moves as fp16 -- vle16/vse16 at eew=16 -- so the widening
 * costs nothing in bandwidth; it is the same idiom the shipped
 * rvv_f16_conv2d_f16_oc_blocked.c conv kernel uses. */

void kernel_batchnorm2d_f16(const _Float16 *input,
                            const _Float16 *scale, const _Float16 *bias,
                            _Float16 *output,
                            int N, int C, int H, int W) {
    const int HW = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const _Float16 s = scale[c];
            const float b = (float)bias[c];
            const size_t base = (size_t)(n * C + c) * HW;
            const _Float16 *ip = input + base;
            _Float16 *op = output + base;
            int i = 0;
            size_t vl;
            for (; i < HW; i += (int)vl) {
                vl = __riscv_vsetvl_e16m4(HW - i);
                vfloat16m4_t v = __riscv_vle16_v_f16m4(ip + i, vl);
                vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(b, vl);
                acc = __riscv_vfwmacc_vf_f32m8(acc, s, v, vl);
                __riscv_vse16_v_f16m4(op + i,
                                      __riscv_vfncvt_f_f_w_f16m4(acc, vl), vl);
            }
        }
    }
}
