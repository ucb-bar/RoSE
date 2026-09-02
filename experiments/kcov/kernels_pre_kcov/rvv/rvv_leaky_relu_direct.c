/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 leaky_relu. For 0 <= slope <= 1 (the usual
   0.01), out = max(x, x*slope): x>=0 -> x, x<0 -> x*slope; bit-identical
   to the branch form since the negative branch is the same multiply. */

void kernel_leaky_relu(const float *input, float *output,
                       int n, float negative_slope) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        vfloat32m8_t s = __riscv_vfmul_vf_f32m8(v, negative_slope, vl);
        v = __riscv_vfmax_vv_f32m8(v, s, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
