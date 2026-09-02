/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 relu6 = min(max(x,0),6). Pure clamp. */

void kernel_relu6(const float *input, float *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        v = __riscv_vfmax_vf_f32m8(v, 0.0f, vl);
        v = __riscv_vfmin_vf_f32m8(v, 6.0f, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
