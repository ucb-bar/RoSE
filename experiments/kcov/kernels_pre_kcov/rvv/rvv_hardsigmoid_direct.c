/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 hardsigmoid = clamp(x/6 + 0.5, 0, 1).
   The linear form hits 0 at x=-3 and 1 at x=+3, so a clamp reproduces
   the piecewise reference exactly. */

void kernel_hardsigmoid(const float *input, float *output, int n) {
    const float inv6 = 1.0f / 6.0f;
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        v = __riscv_vfmul_vf_f32m8(v, inv6, vl);
        v = __riscv_vfadd_vf_f32m8(v, 0.5f, vl);
        v = __riscv_vfmax_vf_f32m8(v, 0.0f, vl);
        v = __riscv_vfmin_vf_f32m8(v, 1.0f, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
