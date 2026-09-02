/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 softsign = x / (1 + |x|). IEEE fdiv, no
   transcendental, bit-identical to the scalar reference. */

void kernel_softsign(const float *input, float *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        vfloat32m8_t ax = __riscv_vfabs_v_f32m8(v, vl);
        vfloat32m8_t d = __riscv_vfadd_vf_f32m8(ax, 1.0f, vl);
        v = __riscv_vfdiv_vv_f32m8(v, d, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
