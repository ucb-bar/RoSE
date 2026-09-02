/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 tensor*scalar (out = in * s). Replaces the scalar
   fallback `for(i) out[i]=in[i]*s`, which measured ~19 cyc/elem on Saturn (vs
   ~1.7 for ReLU) because it never vectorized. vfmul_vf is the same fp32 multiply
   so the result is bit-identical to the scalar loop. */

void kernel_mul_scalar(const float *input, float *output, int n, float s) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        v = __riscv_vfmul_vf_f32m8(v, s, vl);
        __riscv_vse32_v_f32m8(output + i, v, vl);
    }
}
