/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 global Frobenius normalization over a flat
   n-element tensor. Pass 1 reduces sum-of-squares with vfredusum (a tree
   reduction whose order differs from the scalar reference -> numeric_drift),
   sqrt on the scalar, then pass 2 scales every element by 1/denom. */
#include <math.h>
#include <riscv_vector.h>

void kernel_frobenius_norm(const float *input, float *output, int n) {
    vfloat32m1_t vacc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        vfloat32m8_t sq = __riscv_vfmul_vv_f32m8(v, v, vl);
        vacc = __riscv_vfredusum_vs_f32m8_f32m1(sq, vacc, vl);
        i += (int)vl;
    }
    float ssq = __riscv_vfmv_f_s_f32m1_f32(vacc);
    float inv = 1.0f / sqrtf(ssq);
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(input + i, vl);
        __riscv_vse32_v_f32m8(output + i, __riscv_vfmul_vf_f32m8(v, inv, vl), vl);
        i += (int)vl;
    }
}
