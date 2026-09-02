/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 L1 normalization over the middle axis of a
   logically [outer, reduce, inner] tensor. Vectorize the free inner axis:
   pass 1 accumulates denom[lane]=sum_r|x|; pass 2 scales each x by the
   per-lane reciprocal. Output preserves the reduce axis (broadcast div). */
#include <riscv_vector.h>

void kernel_l1_norm(const float *input, float *output,
                    int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        const float *base = input + (long)o * reduce * inner;
        float *obase = output + (long)o * reduce * inner;
        for (int i = 0; i < inner; ) {
            size_t vl = __riscv_vsetvl_e32m8(inner - i);
            vfloat32m8_t denom = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int r = 0; r < reduce; r++) {
                vfloat32m8_t v =
                    __riscv_vle32_v_f32m8(base + (long)r * inner + i, vl);
                denom = __riscv_vfadd_vv_f32m8(
                    denom, __riscv_vfabs_v_f32m8(v, vl), vl);
            }
            vfloat32m8_t inv = __riscv_vfrdiv_vf_f32m8(denom, 1.0f, vl);
            for (int r = 0; r < reduce; r++) {
                long idx = (long)r * inner + i;
                vfloat32m8_t v = __riscv_vle32_v_f32m8(base + idx, vl);
                __riscv_vse32_v_f32m8(
                    obase + idx, __riscv_vfmul_vv_f32m8(v, inv, vl), vl);
            }
            i += (int)vl;
        }
    }
}
