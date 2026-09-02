/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 mean-reduction over the middle axis of a
   logically [outer, reduce, inner] tensor. Same dataflow as sum_dim
   (vectorize the free inner axis, accumulate across the reduce axis in
   reference order), then scale each lane by 1/reduce. */
#include <riscv_vector.h>

void kernel_mean_dim(const float *input, float *output,
                     int outer, int reduce, int inner) {
    float inv = 1.0f / (float)reduce;
    for (int o = 0; o < outer; o++) {
        const float *base = input + (long)o * reduce * inner;
        float *out = output + (long)o * inner;
        for (int i = 0; i < inner; ) {
            size_t vl = __riscv_vsetvl_e32m8(inner - i);
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int r = 0; r < reduce; r++) {
                vfloat32m8_t v =
                    __riscv_vle32_v_f32m8(base + (long)r * inner + i, vl);
                acc = __riscv_vfadd_vv_f32m8(acc, v, vl);
            }
            acc = __riscv_vfmul_vf_f32m8(acc, inv, vl);
            __riscv_vse32_v_f32m8(out + i, acc, vl);
            i += (int)vl;
        }
    }
}
