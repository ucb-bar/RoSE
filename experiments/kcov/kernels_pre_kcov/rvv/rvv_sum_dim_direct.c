/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 sum-reduction over the middle axis of a
   logically [outer, reduce, inner] tensor. Vectorize the free (inner) axis:
   loads are contiguous and each lane accumulates r=0..reduce-1 in the same
   order as the scalar reference, so results match bit-for-bit; declared
   numeric_drift to be safe about fp reduction order. */
#include <riscv_vector.h>

void kernel_sum_dim(const float *input, float *output,
                    int outer, int reduce, int inner) {
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
            __riscv_vse32_v_f32m8(out + i, acc, vl);
            i += (int)vl;
        }
    }
}
