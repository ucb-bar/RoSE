/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 product-reduction over the middle axis of a
   logically [outer, reduce, inner] tensor. Vectorize the free inner axis;
   seed each lane from r=0 then multiply-fold across r=1..reduce-1 (same
   per-lane order as the reference). */
#include <riscv_vector.h>

void kernel_prod_dim(const float *input, float *output,
                     int outer, int reduce, int inner) {
    for (int o = 0; o < outer; o++) {
        const float *base = input + (long)o * reduce * inner;
        float *out = output + (long)o * inner;
        for (int i = 0; i < inner; ) {
            size_t vl = __riscv_vsetvl_e32m8(inner - i);
            vfloat32m8_t acc = __riscv_vle32_v_f32m8(base + i, vl);
            for (int r = 1; r < reduce; r++) {
                vfloat32m8_t v =
                    __riscv_vle32_v_f32m8(base + (long)r * inner + i, vl);
                acc = __riscv_vfmul_vv_f32m8(acc, v, vl);
            }
            __riscv_vse32_v_f32m8(out + i, acc, vl);
            i += (int)vl;
        }
    }
}
