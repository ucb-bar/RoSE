/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp32 min-reduction over the middle axis of a
   logically [outer, reduce, inner] tensor. Mirror of max_dim with vfmin;
   seed each lane from r=0 (reduce>=1), bit-exact vs the FLT_MAX-init
   reference. */
#include <riscv_vector.h>

void kernel_min_dim(const float *input, float *output,
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
                acc = __riscv_vfmin_vv_f32m8(acc, v, vl);
            }
            __riscv_vse32_v_f32m8(out + i, acc, vl);
            i += (int)vl;
        }
    }
}
