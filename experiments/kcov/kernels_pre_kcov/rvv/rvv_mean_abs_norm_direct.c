/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 mean-absolute (L1-mean) normalization over the
   middle axis of a logically [outer, reduce, inner] tensor. Same dataflow as
   l1_norm but the denom is the MEAN absolute value: denom=(sum_r|x|)/reduce;
   output = x / denom, broadcast over the reduce axis. */
#include <riscv_vector.h>

void kernel_mean_abs_norm(const float *input, float *output,
                          int outer, int reduce, int inner) {
    float inv_reduce = 1.0f / (float)reduce;
    for (int o = 0; o < outer; o++) {
        const float *base = input + (long)o * reduce * inner;
        float *obase = output + (long)o * reduce * inner;
        for (int i = 0; i < inner; ) {
            size_t vl = __riscv_vsetvl_e32m8(inner - i);
            vfloat32m8_t s = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int r = 0; r < reduce; r++) {
                vfloat32m8_t v =
                    __riscv_vle32_v_f32m8(base + (long)r * inner + i, vl);
                s = __riscv_vfadd_vv_f32m8(
                    s, __riscv_vfabs_v_f32m8(v, vl), vl);
            }
            vfloat32m8_t denom = __riscv_vfmul_vf_f32m8(s, inv_reduce, vl);
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
