/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 RMS normalization over the reduce axis of a
   logically [outer, reduce, inner] tensor (KernelBench 36: dim=1, keepdim).
   Vectorize the free inner axis so loads are contiguous and each lane keeps
   its own sum-of-squares across r=0..reduce-1 (same order as the scalar
   reference):
     rms[i] = sqrt((sum_r x^2)/reduce + eps)
     out    = x / rms
   Pass 1 accumulates per-lane ssq with vfmacc; the mean/eps/rsqrt is applied
   per lane (vfsqrt vector intrinsic, no libm); pass 2 scales each element by
   1/rms. Declared numeric_drift for fp reduction-order safety. */
#include <riscv_vector.h>

void kernel_rms_norm(const float *input, float *output,
                     int outer, int reduce, int inner, float eps) {
    float inv_r = 1.0f / (float)reduce;
    for (int o = 0; o < outer; o++) {
        const float *base = input + (long)o * reduce * inner;
        float *obase = output + (long)o * reduce * inner;
        for (int i = 0; i < inner; ) {
            size_t vl = __riscv_vsetvl_e32m8(inner - i);
            vfloat32m8_t ssq = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int r = 0; r < reduce; r++) {
                vfloat32m8_t v =
                    __riscv_vle32_v_f32m8(base + (long)r * inner + i, vl);
                ssq = __riscv_vfmacc_vv_f32m8(ssq, v, v, vl);
            }
            /* ms = ssq/reduce + eps ; rms = sqrt(ms) ; inv = 1/rms */
            vfloat32m8_t ms = __riscv_vfmul_vf_f32m8(ssq, inv_r, vl);
            ms = __riscv_vfadd_vf_f32m8(ms, eps, vl);
            vfloat32m8_t rms = __riscv_vfsqrt_v_f32m8(ms, vl);
            vfloat32m8_t inv = __riscv_vfrdiv_vf_f32m8(rms, 1.0f, vl);
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
