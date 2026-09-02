/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 LayerNorm over the last axis of an [M,K]
   tensor. Each row is normalized independently.

   NOTE (Saturn V256D128 fp64 bug): the fp32->fp64 widening-reduction path
   miscomputes on the Saturn vector unit, so all statistics are kept in pure
   fp32.  A one-pass fp32 E[x^2]-E[x]^2 cancels and drifts past
   atol on large-K / large-magnitude rows, so we use a numerically-stable
   TWO-PASS fp32 algorithm instead:
     pass 1: mean = sum_k x / K
     pass 2: var  = sum_k (x-mean)^2 / K     (centering removes the large
             common component, so the fp32 accumulation of the centered
             squares does not catastrophically cancel)
   Each pass accumulates with a plain fp32 reduction (vfredusum f32m8->f32m1,
   which is CONFIRMED correct on Saturn); no fp64 vector ops are used.
   inv=1/sqrt(var+eps) scalar, then per-element affine
     out[k] = (x[k]-mean)*inv*gamma[k] + beta[k]
   with gamma/beta loaded as vectors. Reduction order differs from the scalar
   loop -> numeric_drift. */
#include <math.h>
#include <riscv_vector.h>

void kernel_layer_norm(const float *input, const float *gamma,
                       const float *beta, float *output,
                       int M, int K, float eps) {
    for (int m = 0; m < M; m++) {
        const float *row = input + (long)m * K;
        float *orow = output + (long)m * K;

        /* pass 1: mean = sum(x)/K, fp32 reduction (no fp64) */
        vfloat32m1_t vs = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (int k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m8(K - k);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(row + k, vl);
            vs = __riscv_vfredusum_vs_f32m8_f32m1(v, vs, vl);
            k += (int)vl;
        }
        float mean = __riscv_vfmv_f_s_f32m1_f32(vs) / (float)K;

        /* pass 2: var = sum((x-mean)^2)/K, fp32 reduction (no fp64) */
        vfloat32m1_t vq = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (int k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m8(K - k);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(row + k, vl);
            vfloat32m8_t d = __riscv_vfsub_vf_f32m8(v, mean, vl);
            vfloat32m8_t sq = __riscv_vfmul_vv_f32m8(d, d, vl);
            vq = __riscv_vfredusum_vs_f32m8_f32m1(sq, vq, vl);
            k += (int)vl;
        }
        float var = __riscv_vfmv_f_s_f32m1_f32(vq) / (float)K;
        float inv = 1.0f / sqrtf(var + eps);

        /* pass 3: normalize + affine */
        for (int k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m8(K - k);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(row + k, vl);
            vfloat32m8_t g = __riscv_vle32_v_f32m8(gamma + k, vl);
            vfloat32m8_t b = __riscv_vle32_v_f32m8(beta + k, vl);
            vfloat32m8_t d = __riscv_vfsub_vf_f32m8(v, mean, vl);
            d = __riscv_vfmul_vf_f32m8(d, inv, vl);      /* (x-mean)*inv */
            /* out = b + d*g */
            vfloat32m8_t o = __riscv_vfmacc_vv_f32m8(b, d, g, vl);
            __riscv_vse32_v_f32m8(orow + k, o, vl);
            k += (int)vl;
        }
    }
}
