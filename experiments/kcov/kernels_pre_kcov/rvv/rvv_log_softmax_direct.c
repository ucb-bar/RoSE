/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 row-wise log-softmax over the last axis of an
   [M, K] tensor (numerically-stable log-sum-exp form):
     m_i          = max_k input[i, k]
     lse          = m_i + logf(sum_k expf(input[i, k] - m_i))
     output[i, k] = input[i, k] - lse
   Per row: (a) vectorized max-reduce (vfredmax), (b) sum of expf(x - maxv)
   via a Cephes-style vectorized expf + tree reduction (vfredusum), scalar
   logf on the single row sum, (c) vectorized subtract of the scalar lse.
   fp reduction order differs from the scalar reference => numeric_drift;
   agrees to fp32 tolerance. */
#include <math.h>
#include <riscv_vector.h>

/* Vectorized expf on f32m8 (Cephes single-precision), same core as the
   curated sigmoid/tanh/softmax kernels. Accurate to <1e-6 rel. */
static inline vfloat32m8_t rvv_exp_ps_logsoftmax(vfloat32m8_t x, size_t vl) {
    x = __riscv_vfmin_vf_f32m8(x, 88.3762626647949f, vl);
    x = __riscv_vfmax_vf_f32m8(x, -88.3762626647949f, vl);
    vfloat32m8_t fx = __riscv_vfmul_vf_f32m8(x, 1.44269504088896341f, vl);
    vint32m8_t n = __riscv_vfcvt_x_f_v_i32m8(fx, vl);
    fx = __riscv_vfcvt_f_x_v_f32m8(n, vl);
    vfloat32m8_t t = __riscv_vfmul_vf_f32m8(fx, 0.693359375f, vl);
    x = __riscv_vfsub_vv_f32m8(x, t, vl);
    t = __riscv_vfmul_vf_f32m8(fx, -2.12194440e-4f, vl);
    x = __riscv_vfsub_vv_f32m8(x, t, vl);
    vfloat32m8_t z = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(1.9875691500E-4f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 1.3981999507E-3f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 8.3334519073E-3f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 4.1665795894E-2f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 1.6666665459E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, 5.0000001201E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, z, vl);
    y = __riscv_vfadd_vv_f32m8(y, x, vl);
    y = __riscv_vfadd_vf_f32m8(y, 1.0f, vl);
    n = __riscv_vadd_vx_i32m8(n, 127, vl);
    n = __riscv_vsll_vx_i32m8(n, 23, vl);
    vfloat32m8_t pow2n = __riscv_vreinterpret_v_i32m8_f32m8(n);
    y = __riscv_vfmul_vv_f32m8(y, pow2n, vl);
    return y;
}

void kernel_log_softmax(const float *input, float *output, int M, int K) {
    for (int m = 0; m < M; m++) {
        const float *in = input + (long)m * K;
        float *out = output + (long)m * K;

        /* (a) row max */
        vfloat32m1_t vmax = __riscv_vfmv_s_f_f32m1(in[0], 1);
        for (int k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m8(K - k);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(in + k, vl);
            vmax = __riscv_vfredmax_vs_f32m8_f32m1(v, vmax, vl);
            k += (int)vl;
        }
        float maxv = __riscv_vfmv_f_s_f32m1_f32(vmax);

        /* (b) sum of exp(x - maxv) */
        vfloat32m1_t vsum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (int k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m8(K - k);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(in + k, vl);
            v = __riscv_vfsub_vf_f32m8(v, maxv, vl);
            v = rvv_exp_ps_logsoftmax(v, vl);
            vsum = __riscv_vfredusum_vs_f32m8_f32m1(v, vsum, vl);
            k += (int)vl;
        }
        float sum = __riscv_vfmv_f_s_f32m1_f32(vsum);
        float lse = maxv + logf(sum);

        /* (c) output = input - lse */
        for (int k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m8(K - k);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(in + k, vl);
            v = __riscv_vfsub_vf_f32m8(v, lse, vl);
            __riscv_vse32_v_f32m8(out + k, v, vl);
            k += (int)vl;
        }
    }
}
