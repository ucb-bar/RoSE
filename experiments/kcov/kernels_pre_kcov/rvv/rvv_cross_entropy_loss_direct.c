/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 cross-entropy loss (F.cross_entropy,
   reduction=mean; KernelBench 95). logits is [N, C]; per row the max is a
   vfredmax reduction, the log-sum-exp uses a vectorized Cephes single-
   precision expf (range-reduce to [-ln2/2, ln2/2], degree-5 minimax poly,
   ldexp via integer exponent build) summed with vfredusum, then
   loss_i = (max + log(sum)) - logits[i, t_i]; output is mean over N. The
   vector exp approx + tree reduction diverge from the scalar reference ->
   numeric_drift. */
#include <math.h>
#include <riscv_vector.h>

/* Vectorized expf on f32m8 (Cephes single-precision). */
static inline vfloat32m8_t rvv_exp_ps_ce(vfloat32m8_t x, size_t vl) {
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

void kernel_cross_entropy_loss(const float *logits, const float *targets,
                               float *output, int N, int C) {
    float total = 0.0f, kc = 0.0f; /* scalar Kahan over the N rows */
    for (int i = 0; i < N; i++) {
        const float *row = logits + (long)i * C;
        /* row max */
        vfloat32m1_t vmax = __riscv_vfmv_s_f_f32m1(row[0], 1);
        for (int c = 0; c < C; ) {
            size_t vl = __riscv_vsetvl_e32m8(C - c);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(row + c, vl);
            vmax = __riscv_vfredmax_vs_f32m8_f32m1(v, vmax, vl);
            c += (int)vl;
        }
        float maxv = __riscv_vfmv_f_s_f32m1_f32(vmax);
        /* sum of exp(row - max) */
        vfloat32m1_t vsum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (int c = 0; c < C; ) {
            size_t vl = __riscv_vsetvl_e32m8(C - c);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(row + c, vl);
            v = __riscv_vfsub_vf_f32m8(v, maxv, vl);
            v = rvv_exp_ps_ce(v, vl);
            vsum = __riscv_vfredusum_vs_f32m8_f32m1(v, vsum, vl);
            c += (int)vl;
        }
        float sum = __riscv_vfmv_f_s_f32m1_f32(vsum);
        float lse = maxv + logf(sum);
        int t = (int)(targets[i] + 0.5f);
        float yk = (lse - row[t]) - kc;
        float tk = total + yk;
        kc = (tk - total) - yk;
        total = tk;
    }
    output[0] = total / (float)N;
}
