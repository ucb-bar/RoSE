/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: vectorized RVV fp32 KL divergence, reduction='batchmean'
   (F.kl_div; KernelBench 98). The first arg is already log(pred). Over the
   flat [N, C] tensor: contrib = t*(log(t) - log_input), with t*log(t) taken
   as 0 where t<=0 (masked via vmfgt + vmerge, which also discards the NaN the
   log approx yields there). log(t) uses a vectorized Cephes single-precision
   logf; the sum is a vfredusum tree-reduction (numeric_drift vs the scalar
   double reference), scaled by 1/N. */
#include <math.h>
#include <riscv_vector.h>

/* Vectorized natural log on f32m8 (Cephes single-precision); x<=0 -> NaN. */
static inline vfloat32m8_t rvv_log_ps_kldiv(vfloat32m8_t x, size_t vl) {
    vbool4_t invalid = __riscv_vmfle_vf_f32m8_b4(x, 0.0f, vl);
    x = __riscv_vfmax_vf_f32m8(x, 1.17549435e-38f, vl); /* min normal */
    vuint32m8_t xu = __riscv_vreinterpret_v_f32m8_u32m8(x);
    vuint32m8_t eu = __riscv_vsrl_vx_u32m8(xu, 23, vl);
    xu = __riscv_vand_vx_u32m8(xu, 0x007fffffu, vl);
    xu = __riscv_vor_vx_u32m8(xu, 0x3f000000u, vl); /* mantissa in [0.5,1) */
    x = __riscv_vreinterpret_v_u32m8_f32m8(xu);
    vint32m8_t ei = __riscv_vreinterpret_v_u32m8_i32m8(eu);
    ei = __riscv_vsub_vx_i32m8(ei, 0x7f, vl);
    vfloat32m8_t e = __riscv_vfcvt_f_x_v_f32m8(ei, vl);
    e = __riscv_vfadd_vf_f32m8(e, 1.0f, vl);
    vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(x, 0.707106781186547524f, vl);
    vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
    vfloat32m8_t tmp = __riscv_vmerge_vvm_f32m8(zero, x, mask, vl);
    x = __riscv_vfsub_vf_f32m8(x, 1.0f, vl);
    vfloat32m8_t onem = __riscv_vfmerge_vfm_f32m8(zero, 1.0f, mask, vl);
    e = __riscv_vfsub_vv_f32m8(e, onem, vl);
    x = __riscv_vfadd_vv_f32m8(x, tmp, vl);
    vfloat32m8_t z = __riscv_vfmul_vv_f32m8(x, x, vl);
    vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(7.0376836292E-2f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -1.1514610310E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  1.1676998740E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -1.2420140846E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  1.4249322787E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -1.6668057665E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  2.0000714765E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y, -2.4999993993E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl); y = __riscv_vfadd_vf_f32m8(y,  3.3333331174E-1f, vl);
    y = __riscv_vfmul_vv_f32m8(y, x, vl);
    y = __riscv_vfmul_vv_f32m8(y, z, vl);
    tmp = __riscv_vfmul_vf_f32m8(e, -2.12194440e-4f, vl);
    y = __riscv_vfadd_vv_f32m8(y, tmp, vl);
    tmp = __riscv_vfmul_vf_f32m8(z, 0.5f, vl);
    y = __riscv_vfsub_vv_f32m8(y, tmp, vl);
    x = __riscv_vfadd_vv_f32m8(x, y, vl);
    tmp = __riscv_vfmul_vf_f32m8(e, 0.693359375f, vl);
    x = __riscv_vfadd_vv_f32m8(x, tmp, vl);
    x = __riscv_vfmerge_vfm_f32m8(x, NAN, invalid, vl);
    return x;
}

void kernel_kldiv_loss(const float *log_input, const float *target,
                       float *output, int N, int C) {
    long total_n = (long)N * C;
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    vfloat32m8_t comp = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    long i = 0;
    for (; i + (long)vlmax <= total_n; i += (long)vlmax) {
        vfloat32m8_t t = __riscv_vle32_v_f32m8(target + i, vlmax);
        vfloat32m8_t li = __riscv_vle32_v_f32m8(log_input + i, vlmax);
        vfloat32m8_t logt = rvv_log_ps_kldiv(t, vlmax); /* NaN where t<=0 */
        vfloat32m8_t x = __riscv_vfmul_vv_f32m8(
            t, __riscv_vfsub_vv_f32m8(logt, li, vlmax), vlmax);
        vbool4_t pos = __riscv_vmfgt_vf_f32m8_b4(t, 0.0f, vlmax);
        vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
        x = __riscv_vmerge_vvm_f32m8(zero, x, pos, vlmax);
        /* Kahan */
        vfloat32m8_t y = __riscv_vfsub_vv_f32m8(x, comp, vlmax);
        vfloat32m8_t tt = __riscv_vfadd_vv_f32m8(sum, y, vlmax);
        comp = __riscv_vfsub_vv_f32m8(
            __riscv_vfsub_vv_f32m8(tt, sum, vlmax), y, vlmax);
        sum = tt;
    }
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    float total = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m8_f32m1(sum, z, vlmax));
    for (; i < total_n; i++) {
        float t = target[i];
        if (t > 0.0f) total += t * (logf(t) - log_input[i]);
    }
    output[0] = total / (float)N;
}
